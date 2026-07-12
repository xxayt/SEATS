from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm
from models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from transformers import Qwen3OmniMoeProcessor
import os

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import split_audio

try:
    from lmms_eval.models.model_utils.qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; Please install it via `pip install qwen-omni-utils[decord]`")
from baselines.utils import apply_zip_method_patch


@register_model("qwen3_omni_zip")
class Qwen3_Omni_Zip(lmms):
    """
    Qwen3-Omni-30B-A3B-Instruct
    "https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "flash_attention_2",
        max_num_frames: int = 768,
        use_custom_video_loader: Optional[bool] = False,
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        max_frames: Optional[int] = 768,
        fps: Optional[int] = 2,
        nframes: Optional[int] = None,
        min_pixels: Optional[int] = 128 * 32 * 32,
        max_pixels: Optional[int] = 144 * 32 * 32,
        total_pixels: Optional[int] = 24576 * 32 * 32,
        video_ratio: Optional[float] = 1.0,
        audio_ratio: Optional[float] = 1.0,
        config_path: Optional[str] = None,
        system_prompt: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.fps = fps
        self.nframes = nframes
        self.max_frames = max_frames
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.use_custom_video_loader = use_custom_video_loader
        self.max_num_frames = max_num_frames
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map, attn_implementation=attn_implementation).eval()

        # ===== apply method-specific patch =====
        self.method_name = os.environ.get("METHOD", "full_tokens")
        print(f"Use core unified model for Inference (method={self.method_name}, config={config_path})")
        apply_zip_method_patch(
            self._model,
            method_name=self.method_name,
            video_ratio=video_ratio,
            audio_ratio=audio_ratio,
            config_path=config_path,
            pretrained=pretrained,
        )

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(pretrained)
        self.processor.feature_extractor.chunk_length = 300
        self.processor.feature_extractor.n_samples = 300 * 16000
        self.processor.feature_extractor.nb_max_frames = 300 * 16000 // 160
        self._tokenizer = self.processor.tokenizer

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._model.disable_talker()
        self.system_prompt = system_prompt

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_Omni_Zip")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def resample_audio(self, audio: np.ndarray, current_sample_rate: int) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            return audio
        if audio.ndim == 2:
            axis = 0 if audio.shape[0] <= audio.shape[1] else 1
            audio = np.mean(audio, axis=axis)
        elif audio.ndim > 2:
            audio = audio.mean(axis=tuple(range(audio.ndim - 1)))
        audio = audio.astype(np.float32)
        if current_sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=16000)
            audio = audio.astype(np.float32)
        return audio

    def _decode_audio(self, audio_obj) -> dict:
        if isinstance(audio_obj, dict) and "array" in audio_obj and "sampling_rate" in audio_obj:
            return audio_obj
        type_name = type(audio_obj).__name__
        if type_name != "AudioDecoder":
            raise ValueError(f"Unknown audio type: {type(audio_obj)}")
        return self._decode_audio_decoder(audio_obj)

    def _decode_audio_decoder(self, audio_obj) -> dict:
        if hasattr(audio_obj, "get_all_samples"):
            decoded_audio = audio_obj.get_all_samples()
            audio_array = self._extract_audio_array(decoded_audio)
            sampling_rate = self._extract_sampling_rate(decoded_audio, audio_obj)
            return {"array": audio_array, "sampling_rate": sampling_rate}
        if hasattr(audio_obj, "decode"):
            decoded_audio = audio_obj.decode()
            if isinstance(decoded_audio, dict):
                return decoded_audio
            if hasattr(decoded_audio, "array") and hasattr(decoded_audio, "sampling_rate"):
                return {"array": decoded_audio.array, "sampling_rate": decoded_audio.sampling_rate}
        if hasattr(audio_obj, "array") and hasattr(audio_obj, "sampling_rate"):
            return {"array": audio_obj.array, "sampling_rate": audio_obj.sampling_rate}
        raise ValueError("Could not decode AudioDecoder object")

    def _extract_audio_array(self, decoded_audio):
        if hasattr(decoded_audio, "samples"):
            audio_array = decoded_audio.samples
        elif hasattr(decoded_audio, "array"):
            audio_array = decoded_audio.array
        elif hasattr(decoded_audio, "data"):
            audio_array = decoded_audio.data
        else:
            audio_array = decoded_audio
        if hasattr(audio_array, "cpu") and hasattr(audio_array, "numpy"):
            return audio_array.cpu().numpy()
        return audio_array

    def _extract_sampling_rate(self, decoded_audio, audio_obj) -> int:
        if hasattr(decoded_audio, "sample_rate"):
            return decoded_audio.sample_rate
        if hasattr(decoded_audio, "sampling_rate"):
            return decoded_audio.sampling_rate
        if hasattr(audio_obj, "metadata") and audio_obj.metadata:
            if hasattr(audio_obj.metadata, "sample_rate"):
                return audio_obj.metadata.sample_rate
            if isinstance(audio_obj.metadata, dict) and "sample_rate" in audio_obj.metadata:
                return audio_obj.metadata["sample_rate"]
        return 16000

    def _check_if_video_has_audio(self, video_path):
        clip = VideoFileClip(video_path)
        return clip.audio is not None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        current_use_audio = False

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            should_flatten = True
            if visuals and isinstance(visuals[0], (list, tuple)) and len(visuals[0]) > 1:
                first_visual = visuals[0]
                has_audio = any(isinstance(v, dict) or type(v).__name__ == "AudioDecoder" for v in first_visual)
                has_image = any(isinstance(v, Image.Image) for v in first_visual)
                has_video = any(isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")) for v in first_visual)
                if sum([has_audio, has_image, has_video]) > 1:
                    should_flatten = False
            if should_flatten:
                visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]
            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]

            message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt + (" Please analyze the video carefully and select the most appropriate answer from the given options." if ("daily" in task or "worldsense" in task) else "")}]}]
            for i, context in enumerate(contexts):
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):  # Video file
                        current_use_audio = self._check_if_video_has_audio(visual)
                        if self.use_custom_video_loader:
                            message.append({"role": "user", "content": [{"type": "video", "video": visual}, {"type": "text", "text": context}]})
                        else:  # Model video loader
                            video_msg = {"type": "video", "video": visual}
                            if self.nframes is not None:
                                video_msg["nframes"] = self.nframes
                            elif self.fps is not None:
                                video_msg["fps"] = self.fps
                            if self.max_frames is not None:
                                video_msg["max_frames"] = self.max_frames
                            if self.min_pixels is not None:
                                video_msg["min_pixels"] = self.min_pixels
                            if self.max_pixels is not None:
                                video_msg["max_pixels"] = self.max_pixels
                            if self.total_pixels is not None:
                                video_msg["total_pixels"] = self.total_pixels
                            message.append({"role": "user", "content": [video_msg, {"type": "text", "text": context}]})

                    elif isinstance(visual, Image.Image):  # Single image
                        message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        single_message = {"role": "user", "content": []}
                        for v in visual:
                            single_message["content"].append({"type": "image", "image": v})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    elif isinstance(visual, dict) or type(visual).__name__ == "AudioDecoder":  # Single audio
                        current_use_audio = True
                        audio_dict = self._decode_audio(visual)
                        audio = self.resample_audio(audio_dict["array"], audio_dict["sampling_rate"])
                        audio_splits = split_audio(audio, 4800000)
                        single_message = {"role": "user", "content": []}
                        for split_audio_chunk in audio_splits:
                            single_message["content"].append({"type": "audio", "audio": split_audio_chunk})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    elif isinstance(visual, (list, tuple)) and len(visual) > 0:
                        single_message = {"role": "user", "content": []}
                        for v in visual:
                            if isinstance(v, Image.Image):
                                single_message["content"].append({"type": "image", "image": v})
                            elif isinstance(v, dict) or type(v).__name__ == "AudioDecoder":
                                current_use_audio = True
                                audio_dict = self._decode_audio(v)
                                audio = self.resample_audio(audio_dict["array"], audio_dict["sampling_rate"])
                                audio_splits = split_audio(audio, 4800000)
                                for audio_chunk in audio_splits:
                                    single_message["content"].append({"type": "audio", "audio": audio_chunk})
                            elif isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                                current_use_audio = self._check_if_video_has_audio(v)
                                single_message["content"].append({"type": "video", "video": v})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    else:
                        raise ValueError(f"Unknown visual type: {type(visual)}")
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(message, image_patch_size=16, use_audio_in_video=current_use_audio)
            # frame & audio shape
            nf, c, resized_H, resized_W = videos[0].shape if videos is not None else (0, 0, 0, 0)
            audio_shape = audios[0].shape if audios is not None else (-1, )  # (duration*16k, )
            print(f"generate_video: num_frames={nf}, frame_size=({resized_H}x{resized_W}) with {resized_H*resized_W//32//32} tokens. audio_shape={audio_shape}")
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                    return_tensors="pt", padding=True, use_audio_in_video=current_use_audio)

            if self.device_map == "auto":
                inputs = inputs.to("cuda").to(self.model.dtype)
            else:
                inputs = inputs.to(self.model.device).to(self.model.dtype)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            try:
                cont = self.model.generate(
                    **inputs,
                    return_audio=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    use_audio_in_video=current_use_audio,
                    thinker_do_sample=False,
                )
                if isinstance(cont, tuple):
                    cont = cont[0]
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                answer = ""
                res.append(answer)
                pbar.update(1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                continue

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
