import datetime
import os
from collections import defaultdict
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

TASK_CATEGORIES = [
    "Anomaly Recognition",
    "Event Recognition",
    "Attribute Recognition",
    "Human Interaction",
    "Temporal Localization",
    "Video Emotions",
    "Event Sorting",
    "Hallucination",
    "Text and Diagram Understanding",
    "Attribute Reasoning",
    "Causal Reasoning",
    "Object Counting",
    "Action Counting",
    "Temporal Prediction",
    "Emotion Change",
    "Audio Counting",
    "Scene Recognition",
    "Human-object Interaction",
    "Human Emotions",
    "Object State Change",
    "Relation Reasoning",
    "Spatial Relation",
    "Audio Source Localization",
    "Audio Recognition",
    "Object Existence Recognition",
    "Audio Change",
]

DOMAINS = [
    "Tech & Science",
    "Culture & Politics",
    "Daily Life",
    "Film & TV",
    "Performance",
    "Games",
    "Sports",
    "Music",
]

BASE_SYS = "Carefully watch this video and pay attention to every detail. "
SYS = BASE_SYS + "Based on your observations, select the best option that accurately addresses the question."

FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

FRAMES_TMPL_SUB = """
These are the frames of a video. \
This video's subtitles are listed below:
"{}"
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

FRAMES_TMPL_AUDIO = """
These are the frames of a video and the corresponding audio. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

with open(Path(__file__).parent / "worldsense.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(os.path.expanduser(subtitle_path), "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def worldsense_doc_to_visual(doc):
    """
    Return the path to the video only
    """
    video_paths = []
    # Get the video
    abs_video_path = os.path.join(cache_dir, doc["video_path"])
    abs_video_path = os.path.expanduser(abs_video_path)
    if os.path.exists(abs_video_path):
        video_paths.append(abs_video_path)
    else:
        print(f"Video path does not exist: {abs_video_path}")
    return video_paths


def worldsense_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    """
    Process the document to a prompt for video + subtitle inputs
    """
    abs_subtitle_path = os.path.expanduser(os.path.join(cache_dir, doc["subtitle_path"]))
    if os.path.exists(abs_subtitle_path):
        subtitle = open(abs_subtitle_path).readlines()
    else:
        print(f"Subtitle path does not exist: {abs_subtitle_path}")
        subtitle = ""
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        if "frame_num" in lmms_eval_specific_kwargs:
            frame_num = lmms_eval_specific_kwargs["frame_num"]
            video_path = os.path.expanduser(os.path.join(cache_dir, doc["video_path"]))
            subtitle_path = os.path.expanduser(os.path.join(cache_dir, doc["subtitle_path"]))
            subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
            if frame_num == -1:
                frame_num = total_frame
            uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

            subtitle_by_frame_idx = []
            for frame_idx in uniform_sampled_frames:
                for idx, title in enumerate(subtitle_by_frame):
                    if frame_idx < title[1] and frame_idx >= title[0]:
                        subtitle_by_frame_idx.append(idx)
            subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))
            subtitle_by_frame_idx.sort()  # Reorder the subtitle by frame index
            textlist = []
            for idx in subtitle_by_frame_idx:
                raw_text = subtitle_by_frame[idx][2]
                textlist.append(raw_text)
            subtitle_text = "\n".join(textlist)
    subtitle = subtitle_text
    subtitle_option_prompt = FRAMES_TMPL_SUB.format(subtitle)
    fullprompt = [SYS, subtitle_option_prompt]
    fullprompt.append(doc["question"] + "\n")
    for op in doc["candidates"]:
        fullprompt.append(op + "\n")
    return "".join(fullprompt)


import ast
import re


def _extract_candidates_by_label(raw_text):
    """
    Fallback parser for malformed candidate strings.
    Expected option format: "A. ...", "B. ...", "C. ...", "D. ...".
    """
    s = str(raw_text).replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    marker_re = re.compile(r'(?:(?<=^)|(?<=[\s,\[\'"]))([A-D])\.\s*')
    matches = list(marker_re.finditer(s))
    if not matches:
        return []

    parsed = []
    seen = set()
    for i, m in enumerate(matches):
        label = m.group(1)
        if label in seen:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        text = s[start:end].strip(" \t\r\n'\",]")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parsed.append(f"{label}. {text}")
            seen.add(label)
    return parsed


def _build_option_map(options):
    option_map = {}
    for i, opt in enumerate(options):
        text = str(opt).strip()
        m = re.match(r"^\s*([A-D])\.\s*(.*)$", text)
        if m:
            label = m.group(1)
            content = m.group(2).strip()
        else:
            if i >= 4:
                continue
            label = ["A", "B", "C", "D"][i]
            content = text
        if label not in option_map and content:
            option_map[label] = content
    return option_map


def parse_candidates(raw):
    # already parsed
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in raw if str(x).strip()]

    s = str(raw).strip()
    if not s:
        return []

    # normalize common malformed quote/comma patterns
    fixed = s.replace("\xa0", " ")
    fixed = re.sub(r'""+', '"', fixed)
    fixed = re.sub(r"''+", "'", fixed)
    fixed = re.sub(r"(['\"])\s+(['\"])", r"\1, \2", fixed)
    fixed = re.sub(r",\s*,+", ", ", fixed)

    try:
        parsed = ast.literal_eval(fixed)
        if isinstance(parsed, str):
            parsed = _extract_candidates_by_label(parsed)
        elif isinstance(parsed, (list, tuple, np.ndarray)):
            parsed = [str(x).strip() for x in parsed if str(x).strip()]
        else:
            parsed = []
        if parsed:
            return parsed
    except (ValueError, SyntaxError):
        pass

    parsed = _extract_candidates_by_label(fixed)
    if parsed:
        return parsed

    parsed = _extract_candidates_by_label(s)
    if parsed:
        return parsed
    return [s]


def worldsense_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    candidates = parse_candidates(doc["candidates"])
    candidates_text = "\n".join(candidates)
    # Reference: https://github.com/KD-TAO/OmniZip/blob/main/eval/eval_worldsense.py#L132
    prompt = f"{question}\nOptions:\n{candidates_text}\nAnswer with the option's letter from the given choices directly."
    return prompt



def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"{choice}" in response:
            candidates.append(choice)
            ans_with_brack = True
    # e.g. "C. So, what do you think..." may falsely match "A" in other words.
    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        import random
        pred_index = random.choice(all_choices)
        # pred_index = "A"
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def parse_multi_choice_response_new(response, all_choices, index2ans=None):
    """
    More robust parsing:
    1) Prefer leading choice like "C." or "(C)"
    2) Then bracketed "(C)"
    3) Then standalone letter with word boundaries
    4) Then answer text match
    5) Fallback to random
    """
    if not response:
        import random
        return random.choice(all_choices)

    if isinstance(response, list):
        response = " ".join(str(x) for x in response)
    else:
        response = str(response)

    # 1) Leading choice (e.g., "C.", "(C)", "C)")
    m = re.match(r"^\s*\(?([A-D])\)?[\s\.\):]", response, flags=re.IGNORECASE)
    if m:
        choice = m.group(1).upper()
        if choice in all_choices:
            return choice

    # 2) Any bracketed choice
    m = re.search(r"\(([A-D])\)", response, flags=re.IGNORECASE)
    if m:
        choice = m.group(1).upper()
        if choice in all_choices:
            return choice

    # 3) Standalone letter with boundaries (avoid "Any" -> "A")
    m = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", response, flags=re.IGNORECASE)
    if m:
        choice = m.group(1).upper()
        if choice in all_choices:
            return choice

    # 4) Content match
    if len(response.split()) > 5 and index2ans is not None:
        for index, ans in index2ans.items():
            if index in all_choices and ans and ans.lower() in response.lower():
                return index

    import random
    print(f"\n func parse_multi_choice_response_new(): random choice: {random.choice(all_choices)}")
    return random.choice(all_choices)


def worldsense_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case av_odyssey score), value: metric value
    """
    answers = results
    predicted_answer = None
    resp_text = answers[0].strip() if answers and answers[0] else ""
    for opt in ["A", "B", "C", "D", "E", "F"]:
        if resp_text.upper().strip().startswith(opt):
            predicted_answer = opt
            break
    if predicted_answer is None and len(resp_text) > 0:
        predicted_answer = resp_text[0].upper()

    gt_answer = doc["answer"]
    assert gt_answer in ["A", "B", "C", "D"]
    score = 1.0 if predicted_answer == gt_answer else 0.0
    category = doc["task_type"]
    domain = doc["domain"]
    duration = doc["duration"]
    audio_class = doc["audio_class"]  # a list of audios
    key_name = "worldsense_score"
    return {key_name: {"question_id": doc["index"], "category": category, "score": score, "domain": domain, "duration": duration, "audio_class": audio_class}}


def _parse_audio_classes(audio_class_field):
    """Normalize audio_class field to list[str]."""
    import re
    # Already iterable (list/tuple/ndarray): convert to list.
    if not isinstance(audio_class_field, str):
        try:
            return list(audio_class_field)
        except TypeError:
            return [str(audio_class_field)]

    s = audio_class_field.strip()

    # Typical form: "['Speech' 'Music' 'Event']". Extract quoted tokens with regex.
    matches = re.findall(r"'([^']+)'", s)
    if matches:
        return matches

    # Fallback: split on whitespace if no quoted tokens.
    return [p for p in s.strip("[] ").split() if p]

def worldsense_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(dict)
    domain2score = defaultdict(dict)
    duration2score = defaultdict(dict)
    audio2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        category = result["category"]
        domain = result["domain"]
        duration = result["duration"]
        # audio_classes = result["audio_class"]
        audio_classes = _parse_audio_classes(result["audio_class"])
        if question_id not in category2score[category]:
            category2score[category][question_id] = []
        if question_id not in domain2score[domain]:
            domain2score[domain][question_id] = []
        if question_id not in duration2score[duration]:
            duration2score[duration][question_id] = []
        for audio in audio_classes:
            if question_id not in audio2score[audio]:
                audio2score[audio][question_id] = []
            audio2score[audio][question_id].append(score)
        category2score[category][question_id].append(score)
        domain2score[domain][question_id].append(score)
        duration2score[duration][question_id].append(score)

    # Calculate the average score for each category
    category_avg_scores = {}
    total_score = 0
    total_questions = 0

    # For task category
    for category, questions in category2score.items():
        category_total = 0  # Category total score
        for question_id, score in questions.items():
            category_total += score[0]
        category_avg_scores[category] = category_total / len(questions) * 100.0
        total_score += category_total
        total_questions += len(questions)
    for category, avg_score in category_avg_scores.items():
        eval_logger.info(f"Evaluation on Task Categories: {category}: {avg_score:.2f}")

    # For domain categories
    domain_avg_scores = {}
    for domain, questions in domain2score.items():
        domain_total = 0
        for question_id, score in questions.items():
            domain_total += score[0]
        domain_avg_scores[domain] = domain_total / len(questions) * 100.0
    for domain, avg_score in domain_avg_scores.items():
        eval_logger.info(f"Evaluation on Task Domains: {domain}: {avg_score:.2f}")

    # For duration categories
    duration_avg_scores = {}
    for duration, questions in duration2score.items():
        duration_total = 0
        for question_id, score in questions.items():
            duration_total += score[0]
        duration_avg_scores[duration] = duration_total / len(questions) * 100.0
    for duration, avg_score in duration_avg_scores.items():
        eval_logger.info(f"Evaluation on Video Duration: {duration}: {avg_score:.2f}")

    # For audio classes
    audio_avg_scores = {}
    for audio, questions in audio2score.items():
        audio_total = 0
        for question_id, score in questions.items():
            audio_total += score[0]
        audio_avg_scores[audio] = audio_total / len(questions) * 100.0
    for audio, avg_score in audio_avg_scores.items():
        eval_logger.info(f"Evaluation on Audio Classes: {audio}: {avg_score:.2f}")

    overall_avg_score = total_score / total_questions * 100.0
    eval_logger.info(f"Overall performance (across all questions): {overall_avg_score:.2f}")

    return overall_avg_score
