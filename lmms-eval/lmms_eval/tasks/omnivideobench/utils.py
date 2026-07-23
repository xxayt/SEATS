import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
import ast
from loguru import logger as eval_logger

# 读取 omnivideobench.yaml 配置文件
with open(Path(__file__).parent / "omnivideobench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")  # /mnt/sh/mmvision/data/videollm/benchmarks
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])

def omnivideobench_doc_to_visual(doc):
    # /mnt/sh/mmvision/data/videollm/benchmarks/omnivideobench/videos/video_1.mp4
    """
    Return the path to the video only
    """
    video_path = os.path.join(cache_dir, doc["video"])
    if os.path.exists(video_path):
        pass
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


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


# 生成文本任务的提示信息
def omnivideobench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    # question = question + "\n" + option
    # post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    # full_prompt = option_prompt + "\n" + question + "\n" + post_prompt

    question = doc["question"]
    options = parse_candidates(doc["options"])
    options_text = "\n".join(options)
    prompt = (
        "You are given a video. Based on the content of the video, answer the following question:\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer with the option's letter directly(e.g., A, B, C, or D)."
        "If your access to the video content is limited, at least one option that is more likely than the others must be chosen."
        "Mustn't give any other reason for can not choose!"
    )
    full_prompt = prompt
    return full_prompt

# 提取回答中的字符（A, B, C, D）
def extract_characters_regex(s):
    """从回答中提取字符 A, B, C, D"""
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    # 如果（在去除前缀后）文本按空格分词超过 10 个词，且完全找不到 A/B/C/D，则视为没有明确选项信号，直接返回空串
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    # 在整段文本中查找第一个 A/B/C/D（大写）
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    # 返回匹配到的单个字符，如 'A'/'B'/'C'/'D'
    return matches[0]


def omnivideobench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case omnivideobench score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)

    # 获取任务信息
    data_dict = {
        "duration": doc["duration"],
        "video_type": doc["video_type"],
        "question_type": doc["question_type"],
        "audio_type": doc["audio_type"],
        "pred_answer": pred_ans,
        "answer": doc["correct_option"],
        "score": 1.0 if pred_ans == doc["correct_option"] else 0.0
    }
    return {f"omnivideobench_perception_score": data_dict}


def convert_duration_to_seconds(time_str):
    """
    Converts a time string in 'MM:SS' or 'HH:MM:SS' format to total seconds.
        int: The total duration in seconds.
    """
    parts = time_str.split(':')
    seconds = 0
    if len(parts) == 2:  # MM:SS format
        minutes = int(parts[0])
        seconds = int(parts[1])
        total_seconds = minutes * 60 + seconds
    else:
        raise ValueError("Invalid time format. Please use 'MM:SS' or 'HH:MM:SS'.")
    
    return total_seconds

def get_duration_type(duration):
    if ":" in duration:
        duration = convert_duration_to_seconds(duration)
    # 区分类型 (0,1] min (1,5] min (5,10] min (10,30] min
    if duration <= 60:
        return "0-1min"
    elif duration <= 300:
        return "1-5min"
    elif duration <= 600:
        return "5-10min"
    elif duration <= 1800:
        return "10-30min"
    else:
        return ">30min"


# 汇总评估结果
def omnivideobench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    video_type2score = defaultdict(list)
    question_type2score = defaultdict(list)
    audio_type2score = defaultdict(list)
    duration_type2score = defaultdict(list)
    total_score = 0
    total_questions = len(results)
    for result in results:
        score = result["score"]
        video_type = result["video_type"]
        question_type = result["question_type"]
        audio_type = result["audio_type"]
        duration_type = get_duration_type(result["duration"])
        video_type2score[video_type].append(score)
        question_type2score[question_type].append(score)
        audio_type2score[audio_type].append(score)
        duration_type2score[duration_type].append(score)
        total_score += score

    # Calculate the average score for each video_type

    # For video_task
    video_type_avg_scores = {}
    for video_type, scores in video_type2score.items():
        video_type_total = sum(scores)  # Category total score
        video_type_avg_scores[video_type] = video_type_total / len(scores) * 100.0
    for video_type, avg_score in video_type_avg_scores.items():
        eval_logger.info(f"Evaluation on video_type Categories: {video_type}: {avg_score:.2f}")

    # For question_type
    question_type_avg_scores = {}
    for question_type, scores in question_type2score.items():
        question_type_total = sum(scores)
        question_type_avg_scores[question_type] = question_type_total / len(scores) * 100.0
    for question_type, avg_score in question_type_avg_scores.items():
        eval_logger.info(f"Evaluation on question_type Domains: {question_type}: {avg_score:.2f}")

    # For audio_type
    audio_type_avg_scores = {}
    for audio_type, scores in audio_type2score.items():
        audio_type_total = sum(scores)
        audio_type_avg_scores[audio_type] = audio_type_total / len(scores) * 100.0
    for audio_type, avg_score in audio_type_avg_scores.items():
        eval_logger.info(f"Evaluation on audio_type Audio Types: {audio_type}: {avg_score:.2f}")

    # For duration_type
    duration_type_avg_scores = {}
    for duration_type, scores in duration_type2score.items():
        duration_type_total = sum(scores)
        duration_type_avg_scores[duration_type] = duration_type_total / len(scores) * 100.0
    for duration_type, avg_score in duration_type_avg_scores.items():
        eval_logger.info(f"Evaluation on duration_type Duration Types: {duration_type}: {avg_score:.2f}")

    overall_avg_score = total_score / total_questions * 100.0
    eval_logger.info(f"Overall performance (across all questions): {overall_avg_score:.2f}")

    return overall_avg_score