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

with open(Path(__file__).parent / "lvomnibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")  # /mnt/sh/mmvision/data/videollm/benchmarks
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])

def lvomnibench_doc_to_visual(doc):
    """
    Return the path to the video only
    """
    video_path = os.path.join(cache_dir, "videos", f"{doc['video_id']}.mp4")
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


def lvomnibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = parse_candidates(doc["options"])
    options_text = "\n".join(options)
    prompt = (
        f"Question: {question}\n"
        f"Options: {options_text}\n"
        "Select the best answer from the options above. "
        "Directly provide the letter representing your choice (A/B/C/D) and nothing else. "
        "Do not include the full text of the option; do not provide any explanation."
    )
    full_prompt = prompt
    return full_prompt

def extract_characters_regex(s):
    """Extract choice letter A, B, C, or D from the response."""
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

    # Long response with no A/B/C/D after prefix removal returns empty string.
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def lvomnibench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case lvomnibench score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)

    data_dict = {
        "question_id": doc["question_id"],
        "duration": doc["duration"],
        "video_category": doc["video_category"],
        "sub_category": doc["sub_category"],
        "question_type": doc["question_type"],
        "audio_type": doc["audio_type"],
        "difficulty": doc["difficulty"],
        "pred_answer": pred_ans,
        "answer": doc["answer"],
        "score": 1.0 if pred_ans == doc["answer"] else 0.0
    }
    return {f"lvomnibench_perception_score": data_dict}


def convert_duration_to_seconds(time_str):
    """
    Converts a time string in 'MM:SS' or 'HH:MM:SS' format to total seconds.
        int: The total duration in seconds.
    """
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS format
        minutes = int(parts[0])
        seconds = int(parts[1])
        total_seconds = minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS format
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Invalid time format. Please use 'MM:SS' or 'HH:MM:SS'.")

    return total_seconds

def get_duration_type(duration):
    if ":" in duration:
        duration = convert_duration_to_seconds(duration)
    # Duration buckets: (0,1], (1,5], (5,10], (10,30] minutes.
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


def lvomnibench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    video_category2score = defaultdict(dict)
    sub_category2score = defaultdict(dict)
    question_type2score = defaultdict(dict)
    audio_type2score = defaultdict(dict)
    difficulty2score = defaultdict(dict)
    duration_type2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        video_category = result["video_category"]
        sub_category = result["sub_category"]
        question_type = result["question_type"]
        audio_type = result["audio_type"]
        difficulty = result["difficulty"]
        duration_type = get_duration_type(result["duration"])
        if question_id not in video_category2score[video_category]:
            video_category2score[video_category][question_id] = []
        if question_id not in sub_category2score[sub_category]:
            sub_category2score[sub_category][question_id] = []
        if question_id not in question_type2score[question_type]:
            question_type2score[question_type][question_id] = []
        if question_id not in audio_type2score[audio_type]:
            audio_type2score[audio_type][question_id] = []
        if question_id not in difficulty2score[difficulty]:
            difficulty2score[difficulty][question_id] = []
        if question_id not in duration_type2score[duration_type]:
            duration_type2score[duration_type][question_id] = []
        video_category2score[video_category][question_id].append(score)
        sub_category2score[sub_category][question_id].append(score)
        question_type2score[question_type][question_id].append(score)
        audio_type2score[audio_type][question_id].append(score)
        difficulty2score[difficulty][question_id].append(score)
        duration_type2score[duration_type][question_id].append(score)

    # Calculate the average score for each video_category
    total_score = 0
    total_questions = 0

    # For video_category
    video_category_avg_scores = {}
    for video_category, questions in video_category2score.items():
        video_category_total = 0  # Category total score
        for question_id, score in questions.items():
            video_category_total += score[0]
        video_category_avg_scores[video_category] = video_category_total / len(questions) * 100.0
        total_score += video_category_total
        total_questions += len(questions)
    for video_category, avg_score in video_category_avg_scores.items():
        eval_logger.info(f"Evaluation on video_category Categories: {video_category}: {avg_score:.2f}")

    # For sub_category
    sub_category_avg_scores = {}
    for sub_category, questions in sub_category2score.items():
        sub_category_total = 0
        for question_id, score in questions.items():
            sub_category_total += score[0]
        sub_category_avg_scores[sub_category] = sub_category_total / len(questions) * 100.0
    for sub_category, avg_score in sub_category_avg_scores.items():
        eval_logger.info(f"Evaluation on sub_category Sub-Categories: {sub_category}: {avg_score:.2f}")

    # For question_type
    question_type_avg_scores = {}
    for question_type, questions in question_type2score.items():
        question_type_total = 0
        for question_id, score in questions.items():
            question_type_total += score[0]
        question_type_avg_scores[question_type] = question_type_total / len(questions) * 100.0
    for question_type, avg_score in question_type_avg_scores.items():
        eval_logger.info(f"Evaluation on question_type Domains: {question_type}: {avg_score:.2f}")

    # For audio_type
    audio_type_avg_scores = {}
    for audio_type, questions in audio_type2score.items():
        audio_type_total = 0
        for question_id, score in questions.items():
            audio_type_total += score[0]
        audio_type_avg_scores[audio_type] = audio_type_total / len(questions) * 100.0
    for audio_type, avg_score in audio_type_avg_scores.items():
        eval_logger.info(f"Evaluation on audio_type Audio Types: {audio_type}: {avg_score:.2f}")

    # For difficulty
    difficulty_avg_scores = {}
    for difficulty, questions in difficulty2score.items():
        difficulty_total = 0
        for question_id, score in questions.items():
            difficulty_total += score[0]
        difficulty_avg_scores[difficulty] = difficulty_total / len(questions) * 100.0
    for difficulty, avg_score in difficulty_avg_scores.items():
        eval_logger.info(f"Evaluation on difficulty Difficulty Levels: {difficulty}: {avg_score:.2f}")

    # For duration_type
    duration_type_avg_scores = {}
    for duration_type, questions in duration_type2score.items():
        duration_type_total = 0
        for question_id, score in questions.items():
            duration_type_total += score[0]
        duration_type_avg_scores[duration_type] = duration_type_total / len(questions) * 100.0
    for duration_type, avg_score in duration_type_avg_scores.items():
        eval_logger.info(f"Evaluation on duration_type Duration Types: {duration_type}: {avg_score:.2f}")

    overall_avg_score = total_score / total_questions * 100.0
    eval_logger.info(f"Overall performance (across all questions): {overall_avg_score:.2f}")

    return overall_avg_score