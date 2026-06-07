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

VIDEO_CAT = [
    'Howto & Style', 
    'Science & Technology', 
    'People & Blogs', 
    'News & Politics', 
    'Entertainment', 
    'Gaming', 
    'Education', 
    'Film & Animation', 
    'Sports', 
    'Autos & Vehicles', 
    'Nonprofits & Activism', 
    'Comedy', 
    'Music', 
    'Pets & Animals', 
    'Travel & Events'
]
QA_TYPE = [
    'Event Sequence', 
    'AV Event Alignment', 
    'Inference', 
    'Reasoning', 
    'Context understanding', 
    'Comparative'
]
DURATION = ["30s", "60s"]


BASE_SYS = "Carefully watch this video and pay attention to every detail. "
SYS = BASE_SYS + "Based on your observations, select the best option that accurately addresses the question."
PROMPT = """Your task is to accurately answer multiple-choice questions based on the given video.
Select the single most accurate answer from the given choices.
Question: {question}
Choices: {choices}
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
"""

with open(Path(__file__).parent / "dailyomni.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")  # /mnt/sh/mmvision/data/videollm/benchmarks
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])

def dailyomni_doc_to_visual(doc):
    """
    Return the path to the video only
    """
    video_id = doc["video_id"]
    video_path = os.path.join(cache_dir, "videos", f"{video_id}", f"{video_id}_video.mp4")
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


import ast
import re

def parse_candidates(raw):
    # Insert missing commas between quoted items before parsing.
    fixed = re.sub(r"'\s+'", "', '", raw.strip())
    return ast.literal_eval(fixed)

def dailyomni_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["Question"]
    choices = parse_candidates(doc["Choice"])
    choices = "\n".join(choices)
    full_prompt = PROMPT.format(question=question, choices=choices)
    return full_prompt

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


def dailyomni_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case av_odyssey score), value: metric value
    """
    pred = results[0]
    options = doc["Choice"]
    # For only 3 answers:
    if len(options) == 3:
        option_list = {"A": options[0][3:], "B": options[1][3:], "C": options[2][3:]}
        answer = parse_multi_choice_response(pred, ["A", "B", "C"], option_list)
    else:
        option_list = {"A": options[0][3:], "B": options[1][3:], "C": options[2][3:], "D": options[3][3:]}
        answer = parse_multi_choice_response(pred, ["A", "B", "C", "D"], option_list)
    gt_answer = doc["Answer"]
    assert answer in ["A", "B", "C", "D"]
    assert gt_answer in ["A", "B", "C", "D"]
    score = 1.0 if answer == gt_answer else 0.0
    video_category = doc["video_category"]
    Type = doc["Type"]
    video_duration = doc["video_duration"]
    key_name = "dailyomni_score"
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/video_category to help us aggregate the results later
    return {key_name: {"question_id": doc["question_id"], "video_category": video_category, "score": score, "Type": Type, "video_duration": video_duration}}


def dailyomni_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(dict)
    domain2score = defaultdict(dict)
    duration2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        video_category = result["video_category"]
        Type = result["Type"]
        video_duration = result["video_duration"]
        if question_id not in category2score[video_category]:
            category2score[video_category][question_id] = []
        if question_id not in domain2score[Type]:
            domain2score[Type][question_id] = []
        if question_id not in duration2score[video_duration]:
            duration2score[video_duration][question_id] = []

        category2score[video_category][question_id].append(score)
        domain2score[Type][question_id].append(score)
        duration2score[video_duration][question_id].append(score)

    # Calculate the average score for each category
    category_avg_scores = {}
    total_score = 0
    total_questions = 0

    # For video_category
    for video_category, questions in category2score.items():
        category_total = 0  # Category total score
        for question_id, score in questions.items():
            category_total += score[0]
        category_avg_scores[video_category] = category_total / len(questions) * 100.0
        total_score += category_total
        total_questions += len(questions)
    for video_category, avg_score in category_avg_scores.items():
        eval_logger.info(f"Evaluation on Video Categories: {video_category}: {avg_score:.2f}")

    # For Type
    domain_avg_scores = {}
    for Type, questions in domain2score.items():
        domain_total = 0
        for question_id, score in questions.items():
            domain_total += score[0]
        domain_avg_scores[Type] = domain_total / len(questions) * 100.0
    for Type, avg_score in domain_avg_scores.items():
        eval_logger.info(f"Evaluation on QA Types: {Type}: {avg_score:.2f}")

    # For video_duration
    duration_avg_scores = {}
    for video_duration, questions in duration2score.items():
        duration_total = 0
        for question_id, score in questions.items():
            duration_total += score[0]
        duration_avg_scores[video_duration] = duration_total / len(questions) * 100.0
    for video_duration, avg_score in duration_avg_scores.items():
        eval_logger.info(f"Evaluation on Video Durations: {video_duration}: {avg_score:.2f}")

    overall_avg_score = total_score / total_questions * 100.0
    eval_logger.info(f"Overall performance (across all questions): {overall_avg_score:.2f}")

    return overall_avg_score