import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml
import ast
from loguru import logger as eval_logger

VIDEO_TYPES = ['interview', 'music', 'education', 'TED speech', 'animation', 'sports', 'unboxing_review', 'meeting', 'documentary', 'vlog', 'live', 'video reaction', 'game', 'news', 'debate', 'food', 'DIY', 'talkshow']

TASK_TYPES = ['Audio Character Matching', 'Audio OCR Matching', 'Audio Information Extraction', 'Audio Object Matching', 'Audio Content Counting', 'Audio Event Location']

SUBSET_TYPES = ['human', 'gemini']

BASE_SYS = "Carefully watch this video and pay attention to every detail. "
SYS = BASE_SYS + "Based on your observations, select the best option that accurately addresses the question."
FRAMES_TMPL_AUDIO = """
These are the frames of a video and the corresponding audio. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

with open(Path(__file__).parent / "avut_av_human.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")  # /mnt/sh/mmvision/data/videollm/benchmarks
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])

def avutbenchmark_doc_to_visual(doc):
    """
    Return the path to the video only
    """
    video_path = os.path.join(cache_dir, "videos", doc["video_path"])
    if os.path.exists(video_path):
        pass
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]



def parse_candidates(raw):
    # Fix missing commas between quoted strings
    fixed = re.sub(r"'\s+'", "', '", raw.strip())
    return ast.literal_eval(fixed)


def avutbenchmark_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = "Select the best answer to the following multiple-choice question based on the given video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    candidates = parse_candidates(doc["options"])
    option = "\n".join([f"{chr(ord('A') + idx)}. {op}" for idx, op in enumerate(candidates)])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
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


def avutbenchmark_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case avutbenchmark score), value: metric value
    """
    pred = results[0]
    options = doc["options"]
    # For only 3 answers:
    if len(options) == 3:
        option_list = {"A": options[0][3:], "B": options[1][3:], "C": options[2][3:]}
        answer = parse_multi_choice_response(pred, ["A", "B", "C"], option_list)
    else:
        option_list = {"A": options[0][3:], "B": options[1][3:], "C": options[2][3:], "D": options[3][3:]}
        answer = parse_multi_choice_response(pred, ["A", "B", "C", "D"], option_list)
    gt_answer = doc["answer"]
    assert answer in ["A", "B", "C", "D"]
    score = 1.0 if answer == gt_answer else 0.0
    video_type = doc["task_type"]
    task_type = doc["task_type"]
    subset_type = doc["subset_type"]
    key_name = "avutbenchmark_score"
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/video_type to help us aggregate the results later
    return {key_name: {"question_id": doc["question_id"], "score": score, "pred_answer": answer, "video_type": video_type, "task_type": task_type, "subset_type": subset_type}}



def avutbenchmark_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    video_type2score = defaultdict(dict)
    task_type2score = defaultdict(dict)
    subset_type2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        video_type = result["video_type"]
        task_type = result["task_type"]
        subset_type = result["subset_type"]
        if question_id not in video_type2score[video_type]:
            video_type2score[video_type][question_id] = []
        if question_id not in task_type2score[task_type]:
            task_type2score[task_type][question_id] = []
        if question_id not in subset_type2score[subset_type]:
            subset_type2score[subset_type][question_id] = []
        video_type2score[video_type][question_id].append(score)
        task_type2score[task_type][question_id].append(score)
        subset_type2score[subset_type][question_id].append(score)

    # Calculate the average score for each video_type
    category_avg_scores = {}
    total_score = 0
    total_questions = 0

    # For video_task
    for video_type, questions in video_type2score.items():
        category_total = 0  # Category total score
        for question_id, score in questions.items():
            category_total += score[0]
        category_avg_scores[video_type] = category_total / len(questions) * 100.0
        total_score += category_total
        total_questions += len(questions)
    for video_type, avg_score in category_avg_scores.items():
        eval_logger.info(f"Evaluation on video_type Categories: {video_type}: {avg_score:.2f}")

    # For task_type
    domain_avg_scores = {}
    for task_type, questions in task_type2score.items():
        domain_total = 0
        for question_id, score in questions.items():
            domain_total += score[0]
        domain_avg_scores[task_type] = domain_total / len(questions) * 100.0
    for task_type, avg_score in domain_avg_scores.items():
        eval_logger.info(f"Evaluation on task_type Domains: {task_type}: {avg_score:.2f}")

    # For subset_type
    subset_avg_scores = {}
    for subset_type, questions in subset_type2score.items():
        subset_total = 0
        for question_id, score in questions.items():
            subset_total += score[0]
        subset_avg_scores[subset_type] = subset_total / len(questions) * 100.0
    for subset_type, avg_score in subset_avg_scores.items():
        eval_logger.info(f"Evaluation on subset_type Subsets: {subset_type}: {avg_score:.2f}")

    overall_avg_score = total_score / total_questions * 100.0
    eval_logger.info(f"Overall performance (across all questions): {overall_avg_score:.2f}")

    return overall_avg_score