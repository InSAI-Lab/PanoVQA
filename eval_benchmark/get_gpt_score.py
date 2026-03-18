#!/usr/bin/env python3

import os
import json
import re
import argparse
import openai
import threading
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

MAX_RETRIES = 3
REQUEST_DELAY = 0.0  # seconds, if you want to add delay between requests

class OpenAIClientPool:
    """Handles multiple OpenAI API keys for load balancing and retries."""
    def __init__(self, api_key, model_name, reasoning_effort=None):
        self.api_key = api_key
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.client = openai.OpenAI(api_key=api_key)
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "gpt_score_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "gpt_score": {
                            "type": "integer",
                            "description": "The GPT-evaluated score as an integer value"
                        }
                    },
                    "required": ["gpt_score"],
                    "additionalProperties": False
                }
            }
        }

    def get_client(self):
        return self.client

    def score(self, messages, retry_count=0):
        if retry_count >= MAX_RETRIES:
            logging.error("Max retries reached for OpenAI request.")
            return None
        try:
            client = self.get_client()
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                # "reasoning_effort": self.reasoning_effort,
                "response_format": self.response_format
            }
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.warning(f"OpenAI API error (retry {retry_count+1}/{MAX_RETRIES}): {e}")
            return self.score(messages, retry_count + 1)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions using OpenAI GPT")
    parser.add_argument("--input", "-i", type=str, required=False, default="./datasets/models_eva/vllm_single/whole/InternVL3-8B-Instruct/inference_results.json", help="Path to input JSON file with VQA results")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output JSON file for evaluation results")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit evaluation to first N samples (for testing)")
    parser.add_argument("--api-key", "-k", type=str, required=False, default="YOU API KEY", help="OpenAI API key")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini", help="OpenAI GPT model to use for evaluation")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Number of parallel workers for evaluation")
    return parser.parse_args()

def score_sample(args):
    sample_id, sample = args
    print(f"Scoring id: {sample_id}")
    # Extract category from id (e.g., N4_val_1538... -> N4)
    if isinstance(sample_id, str) and '_' in sample_id:
        category = sample_id.split('_')[0]
    else:
        category = sample.get('category')
    question = sample.get('question')
    gt_answer, pred_answer = sample.get('gt_answer'), sample.get('pred_answer')

    client = openai.OpenAI(api_key=GLOBAL_API_KEY)
    messages = [
        {
            "role": "system",
            "content":
                "You are an intelligent evaluator designed to evaluate the correctness and similarity of generative outputs for question-answer pairs. "
                "Your task is to compare the model prediction answer with the correct answer and determine if they match in meaning. Here's the scoring criteria:\n\n"
                "### Scoring Criteria:\n"
                "5 = Perfect match or Correct in meaning\n"
                "4 = Key information correct, minor flaws\n"
                "3 = Partially correct\n"
                "2 = Mostly wrong answer for key query, but some relevance\n"
                "1 = Completely wrong or nonsense sentences\n\n"
                "Your response must ONLY be the integer score (e.g., 4). DO NOT include any text or explanation."
        },
        {
            "role": "user",
            "content":
                f"Question: {question}\n"
                f"Correct Answer: {gt_answer}\n"
                f"Predicted Answer: {pred_answer}\n\n"
                "Please provide a score from 1 to 5 based on how well the predicted answer matches the correct answer."
        }
    ]
    try:
        response = client.chat.completions.create(
            model=GLOBAL_MODEL,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "gpt_score_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "gpt_score": {
                                "type": "integer",
                                "description": "The GPT-evaluated score as an integer value"
                            }
                        },
                        "required": ["gpt_score"],
                        "additionalProperties": False
                    }
                }
            }
        )
        reply = response.choices[0].message.content.strip()
        reply_json = json.loads(reply)
        score = reply_json['gpt_score']
    except Exception as e:
        print(f"[!] GPT API error: {e}")
        score = 1
    return {
        "id": sample_id,
        "category": category,
        "question": question,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "score": score
    }

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.output is None:
        args.output = "./eval_benchmark/outputs/gpt_score/" + args.input.split('/')[-1]


    # Load evaluation data (dict: id -> sample)
    with open(args.input, 'r') as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} evaluation samples.")

    # Apply limit if specified
    items = list(eval_data.items())
    if args.limit:
        items = items[:args.limit]
        print(f"Limited to first {args.limit} samples for evaluation.")

    global GLOBAL_API_KEY, GLOBAL_MODEL
    GLOBAL_API_KEY = args.api_key
    GLOBAL_MODEL = args.model

    # Prepare samples for evaluation: list of (id, sample)
    samples_to_evaluate = items

    # Evaluate samples (parallel if workers > 1)
    results = []
    checkpoint_interval = 20
    if args.workers > 1:
        with Pool(min(args.workers, cpu_count())) as pool:
            for idx, result in enumerate(tqdm(pool.imap(score_sample, samples_to_evaluate), total=len(samples_to_evaluate), desc="Evaluating")):
                results.append(result)
                if (idx + 1) % checkpoint_interval == 0:
                    # Save checkpoint
                    output_data = {"samples": results}
                    with open(args.output, 'w') as f:
                        json.dump(output_data, f, indent=4)
                    print(f"Checkpoint saved at {idx + 1} samples to {args.output}")
    else:
        for idx, sample_data in enumerate(tqdm(samples_to_evaluate, desc="Evaluating")):
            results.append(score_sample(sample_data))
            if (idx + 1) % checkpoint_interval == 0:
                # Save checkpoint
                output_data = {"samples": results}
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=4)
                print(f"Checkpoint saved at {idx + 1} samples to {args.output}")

    # Category results
    # category_list = ["N1", "N2", "N3", "N4", "D1", "D2", "D3", "D4", "D5", "O1", "O2", "O3"]
    category_list = ["N1", "N2", "N3", "N4", "O1", "O2", "O3", "D1", "D2", "D3", "D4", "D5"]

    category_results = []
    for category in category_list:
        cat_samples = [r for r in results if r['category'] == category]
        if not cat_samples:
            continue
        avg_score = sum(r['score'] for r in cat_samples) / len(cat_samples)
        category_results.append({
            "category": category,
            "num_samples": len(cat_samples),
            "avg_score": avg_score
        })

    # Generate summary
    category_summary = []
    for item in category_results:
        std_score = (item["avg_score"] - 1) / 4 * 100
        category_summary.append({
            "category": item["category"],
            "avg_score": round(item["avg_score"], 2),
            "standardized_score": round(std_score, 2)
        })

    # Print summary
    print("\nCategory Summary:")
    for item in category_summary:
        print(f"{item['category']}: Score={item['avg_score']} ({item['standardized_score']}%)")

    # Overall score
    all_scores = [r['score'] for r in results]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    overall_std = (overall_avg - 1) / 4 * 100
    print(f"\nOverall Score: {round(overall_avg, 2)} ({round(overall_std, 2)}%)")

    # Final save results
    output_data = {
        "samples": results,
        "category_results": category_results,
        "category_summary": category_summary,
        "overall": {
            "avg_score": round(overall_avg, 2),
            "standardized_score": round(overall_std, 2)
        }
    }

    score_list = []
    for item in category_summary:
        score_list.append(item['standardized_score'])
    average = sum(score_list) / len(score_list) if score_list else 0
    score_list.append(round(average, 2))
    tab_line = "\t".join([str(x) for x in score_list])

    # When saving, first line is tab-separated scores, then followed by JSON
    with open(args.output, 'w') as f:
        f.write(tab_line + "\n")
        # f.write(json.dumps({"result": tab_line}) + "\n")

        json.dump(output_data, f, indent=4)
    print(f"Results saved to {args.output}")
    # Output tab-separated results, convenient for copying to Excel
    print("category\tavg_score\tstandardized_score")
    for item in category_summary:
        print(f"{item['category']}\t{item['avg_score']}\t{item['standardized_score']}")
    print("------------------------------------------------")
    print(tab_line)

if __name__ == "__main__":
    main()
