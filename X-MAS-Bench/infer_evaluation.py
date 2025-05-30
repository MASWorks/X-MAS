import os
import json
import concurrent.futures
import logging
from tqdm import tqdm
import traceback
import sys
import numpy as np
from collections import defaultdict
from copy import deepcopy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llama-3-70b-instruct", help="The agent backend to be used for inference.")
parser.add_argument("--model_temperature", type=float, default=0.5, help="Temperature for sampling.")
parser.add_argument("--model_max_tokens", type=int, default=2048, help="Maximum tokens for sampling.")
parser.add_argument("--model_timeout", type=int, default=600, help="Timeout for sampling.")
parser.add_argument("--model_config", type=str, default="config_.json")
parser.add_argument("--test_dataset_name", type=str, default="MedMCQA")
parser.add_argument("--sample_num", type=int, default=500)
parser.add_argument("--sequential", action="store_true")
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()
general_config = vars(args)

from utils import LLM

print("="*50)
print(json.dumps(vars(args), indent=4))

def evaluate_init_answers(query, answer):

    str = "You are given a [Question] and a [Solution] to the question. Your task is to provide a final evaluation like True or False to the solution by considering the given solution. The question and the solution are as follows:\n\n"

    str += f"-----\n# [Question]:\n {query}\n\n"

    # for i, result in enumerate(answers):
    str += f"## [Solution]:\n{answer}\n\n"

    str += "-----\n\nNow, given the question and solution, evaluate them carefully and directly provide a final evaluation 'The solution is correct' or 'The solution is incorrect' at the end. Be careful not to give evaluations other than 'The solution is correct' or 'The solution is incorrect' at the end"
    return str


def get_sample_pool(test_dataset_name):
    sample_pool = []
    with open(f"X-MAS-Bench/results/{test_dataset_name}/qa/qwen2.5-32b-instruct_qa.jsonl", "r") as f:

        for i, line in enumerate(f):
            sample = json.loads(line)
            query = sample["query"]
            sample_copy = deepcopy(sample)
            sample_copy["evaluate_query"] = evaluate_init_answers(query, sample_copy["generated_output"])
            del sample_copy["generated_output"]
            sample_pool.append(sample_copy)

    return sample_pool


# # ============== parallel execution ==============
def process_sample(sample):
    
    llm = LLM(general_config, model_list)
    query = sample["evaluate_query"]
    try:
        response = llm.call_llm(prompt = query)
        if isinstance(response, str):
            if "Error occurred:" in response and "Error code: 400" in response:
                with open(output_json, "a") as result_file:
                    sample["generated_output"] = response
                    sample["num_prompt_tokens"] = 0
                    sample["num_completion_tokens"] = 0
                    json.dump(sample, result_file)
                    result_file.write("\n")        
            else:
                print(f"{query[:20]} failed to execute:{response}")     
        else:
            generated_output = response.choices[0].message.content
            if "r1" in args.model_name or "qwq" in args.model_name:
                try:
                    generated_output = generated_output.split("</think>")[-1].strip()
                except:
                    pass
            elif "openthinker" in args.model_name:
                try:
                    generated_output = generated_output.split('<|begin_of_solution|>')[-1].split('<|end_of_solution|>')[0].strip()
                except:
                    pass
            elif "huatuo" in args.model_name:
                try:
                    generated_output = generated_output.split('## Final Response')[-1].strip()
                except:
                    pass
            with open(output_json, "a") as result_file:
                if "The solution is correct" in generated_output or "the solution is correct" in generated_output:
                    sample["generated_output"] = "True"
                elif "The solution is incorrect" in generated_output or "the solution is incorrect" in generated_output:
                    sample["generated_output"] = "False"
                else:
                    sample["wrong_format_generated_output"] = generated_output
                    sample["generated_output"] = ""
                sample["num_prompt_tokens"] = response.usage.prompt_tokens
                sample["num_completion_tokens"] = response.usage.completion_tokens
                json.dump(sample, result_file)
                result_file.write("\n")        

    except Exception as e:
        logging.error(f"{query[:20]} failed to execute: {e}\nTraceback:\n{traceback.format_exc()}")

# ============== main ==============
test_dataset_name = args.test_dataset_name

try:


    # ================== Define the output files ==================
    output_logging = f"X-MAS-Bench/results/{test_dataset_name}/evaluation/log/{args.model_name}_evaluation.txt"
    output_json = f"X-MAS-Bench/results/{test_dataset_name}/evaluation/{args.model_name}_evaluation.jsonl"
    output_dir_log = os.path.dirname(output_logging)
    output_dir_json = os.path.dirname(output_json)
    os.makedirs(output_dir_log, exist_ok=True)
    os.makedirs(output_dir_json, exist_ok=True)
    logging.basicConfig(filename=output_logging, level=logging.INFO, format='%(asctime)s - %(message)s')

    # ================== Load the files to be processed ==================
    if test_dataset_name == "SciKnowEval":
        sample_num = 800
    else:
        sample_num = args.sample_num
    sample_pool = get_sample_pool(test_dataset_name)
    sample_pool = sample_pool[:sample_num] if sample_num > 0 else sample_pool
    sample_pool = sample_pool[:5] if args.dry_run else sample_pool
    print(f">> evaluate Initially: {len(sample_pool)} samples")

    # ================== Load the processed queries ==================
    processed_queries = set()
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            for line in f:
                infered_sample = json.loads(line)
                processed_queries.add(infered_sample["query"])
    sample_pool = [sample for sample in sample_pool if sample["query"] not in processed_queries]
    print(f">> evaluate After filtering: {len(sample_pool)} samples with {args.model_name} on {test_dataset_name}")

    # ================== Define the model list ==================
    with open(args.model_config, "r") as f:
        config = json.load(f)
        model_dict = config["model_dict"]

    model_list = model_dict[args.model_name]["model_list"]
    max_workers = model_dict[args.model_name]["max_workers_per_model"] * len(model_list)

    if args.sequential:
        for sample in tqdm(sample_pool, desc=f"Processing evaluate queries with {args.model_name}"):
            process_sample(sample)
    else:
        # Use ThreadPoolExecutor to process samples in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in tqdm(executor.map(process_sample, sample_pool), total=len(sample_pool), desc=f"Processing evaluate queries with {args.model_name} on {test_dataset_name}"):
                pass
except Exception as e:
    print(f"evaluate Traceback: {traceback.format_exc()}")