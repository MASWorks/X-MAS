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
parser.add_argument("--model_config", type=str, default="config_yerui.json")
parser.add_argument("--aggregate_model_names", type=str, nargs='+', default=["llama-3.3-70b-instruct", "qwen2.5-32b-instruct", "deepseek-r1-distill-qwen-32b"])
parser.add_argument("--test_dataset_names", type=str, nargs='+', default=["MATH", "GSM8K", "AQUA-RAT", "MedMCQA"])
parser.add_argument("--sample_num", type=int, default=500)
parser.add_argument("--sequential", action="store_true")
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../mas_pool')))
from utils import LLM

print("="*50)
print(json.dumps(vars(args), indent=4))

def aggregate_init_answers(query, answer_dict, shuffle_index):
    answers = list(answer_dict.values())
    answers = [answers[i] for i in shuffle_index]

    str = "You are given a [Question] and several [Solution] to the question. Your task is to provide a final complete solution to the question by reasoning over the given solutions. The question and the solutions are as follows:\n\n"

    str += f"-----\n# [Question]:\n {query}\n\n# [Solutions]:\n"

    for i, result in enumerate(answers):
        str += f"## [Solution {i+1}]:\n{result}\n\n"

    str += "-----\n\nNow, given the question and all the above solutions, reason over them carefully and provide a final complete solution to the question."
    return str


def get_sample_pool(test_dataset_name, aggregate_model_names):
    query_dict = defaultdict(dict)
    for model_name in aggregate_model_names:
        with open(f"XMAS-Bench/results/{test_dataset_name}/{model_name}_direct.jsonl", "r") as f:
            for line in f:
                sample = json.loads(line)
                query = sample["query"]
                query_dict[query][model_name] = sample['generated_output']
                
    shuffle_matrix = create_shuffled_matrix(len(query_dict), len(aggregate_model_names))
    sample_pool = []
    with open(f"XMAS-Bench/results/{test_dataset_name}/{aggregate_model_names[0]}_direct.jsonl", "r") as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            query = sample["query"]
            sample_copy = deepcopy(sample)
            del sample_copy["generated_output"]
            sample_copy["aggregate_query"] = aggregate_init_answers(query, query_dict[query], shuffle_matrix[i])
            sample_pool.append(sample_copy)

    return sample_pool

def create_shuffled_matrix(N, M):
    """This function is for ensuring that the solutions to be aggregated are in same random order."""

    np.random.seed(2025)
    matrix = np.tile(np.arange(M), (N, 1))
    
    # 打乱每一行的顺序
    for i in range(N):
        np.random.shuffle(matrix[i])
    
    return matrix

# # ============== parallel execution ==============
def process_sample(sample):
    
    llm = LLM(model_list)
    query = sample["aggregate_query"]
    try:
        response = llm.call_llm(query)


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
                sample["generated_output"] = generated_output
                sample["num_prompt_tokens"] = response.usage.prompt_tokens
                sample["num_completion_tokens"] = response.usage.completion_tokens
                json.dump(sample, result_file)
                result_file.write("\n")        
    
    except Exception as e:
        logging.error(f"{query[:20]} failed to execute: {e}\nTraceback:\n{traceback.format_exc()}")

# ============== main ==============
for i, test_dataset_name in enumerate(args.test_dataset_names):

    try:

        print(f">> Processing Aggregate {i}-th dataset: {test_dataset_name} with {args.model_name}")

        # ================== Define the output files ==================
        output_logging = f"XMAS-Bench/results/{test_dataset_name}/log/{args.model_name}_aggregate.txt"
        output_json = f"XMAS-Bench/results/{test_dataset_name}/{args.model_name}_aggregate.jsonl"
        output_dir = os.path.dirname(output_logging)
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(filename=output_logging, level=logging.INFO, format='%(asctime)s - %(message)s')

        # ================== Load the files to be processed ==================
        if test_dataset_name == "SciKnowEval":
            sample_num = 800
        else:
            sample_num = args.sample_num
        sample_pool = get_sample_pool(test_dataset_name, args.aggregate_model_names)
        sample_pool = sample_pool[:sample_num] if sample_num > 0 else sample_pool
        sample_pool = sample_pool[:5] if args.dry_run else sample_pool
        print(f">> Aggregate Initially: {len(sample_pool)} samples ")

        # ================== Load the processed queries ==================
        processed_queries = set()
        if os.path.exists(output_json):
            with open(output_json, "r") as f:
                for line in f:
                    infered_sample = json.loads(line)
                    processed_queries.add(infered_sample["query"])
        sample_pool = [sample for sample in sample_pool if sample["query"] not in processed_queries]
        print(f">> Aggregate After filtering: {len(sample_pool)} samples with {args.model_name} on {test_dataset_name}")

        # ================== Define the model list ==================
        with open(args.model_config, "r") as f:
            config = json.load(f)
            model_dict = config["model_dict"]
            worker_dict = config["worker_dict"]

        model_list = model_dict[args.model_name]
        max_workers = worker_dict[args.model_name] * len(model_list)

        if args.sequential:
            for sample in tqdm(sample_pool, desc=f"Processing aggregate queries with {args.model_name}"):
                process_sample(sample)
        else:
            # Use ThreadPoolExecutor to process samples in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in tqdm(executor.map(process_sample, sample_pool), total=len(sample_pool), desc=f"Processing aggregate queries with {args.model_name} on {test_dataset_name}"):
                    pass
    except Exception as e:
        print(f"Aggregate Traceback: {traceback.format_exc()}")