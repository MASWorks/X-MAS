import os
import re
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
parser.add_argument("--plan_model_names", type=str, nargs='+', default=["llama-3.1-8b-instruct", "qwen-2.5-7b-instruct", "qwen-2.5-14b-instruct"])
parser.add_argument("--test_dataset_name", type=str, default="MedMCQA")
parser.add_argument("--sample_num", type=int, default=500)
parser.add_argument("--sequential", action="store_true")
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()
general_config = vars(args)

from utils import LLM

print("="*50)
print(json.dumps(vars(args), indent=4))

def extract_roles(response: str):
    role_descriptions = []
    if "###ROLE START:" in response and "ROLE END###" in response:
        pattern = r"###ROLE START:\s*(.*?)\s*ROLE END###"
        matches = re.findall(pattern, response, re.DOTALL)
        role_descriptions = [match.strip() for match in matches]
    elif "ROLE START" in response and "ROLE END" in response:
        pattern = r"ROLE START\s*(.*?)\s*ROLE END"
        matches = re.findall(pattern, response, re.DOTALL)  
        role_descriptions = [match.strip() for match in matches]
    elif "ROLE START" in response and "ROLE END" not in response:
        # Find all occurrences of 'ROLE START' in the response
        matches = list(re.finditer(r'ROLE START\s*', response, re.IGNORECASE | re.MULTILINE))
        # Extract the content after each 'ROLE START' and save it to a list
        for i, match in enumerate(matches):
            start_index = match.end()  # Get the end position of 'ROLE START' and any whitespace following it
            # If it's the last 'ROLE START', extract until the end of the string
            if i == len(matches) - 1:
                role_description = response[start_index:].strip()
            else:
                # Otherwise, extract until the start of the next 'ROLE START'
                next_start_index = matches[i + 1].start()
                role_description = response[start_index:next_start_index].strip()
            # Add to the list (unless it's empty)
            if role_description:
                role_descriptions.append(role_description)
    
    print("\nnum of roles:", len(role_descriptions))
    return role_descriptions

def plan_init_answers(query):

    str = "You are given a [Question]. Your task is to provide a suitable role plan to the question by giving the role descriptions of the agents required to solve this problem in the order of workflow in a fixed format which starts with "###ROLE START:" and ends with "ROLE END###\n"". Determine the number of agents and content of role descriptions in role plan based on the question. One example for the role plan format is as follows:\n\n"

    str += "-----\n###ROLE START: You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems.Your task is to come up with a workable algorithm based on the problem to solve it.ROLE END###\n###ROLE START: You are a Mathematics Expert skilled in solving complex equations, mathematical modeling, and logical problem-solving. Given the problem and the algorithm in teh previews response, provide detailed calculations, step-by-step reasoning, and a precise mathematical solution using advanced mathematical concepts where necessary.ROLE END###\n###ROLE START: You are a experienced math-problem solver. You are good at solve math problem, reflect previews solutions and summarize the final answers to math problems.Your task is to reflect and give the final answer to teh question.And be sure to restate the final answer at the end.ROLE END###\n"

    str += "The question is as follows:\n\n"

    str += f"-----\n# [Question]:\n {query}\n\n"

    str += "-----\n\nNow, given the question, provide a role plan to the question directly."
    return str


def get_sample_pool(test_dataset_name):    
    test_data_path = f"X-MAS-Bench/benchmarks/{test_dataset_name}.json"
    with open(test_data_path, "r") as f:
        samples = json.load(f)        
          
    sample_pool = []
    for i,sample in enumerate(samples):
        query = sample["query"]
        sample_copy = deepcopy(sample)
        sample_copy["plan_query"] = plan_init_answers(query)
        sample_pool.append(sample_copy)

    return sample_pool

# def create_shuffled_list(M):
#     np.random.seed(2025)
#     list = np.random.randint(0, M, size=200)
    
#     return list

# # ============== parallel execution ==============
def process_sample(sample, plan_model_names):
    # shuffle_list = create_shuffled_list(len(plan_model_names))
    shuffle_list =[
    2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
    1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0,
    2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
    1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0,
    2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
    1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0,
    2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
    1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 0, 1
    ]
    # print("\nshuffle_list:", shuffle_list)
    
    llm = LLM(general_config, model_list)
    query = sample["plan_query"]
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
            def output_trans(response):
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
                return generated_output
            sample["plan_response"] = output_trans(response)
            sample["num_prompt_tokens"] = response.usage.prompt_tokens
            sample["num_completion_tokens"] = response.usage.completion_tokens
            role_descriptions = extract_roles(output_trans(response))
            if role_descriptions==[]:
                with open(output_json, "a") as result_file:
                    sample["generated_output"] = ""
                    sample["num_prompt_tokens"] = 0
                    sample["num_completion_tokens"] = 0
                    json.dump(sample, result_file)
                    result_file.write("\n")
                return
            model_id_list = shuffle_list[:len(role_descriptions)]
            plan_response = ""
            init_query = sample["query"]
            for i,id in enumerate(model_id_list):
                print(f"plan_query_{i}_model:", plan_model_names[id])
                plan_model_list = model_dict[plan_model_names[id]]
                plan_llm = LLM(general_config, plan_model_list)
                if i == 0 :
                    plan_query = role_descriptions[0] + f"\nThe question is : {init_query}"
                else:
                    plan_query = role_descriptions[i] + f"\nThe question is : {init_query}" + f"\nThe response of the last agent:\n {plan_response}"
                sample[f"plan_query_{i}"] = plan_query
                plan_response = plan_llm.call_llm(prompt = plan_query)
                if isinstance(plan_response, str):
                    if "Error occurred:" in plan_response:
                        sample[f"plan_response_{i}"] = plan_response
                        sample["num_prompt_tokens"] = 0
                        sample["num_completion_tokens"] = 0
                else:
                    sample[f"plan_response_{i}"] = output_trans(plan_response)
                    sample["plan_num_prompt_tokens_{i}"] = plan_response.usage.prompt_tokens
                    sample["plan_num_completion_tokens_{i}"] = plan_response.usage.completion_tokens
                    
                
            with open(output_json, "a") as result_file:
                sample["generated_output"] = output_trans(plan_response)
                json.dump(sample, result_file)
                result_file.write("\n")
    
    except Exception as e:
        logging.error(f"{query[:20]} failed to execute: {e}\nTraceback:\n{traceback.format_exc()}")

# ============== main ==============
test_dataset_name = args.test_dataset_name

try:


    # ================== Define the output files ==================
    output_logging = f"X-MAS-Bench/results/{test_dataset_name}/planning/log/{args.model_name}_planning.txt"
    output_json = f"X-MAS-Bench/results/{test_dataset_name}/planning/{args.model_name}_planning.jsonl"
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
    print(f">> plan Initially: {len(sample_pool)} samples ")

    # ================== Load the processed queries ==================
    processed_queries = set()
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            for line in f:
                infered_sample = json.loads(line)
                processed_queries.add(infered_sample["query"])
    sample_pool = [sample for sample in sample_pool if sample["query"] not in processed_queries]
    print(f">> plan After filtering: {len(sample_pool)} samples with {args.model_name} on {test_dataset_name}")

    # ================== Define the model list ==================
    with open(args.model_config, "r") as f:
        config = json.load(f)
        model_dict = config["model_dict"]

    model_list = model_dict[args.model_name]["model_list"]
    max_workers = model_dict[args.model_name]["max_workers_per_model"] * len(model_list)

    if args.sequential:
        for sample in tqdm(sample_pool, desc=f"Processing plan queries with {args.model_name}"):
            process_sample(sample, args.plan_model_names)
    else:
        # Use ThreadPoolExecutor to process samples in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            args_for_execution = [(sample, args.plan_model_names) for sample in sample_pool]
            
            for _ in tqdm(executor.map(lambda x: process_sample(*x), args_for_execution), total=len(sample_pool), desc=f"Processing plan queries with {args.model_name} on {test_dataset_name}"):
                pass
except Exception as e:
    print(f"plan Traceback: {traceback.format_exc()}")