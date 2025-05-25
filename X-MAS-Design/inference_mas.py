import os
import json
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
import traceback

from methods.mas_base import MAS
from methods import get_method_class

def load_model_api_config(model_api_config):
    with open(model_api_config, "r") as f:
        model_api_config = json.load(f)
    model_api_config = model_api_config["model_dict"]
    for model_name in model_api_config:
        actural_max_workers = model_api_config[model_name]["max_workers_per_model"] * len(model_api_config[model_name]["model_list"])
        model_api_config[model_name]["max_workers"] = actural_max_workers
    return model_api_config

def write_to_jsonl(lock, file_name, data):
    with lock:
        with open(file_name, 'a') as f:
            json.dump(data, f)
            f.write('\n')

def process_sample(args, general_config, sample, output_path, lock):
    MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
    if args.method_config_name is not None:
        mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)
    else:
        mas = MAS_METHOD(general_config)
    save_data = sample.copy()
    query = sample["query"]
    try:
        response = mas.inference(query)
        save_data["generated_output"] = response
    except Exception as e:
        save_data["error"] = f"Inference Error: {traceback.format_exc()}"
    save_data.update({"token_stats": mas.get_token_stats()})
    write_to_jsonl(lock, output_path, save_data)

def reserve_unprocessed(output_json, test_dataset):
    processed_queries = set()
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            for line in f:
                infered_sample = json.loads(line)
                processed_queries.add(infered_sample["query"])

    test_dataset = [sample for sample in test_dataset if sample["query"] not in processed_queries]
    return test_dataset

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args related to the method
    parser.add_argument("--method_name", type=str, default="vanilla", help="MAS name.")
    parser.add_argument("--method_config_name", type=str, default=None, help="The config file name. If None, the default config file will be used.")

    # args related to the model
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18", help="The agent backend to be used for inference.")
    parser.add_argument("--model_api_config", type=str, default="configs/X-MAS_Design_config.json")
    parser.add_argument("--model_temperature", type=float, default=0.5, help="Temperature for sampling.")
    parser.add_argument("--model_max_tokens", type=int, default=2048, help="Maximum tokens for sampling.")
    parser.add_argument("--model_timeout", type=int, default=600, help="Timeout for sampling.")
    
    # args related to dataset
    parser.add_argument("--test_dataset_name", type=str, default="example_math", help="The dataset to be used for testing.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output file.")
    parser.add_argument("--sample_num", type=int, default=500)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    
    general_config = vars(args)
    
    # Load model config
    model_api_config = load_model_api_config(args.model_api_config)
    general_config.update({"model_api_config": model_api_config})
    print("-"*50, f"\n>> Model API config: {general_config['model_api_config']}")
    print("-"*50, f"\n>> Model API config for X-MAS: {model_api_config[args.model_name]}")
    
    if args.debug:
        # MAS inference
        query = "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction."     # ground-truth is "\\frac{11}{2}"
        # query = "hello"
        MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
        if args.method_config_name is not None:
            mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)
        else:
            mas = MAS_METHOD(general_config)

        response = mas.inference(query)
        
        print(response)
        print(f"\n>> Token stats: {mas.get_token_stats()}")
    
    else:
        print(f">> Method: {args.method_name} | Dataset: {args.test_dataset_name}")
        # load dataset
        with open(f"./X-MAS-Design/benchmarks/{args.test_dataset_name}.json", "r") as f:
            test_dataset = json.load(f)
        
        # get output path
        output_path = args.output_path if args.output_path is not None else f"./X-MAS-Design/results/{args.test_dataset_name}/{args.method_name}_manual_4m_infer.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # reserve unprocessed samples
        if args.test_dataset_name == "SciKnowEval":
            sample_num = 800
        else:
            sample_num = args.sample_num
        test_dataset = test_dataset[:sample_num] if sample_num > 0 else test_dataset
        test_dataset = reserve_unprocessed(output_path, test_dataset)
        print(f">> After filtering: {len(test_dataset)} samples")
        
        lock = threading.Lock()
        # inference the mas
        if args.sequential:
            for sample in test_dataset:
                process_sample(args, general_config, sample, output_path)
        else:
            max_workers = model_api_config[args.model_name]["max_workers"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in tqdm(executor.map(lambda sample: process_sample(args, general_config, sample, output_path, lock), test_dataset), total=len(test_dataset), desc=f"Processing queries with {args.method_name} on {args.test_dataset_name}"):
                    pass