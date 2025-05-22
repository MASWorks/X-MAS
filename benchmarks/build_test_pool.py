import json
import os
import random
import argparse
import re

from datasets import load_dataset, concatenate_datasets, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="MATH")
parser.add_argument("--num2sample", type=int, default=1000)
args = parser.parse_args()

save_path = f"./test_pool/{args.dataset_name}.json"

sample_pool = []

def shuffle_and_sample(data_list, num2sample, return_the_other=False):
    random.seed(2024)
    random.shuffle(data_list)
    if return_the_other:
        return data_list[:num2sample], data_list[num2sample:]
    else:
        return data_list[:num2sample]

def deduplicate(data_list):
    seen_queries = set()
    unique_data = []

    for item in data_list:
        if item["query"] not in seen_queries:
            unique_data.append(item)  # 添加第一个出现的样本
            seen_queries.add(item["query"])  # 标记这个 query 已经出现
    return unique_data

# load MATH dataset
if args.dataset_name == "MATH":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/MATH"
    dataset = load_dataset(load_dataset_path, "all", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["problem"],
            "gt": example["solution"],
            "tag": ["math", example["type"], example["level"]],
            "source": "MATH"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    sample_pool.extend(data_list)

# load GSM8K dataset
elif args.dataset_name == "GSM8K":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/gsm8k"
    dataset = load_dataset(load_dataset_path, "main", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["question"],
            "gt": example["answer"],
            "tag": ["math"],
            "source": "GSM8K"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    sample_pool.extend(data_list)

# load AQUA-RAT dataset
elif args.dataset_name == "AQUA-RAT":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/aqua_rat"
    dataset = load_dataset(load_dataset_path, "raw", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)   # question / options / rationale / correct
    def format_aqua_rat_query(example):
        query = example["question"]
        query += " Choose the correct answer from the following options:"
        for option in example["options"]:
            query += f"\n{option}"
        return query
    data_list = [
        {
            "query": format_aqua_rat_query(example),
            "gt": example["rationale"],
            "tag": ["math", "reasoning", "multiple-choice"],
            "source": "AQUA-RAT"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    sample_pool.extend(data_list)

# load MedMCQA
elif args.dataset_name == "MedMCQA":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/medmcqa"
    dataset = load_dataset(load_dataset_path, split="validation", trust_remote_code=True)
    filtered_dataset = dataset.filter(lambda example: example['choice_type'] != 'multi')
    print(f"{'='*50}\n", filtered_dataset)
    def format_medmcqa_query(example):
        query = example["question"]
        query += "\n\nChoose the correct answer from the following options:"
        query += f"\n(A) {example['opa']}"
        query += f"\n(B) {example['opb']}"
        query += f"\n(C) {example['opc']}"
        query += f"\n(D) {example['opd']}"
        return query
    def format_medmcqa_gt(example):
        answer_list = [f"(A) {example['opa']}", f"(B) {example['opb']}", f"(C) {example['opc']}", f"(D) {example['opd']}"]
        answer = f"The correct answer is: {answer_list[example['cop']]}"
        return answer
    data_list = [
        {
            "query": format_medmcqa_query(example),
            "gt": format_medmcqa_gt(example),
            "tag": ["medical", example['subject_name'], example['topic_name']],
            "source": "MedMCQA"
        }
        for example in filtered_dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    sample_pool.extend(data_list)

# load MedQA
elif args.dataset_name == "MedQA":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/med_qa"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_medqa_query(example):
        query = example["question"]
        query += " Choose the correct answer from the following options:"
        for option in example["options"]:
            query += f"\n({option['key']}) {option['value']}"
        return query
    def format_medqa_gt(example):
        answer = f"The correct answer is: ({example['answer_idx']}) {example['answer']}"
        return answer
    data_list = [
        {
            "query": format_medqa_query(example),
            "gt": format_medqa_gt(example),
            "tag": ["medical"],
            "source": "MedQA"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    sample_pool.extend(data_list)

# load MMLU
elif args.dataset_name == "MMLU":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/mmlu"
    dataset = load_dataset(load_dataset_path, "all", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)

    def format_mmlu_query(example):
        query = f"""{example["question"]}

For this question, four choices are provided:
(A) {example["choices"][0]}
(B) {example["choices"][1]}
(C) {example["choices"][2]}
(D) {example["choices"][3]}

Please choose the correct answer from the above choices."""
        return query
    
    def format_mmlu_gt(example):
        choice_list = ["A", "B", "C", "D"]
        answer = f"({choice_list[example['answer']]})"
        return answer
    
    data_list = [
        {
            "query": format_mmlu_query(example),
            "gt": format_mmlu_gt(example),
            "tag": ["mmlu", example['subject']],
            "source": "MMLU"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    sample_pool.extend(data_list)

# load MMLU-Pro
elif args.dataset_name == "MMLU-Pro":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/MMLU-Pro"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def format_mmlu_query(example):
        query = "The following is a multiple-choice question:\n"
        query += example["question"]
        query += "\n\nThe following choices are provided:"
        for idx, option in enumerate(example["options"]):
            query += f"\n({option_list[idx]}) {option}"
        query += '\n\nPlease finish your answer with "the answer is (X)" where X is the correct letter choice.'
        return query
    
    def format_mmlu_gt(example):
        answer = f"The answer is ({option_list[example['answer_index']]}) {example['options'][example['answer_index']]}"
        return answer
    
    data_list = [
        {
            "query": format_mmlu_query(example),
            "gt": format_mmlu_gt(example),
            "tag": ["MMLU-Pro", example['category'], example['src']],
            "source": "MMLU-Pro",
            "num_choices": len(example["options"])
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)  # 1001, 2004
    sample_pool.extend(data_list)

# load GSM-Hard dataset
elif args.dataset_name == "GSM-Hard":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/gsm-hard"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["input"],
            "gt": str(example["target"]),
            "tag": ["math", "GSM-Hard"],
            "source": "GSM-Hard"
        }
        for example in dataset
    ]
    data_list_test, data_list_train = shuffle_and_sample(data_list, args.num2sample, return_the_other=True)
    print(f">> A data sample from {args.dataset_name}:\n{data_list_test[0]}")
    sample_pool.extend(data_list_test)

    with open(f"./train/sample_pool/{args.dataset_name}.json", 'w') as output_json:
        json.dump(data_list_train, output_json, indent=4)
        print(f">> {len(data_list_train)} samples are saved to the training pool.")

# load SVAMP dataset
elif args.dataset_name == "SVAMP":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/SVAMP"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["question_concat"],
            "gt": example["Answer"],
            "tag": ["math", "SVAMP", example["Type"]],
            "source": "SVAMP"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

# load ARC dataset
elif args.dataset_name == "ARC":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/ai2_arc"
    dataset = load_dataset(load_dataset_path, "ARC-Challenge", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_arc_query(example):
        query = example["question"]
        query += " Choose the correct answer from the following options:"
        for label, text in zip(example["choices"]["label"], example["choices"]["text"]):
            query += f"\n({label}) {text}"
        return query
    def format_arc_gt(example):
        answer = [f"({label}) {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"]) if label == example["answerKey"]][0]
        return answer
    data_list = [
        {
            "query": format_arc_query(example),
            "gt": format_arc_gt(example),
            "tag": ["ARC"],
            "source": "ARC"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

# load GPQA dataset
elif args.dataset_name.startswith("GPQA"):
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/gpqa"
    if args.dataset_name == "GPQA-Diamond":
        dataset = load_dataset(load_dataset_path, "gpqa_diamond", split="train", trust_remote_code=True)
    else:
        dataset = load_dataset(load_dataset_path, "gpqa_main", split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_gpqa_query(example):
        query = example["Question"]
        query += "\n\nChoose the correct answer from the following options:"
        query += f"\n(A) {example['Correct Answer']}"
        query += f"\n(B) {example['Incorrect Answer 1']}"
        query += f"\n(C) {example['Incorrect Answer 2']}"
        query += f"\n(D) {example['Incorrect Answer 3']}"
        return query
    def format_gpqa_gt(example):
        answer = f"(A) {example['Correct Answer']}"
        return answer
    data_list = [
        {
            "query": format_gpqa_query(example),
            "gt": format_gpqa_gt(example),
            "tag": [args.dataset_name, example["High-level domain"], example["Subdomain"], example["Writer's Difficulty Estimate"]],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

# load SciBench dataset
elif args.dataset_name == "SciBench":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/scibench"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_scibench_gt(example):
        answer = f"{example['answer_number']}, the unit is {example['unit']}."
        return answer
    data_list = [
        {
            "query": example['problem_text'],
            "gt": format_gpqa_gt(example),
            "tag": [args.dataset_name, 'science', example['source']],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == 'HumanEval':
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/openai_humaneval"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_humaneval_query(example):
        query = """Write a Python function that follows the function name and signature provided below."""
        query += f"\n\n{example['prompt']}"
        return query
    data_list = [
        {
            "query": format_humaneval_query(example),
            "gt": example["canonical_solution"],
            "test": example['test'],
            "entry_point": example['entry_point'],
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "HumanEval-Plus":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/humanevalplus"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_humanevalplus_query(example):
        query = """Write a Python function that follows the function name and signature provided below."""
        query += f"\n\n{example['prompt']}"
        return query
    data_list = [
        {
            "query": format_humanevalplus_query(example),
            "gt": example["canonical_solution"],
            "test": example['test'],
            "entry_point": example['entry_point'],
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "MBPP":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/mbpp"
    dataset = load_dataset(load_dataset_path, "full", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_mbpp_query(example):
        test_list = example["test_list"]
        example_test = test_list[0]
        query = f"{example['text']}\nHere is an example test:\n{example_test}\nMake sure your function aligns with the function signature and usage in the example test."
        return query
    data_list = [
        {
            "query": format_mbpp_query(example),
            "gt": example["code"],
            "test_cases": example["test_list"]+example["challenge_test_list"],
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "MBPP-Plus":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/mbppplus"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def extract_function_signature(text):
        # Extract the function signature
        # pattern = r'def\s+(.*?)\s*\('
        pattern = r'assert\s+([a-zA-Z_]\w*)\('
        match = re.search(pattern, text)
        return match.group(1) if match else None
    data_list = []
    for example in dataset:
        test_list = example["test_list"]
        gt = example["code"]
        function_signature = extract_function_signature(test_list[0])
        if not function_signature:
            continue
        query = example["prompt"] + "\n\nHere are a few example test cases:\n" + "\n".join(test_list) + "\n\nMake sure your function aligns with the function signature and usage in the example test cases."
        test_code = example["test"]
        if "for i, (inp, exp) in enumerate(zip(inputs, results)):" not in test_code:
            continue
        index = test_code.find("for i, (inp, exp) in enumerate(zip(inputs, results)):")
        test_code = test_code[:index]
        data_list.append(
            {
                "query": query,
                "gt": gt,
                "function_signature": function_signature,
                "test_code": test_code,
                "tag": ["code", args.dataset_name],
                "source": args.dataset_name
            }
        )
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "FullStackBench":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/FullStackBench"
    dataset = load_dataset(load_dataset_path, "en", split="test", trust_remote_code=True)
    def filter_function(example):
        language = example["labels"]["execution_language"]
        return language == "python"

    dataset = dataset.filter(filter_function)
    print(f"{'='*50}\n", dataset)

    def format_fullstackbench_test_cases(example):
        test = example["test"]
        assert_statements = [f"assert {s.strip()}" for s in test["code"].split("assert") if s.strip()][1:]
        return assert_statements
    data_list = [
        {
            "query": example["content"],
            "gt": example["canonical_solution"],
            "test_cases": format_fullstackbench_test_cases(example),
            "tag": ["code", args.dataset_name],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)
    
elif args.dataset_name == "DS-1000":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/DS-1000"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_ds_1000_query(example):
        query = example["prompt"]
        suffix = " BEGIN SOLUTION <code>"
        if query.endswith(suffix):
            query = query[:-len(suffix)]
        query += "\n\nYour code will go through a test. In the test, the variables mentioned in the problem are already defined. Your code should directly operate on these variables without re-defining them or changing their names and store the result in the result variable. Make sure your code can be directly executed."
        return query
    data_list = [
        {
            "query": format_ds_1000_query(example),
            "gt": example["reference_code"],
            "test_case_cnt": example["metadata"]["test_case_cnt"],
            "test_code": example["code_context"],
            "tag": ["code", args.dataset_name],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)
elif args.dataset_name == "LiveCodeBench":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/LiveCodeBench" # the json path
    with open(load_dataset_path, "r",encoding='utf-8') as f:
        data_list = json.load(f)
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name in ["EvoEval_difficult", "EvoEval_creative", "EvoEval_subtle", "EvoEval_combine"]:
    load_dataset_path = f"/mnt/petrelfs/yerui/mac/datasets/{args.dataset_name}"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_evoeval_query(example):
        query = """Write a Python function that follows the function name and signature provided below."""
        query += f"\n\n{example['prompt']}"
        return query
    data_list = [
        {
            "query": format_evoeval_query(example),
            "gt": example["canonical_solution"],
            "test": example['test'],
            "entry_point": example['entry_point'],
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "BigCodeBench":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/bigcodebench"
    dataset = load_dataset(load_dataset_path, split="v0.1.3", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["instruct_prompt"],
            "gt": example["code_prompt"] + example["canonical_solution"],
            "test": example["test"],
            "tag": [args.dataset_name, "code"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "AIME_2024":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/AIME_2024"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["Problem"],
            "gt": str(example["Answer"]),
            "tag": [args.dataset_name, "math"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)
    
elif args.dataset_name == "FinQA":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/finqa"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    def remove_space(text_in):
        res = []

        for tmp in text_in.split(" "):
            if tmp != "":
                res.append(tmp)

        return " ".join(res)
    def table_row_to_text(header, row):
        '''
        use templates to convert table row to text
        '''
        res = ""

        for head, cell in zip(header[1:], row[1:]):
            res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")

        res = remove_space(res)
        return res.strip()
    def format_FinQA_query(example):
        # no retriever, use longformer
        print("\nexample", example)
        question = example["question"]
        context = ""
        table = example["table"]
        table_text = ""
        for row in table[1:]:
            this_sent = table_row_to_text(table[0], row)
            table_text += this_sent

        context = " ".join(example["pre_text"]) + " " + \
            " ".join(example["post_text"]) + " " + table_text
        context = context.strip()
        # process "." and "*" in text
        context = context.replace(". . . . . .", "")
        context = context.replace("* * * * * *", "")    
        query = question + "\n" + context.strip()
        return query
    print(f"{'='*50}\n", dataset)
    
    data_list = [
        {
            "query": format_FinQA_query(example),
            "gt": str(example["answer"]),
            "tag": [args.dataset_name, "finance"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)
    
elif args.dataset_name == "FinanceBench":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/financebench"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    def format_financebench_query(example):
        # no retriever, use longformer
        print("\nexample", example)
        question = example["question"]
        context = "\n\n".join([evidence["evidence_text_full_page"] for evidence in example["evidence"]]) 
        query = f"Answer this question: {question} \nHere is the relevant evidence that you need to answer the question:\n[START OF FILING] {context} [END OF FILING]"
        return query
    print(f"{'='*50}\n", dataset)
    
    data_list = [
        {
            "query": format_financebench_query(example),
            "gt": str(example["answer"]),
            "tag": [args.dataset_name, "finance"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "FPB":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/financial_phrasebank"
    dataset = load_dataset(load_dataset_path, 'sentences_allagree', trust_remote_code=True)
    def format_FPB_query(example):
        # no retriever, use longformer
        print("\nexample:", example)
        question = example["sentence"]
        query = f"Predicting the sentiment based on the news headlines: {question} \nThe sentiment can only be negative, neutral or positive."
        return query
    def format_FPB_ans(example):
        # no retriever, use longformer
        if example["label"] == 0:
            ans = "negative"
        elif example["label"] == 1:
            ans = "neutral"
        elif example["label"] == 2:
            ans = "positive"      

        return ans
    print(f"{'='*50}\n", dataset)
    
    data_list = [
        {
            "query": format_FPB_query(example),
            "gt": format_FPB_ans(example),
            "tag": [args.dataset_name, "finance"],
            "source": args.dataset_name
        }
        for example in dataset['train']
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)
    
elif args.dataset_name == "PubMedQA":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/PubMedQA"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    def format_FPB_query(example):
        # no retriever, use longformer
        print("\nexample:", example)
        question = example["sentence"]
        query = f"Predicting the sentiment based on the news headlines: {question} \nThe sentiment can only be negative, neutral or positive."
        return query
    print(f"{'='*50}\n", dataset)
    
    data_list = [
        {
            "query": example["instruction"] + "\n" + example["input"],
            "gt": example['output'],
            "tag": [args.dataset_name, "medical"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "SciEval":
    load_dataset_path = '/GPFS/rhome/qiminwu/Scieval/scieval-test-local.json'
    # dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    dataset = load_dataset("json", data_files = load_dataset_path, trust_remote_code=True)
    print(f"{'='*50}\n", dataset)

    def scieval_query(example):
        query = example['prompt']
        query += "Then the question is:"
        query += example["question"]
        return query
    def process_answer(example):
        answer = example['answer']
        if len(answer) > 1:
            print(f"Warning: The answer is too long: {answer}")
        return str(answer[0])
    data_list = [
        {
            "query": scieval_query(example),
            "gt": process_answer(example),
            "tag": [args.dataset_name, example['category'], example['category']+"_" + example['task_name']],
            "source": args.dataset_name
        }
        for example in dataset['train']
    ]

    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "SciKnowEval":
    from collections import defaultdict

    load_dataset_path = '/mnt/petrelfs/yerui/mac/datasets/SciKnowEval'
    subset_names = ['Biology', 'Chemistry', 'Material', 'Physics']
    random.seed(2024)

    subset_to_examples = defaultdict(list)

    for subset in subset_names:
        print(f"Loading subset: {subset}")
        full_dataset = list(load_dataset(load_dataset_path, subset, split="test", trust_remote_code=True))

        if len(full_dataset) < 200:
            print(f"Warning: Subset {subset} only has {len(full_dataset)} examples.")

        selected_questions = set()
        sampled_examples = []

        attempts = 0
        max_attempts = 1000
        while len(sampled_examples) < 200 and attempts < max_attempts:
            candidate = random.choice(full_dataset)
            question = candidate['question']
            if question not in selected_questions:
                sampled_examples.append(candidate)
                selected_questions.add(question)
            attempts += 1

        if len(sampled_examples) < 200:
            print(f"Warning: Could only get {len(sampled_examples)} unique samples for {subset}")
        else:
            print(f"Got 200 unique examples for {subset}")

        subset_to_examples[subset] = sampled_examples

    all_examples = []
    for subset in subset_names:
        all_examples.extend(subset_to_examples[subset])

    def sciknoweval_query(example):
        query = example['prompt']['default'] + "\nThe question is:" + example['question']
        choices = example.get("choices", {})
        if choices and choices.get("text"):
            choice_lines = [f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])]
            query += "\n" + "\n".join(choice_lines)
        return query

    def sciknoweval_gt(example):
        choices = example.get("choices", {})
        if choices and choices.get("text"):
            return example["answerKey"]
        else:
            return example["answer"]

    data_list = []
    for example in all_examples:
        try:
            data_list.append({
                "query": sciknoweval_query(example),
                "gt": sciknoweval_gt(example),
                "tag": [args.dataset_name, example['domain'], example['details']['level'], example['details']['task']],
                "source": args.dataset_name
            })
        except Exception as e:
            print(f"Error processing example: {example}")
            print(f"Exception: {e}")

    print(f">> Total collected: {len(data_list)}")
    assert len(data_list) == 800, f"Expected 800 samples, got {len(data_list)}"
    
    random.shuffle(data_list)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "MATH-500":
    load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/MATH-500"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["problem"],
            "gt": example["answer"],
            "tag": [args.dataset_name, "math", example["subject"], f"Level {example['level']}", ],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "MATH-MAS":
    load_dataset_path = "/home/ma-user/modelarts/work/ruiye/xiangruiliu/ReSo-main/datasets/MATH-MAS"
    # load_dataset_path = '/GPFS/rhome/qiminwu/Scieval/scieval-test-local.json'
    # dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    # dataset = load_dataset("json", data_files = load_dataset_path, trust_remote_code=True)
    file_names = [
        "MATH-MAS-Easy.json",
        "MATH-MAS-Medium.json",
        "MATH-MAS-Hard.json"
    ]
    import json
    dataset = []

    for fname in file_names:
        file_path = os.path.join(load_dataset_path, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            dataset.extend(data)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["problem_text_sort"],
            "gt": example["answer_number"],
            "tag": [args.dataset_name, "math", example["source"], f"complexity {example['complexity']}", example["Q_ID"]],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "AIME2025":
    load_dataset_path = "/home/ma-user/modelarts/work/ruiye/xiangruiliu/AIME2025"
    dataset1 = load_dataset(load_dataset_path, 'AIME2025-I', split = 'test',  trust_remote_code=True)
    dataset2 = load_dataset(load_dataset_path, 'AIME2025-II', split = 'test', trust_remote_code=True)
    dataset = concatenate_datasets([dataset1, dataset2])
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["question"],
            "gt": str(example["answer"]),
            "tag": [args.dataset_name, "math"],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)

elif args.dataset_name == "Scibench-MAS":
    load_dataset_path = "/home/ma-user/modelarts/work/ruiye/xiangruiliu/ReSo-main/datasets/Scibench-MAS"
    # load_dataset_path = '/GPFS/rhome/qiminwu/Scieval/scieval-test-local.json'
    # dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    # dataset = load_dataset("json", data_files = load_dataset_path, trust_remote_code=True)
    file_names = [
        "Scibench-MAS-Easy.json",
        "Scibench-MAS-Medium.json",
        "Scibench-MAS-Hard.json"
    ]
    import json
    dataset = []

    for fname in file_names:
        file_path = os.path.join(load_dataset_path, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            dataset.extend(data)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["problem_text_sort"],
            "gt": example["answer_number"],
            "tag": [args.dataset_name, "math", example["source"], f"complexity {example['complexity']}", example["Q_ID"]],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)
    print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
    sample_pool.extend(data_list)


# # load MBPP dataset
# elif args.dataset_name == "MBPP":
#     load_dataset_path = "/mnt/petrelfs/yerui/mac/datasets/mbpp"
#     dataset = load_dataset(load_dataset_path, "full", split="test", trust_remote_code=True)
#     print(f"{'='*50}\n", dataset)
#     data_list = [
#         {
#             "query": example["text"],
#             "gt_test": example["test_list"],
#             "gt_test_challenge": example["challenge_test_list"],
#             "task_id": example["task_id"],
#             "tag": ["code", args.dataset_name],
#             "source": args.dataset_name
#         }
#         for example in dataset
#     ]
#     data_list = shuffle_and_sample(data_list, num2sample)
#     print(f">> A data sample from {args.dataset_name}:\n{data_list[0]}")
#     sample_pool.extend(data_list)

sample_pool = deduplicate(sample_pool)
print(f">> A data sample from the pool:\n{sample_pool[0]}")

print(f"{'='*50}\n There are {len(sample_pool)} queries in the pool.")

with open(save_path, 'w') as output_json:
    json.dump(sample_pool, output_json, indent=4)