#!/bin/bash

# 配置文件和数据集名称
config_path="./configs/X-MAS_Bench_config.json"
TEST_DATASET_NAMES=("GSM-Hard" "MATH-500" "AQUA-RAT" "AIME_2024" "MBPP-Plus" "MBPP" "HumanEval" "HumanEval-Plus" "MedQA" "MedMCQA" "PubMedQA" "FinanceBench" "FinQA" "FPB" "SciEval" "SciKnowEval" "SciBench" "GPQA" "GPQA-Diamond" "MMLU-Pro" "MMLU")
# TEST_DATASET_NAMES=("AIME_2024" "FinanceBench" "FinQA" "FPB" "MBPP-Plus" "MBPP")
# TEST_DATASET_NAMES=("AIME_2024")


infer_names=(
  "qwen-2.5-7b-instruct_direct.jsonl"
  "qwen-2.5-14b-instruct_direct.jsonl"
  "qwen-2.5-32b-instruct_direct.jsonl"
  "qwen-2.5-72b-instruct_direct.jsonl"
  "qwen-2.5-coder-7b-instruct_direct.jsonl"
  "qwen-2.5-coder-14b-instruct_direct.jsonl"
  "qwen-2.5-coder-32b-instruct_direct.jsonl"
  "qwen-2.5-math-7b-instruct_direct.jsonl"
  "qwen-2.5-math-72b-instruct_direct.jsonl"
  "llama-3.1-8b-instruct_direct.jsonl"
  "llama-3.1-70b-instruct_direct.jsonl"
  "chemdfm-v1.5-8b_direct.jsonl"
  "llama3-xuanyuan3-70b-chat_direct.jsonl"
  "llama3-openbiollm-70b_direct.jsonl"
)


for infer_name in "${infer_names[@]}"; do
  python save_expri_log.py --messages "start eval $infer_name"
  python eval_output_x-mas-design.py --model_name llama-3.1-70b-instruct --model_config $config_path --dataset_names "${TEST_DATASET_NAMES[@]}" --infer_name $infer_name --eval_mode bench-test
  python save_expri_log.py --messages "end eval $infer_name"
done

# wait
