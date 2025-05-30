#!/bin/bash

config_path="./configs/X-MAS_Bench_config.json"
# TEST_DATASET_NAMES=("GSM-Hard" "MATH-500" "AQUA-RAT" "AIME-2024" "MBPP-Plus" "MBPP" "HumanEval" "HumanEval-Plus" "MedQA" "MedMCQA" "PubMedQA" "FinanceBench" "FinQA" "FPB" "SciEval" "SciKnowEval" "SciBench" "GPQA" "GPQA-Diamond" "MMLU-Pro" "MMLU")
TEST_DATASET_NAMES=("AIME-2024")
MODEL_NAMES=("qwen-2.5-32b-instruct")
FUNCTION_NAMES=("qa" "aggregation" "planning" "revise" "evaluation")



for dataset_name in "${TEST_DATASET_NAMES[@]}"; do
  for model_name in "${MODEL_NAMES[@]}"; do
    for function_name in "${FUNCTION_NAMES[@]}"; do
      python X-MAS-Bench/eval_bench.py --eval_model_name llama-3.1-70b-instruct --model_config $config_path --dataset_name $dataset_name --model_name $model_name --function_name $function_name --eval_mode bench-test
    done
  done
done
# wait
