#!/bin/bash

config_path="./configs/X-MAS_Bench_config.json"
# TEST_DATASET_NAMES=("MedQA" "MedMCQA")
# model_names=("deepseek-r1-distill-qwen-32b" "llama-3.3-70b-instruct" "qwen2.5-32b-instruct")
TEST_DATASET_NAMES=("AIME-2024")
model_names=("qwen-2.5-32b-instruct")

run_direct() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_direct.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}

run_aggregate() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_aggregate.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}


run_revise() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_revise.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}

run_evaluate() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_evaluate.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}

run_plan() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_plan.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}

run_function() {
  local model_name=$1 dataset_name=$2
  run_direct "$model_name" "$dataset_name"
  run_aggregate "$model_name" "$dataset_name"
  run_plan "$model_name" "$dataset_name"
  run_revise "$model_name" "$dataset_name"
  run_evaluate "$model_name" "$dataset_name"
}
for dataset_name in "${TEST_DATASET_NAMES[@]}"; do
  for model_name in "${model_names[@]}"; do
    run_function "$model_name" "$dataset_name" &
  done
done

wait

