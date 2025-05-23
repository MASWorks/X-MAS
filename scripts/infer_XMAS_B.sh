#!/bin/bash

config_path="./configs/X-MAS_B_config.json"
TEST_DATASET_NAMES=("MedQA" "MedMCQA")
model_names=("deepseek-r1-distill-qwen-32b" "llama-3.3-70b-instruct" "qwen2.5-32b-instruct")


run_direct() {
  local model_name=$1
  python save_expri_log.py --messages "start direct $model_name"
  python X-MAS-Bench/infer_direct.py --model_name $model_name --model_config $config_path --test_dataset_names "${TEST_DATASET_NAMES[@]}"
  python save_expri_log.py --messages "end direct $model_name"
}

run_aggregate() {
  local model_name=$1
  python save_expri_log.py --messages "start aggregate $model_name"
  python X-MAS-Bench/infer_aggregate.py --model_name $model_name --model_config $config_path --test_dataset_names "${TEST_DATASET_NAMES[@]}"
  python save_expri_log.py --messages "end aggregate $model_name"
}


run_revise() {
  local model_name=$1
  python save_expri_log.py --messages "start revise $model_name"
  python X-MAS-Bench/infer_revise.py --model_name $model_name --model_config $config_path --test_dataset_names "${TEST_DATASET_NAMES[@]}"
  python save_expri_log.py --messages "end revise $model_name"
}

run_evaluate() {
  local model_name=$1
  python save_expri_log.py --messages "start evaluate $model_name"
  python X-MAS-Bench/infer_evaluate.py --model_name $model_name --model_config $config_path --test_dataset_names "${TEST_DATASET_NAMES[@]}"
  python save_expri_log.py --messages "end evaluate $model_name"
}

run_plan() {
  local model_name=$1
  python save_expri_log.py --messages "start plan $model_name"
  python X-MAS-Bench/infer_plan.py --model_name $model_name --model_config $config_path --test_dataset_names "${TEST_DATASET_NAMES[@]}"
  python save_expri_log.py --messages "end plan $model_name"
}

run_function() {
  local model_name=$1
  run_direct "$model_name" 
  run_aggregate "$model_name"
  run_plan "$model_name"
  run_revise "$model_name" 
  run_evaluate "$model_name" 
}

for model_name in "${model_names[@]}"; do
  run_function "$model_name" &
done

wait

