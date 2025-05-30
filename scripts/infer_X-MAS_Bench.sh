#!/bin/bash

# ==================================
# If you want to infer with **revise, aggregation or evaluation**, please make sure you have replace the content of "./X-MAS-Bench/results/" with **source file** in [Google Drive](https://drive.google.com/file/d/1oukYZLDOuc98i-ICkoZ6OYME9a7-AuH1/view?usp=drive_link) first. You can download the .zip file named results.zip to the "./X-MAS-Bench/results/" path and unzip it.

#If you want to infer with **planning**, please make sure you have load the default source model **"llama-3.1-8b-instruct", "qwen-2.5-7b-instruct" and "qwen-2.5-14b-instruct"**.

# ==================================

config_path="./configs/X-MAS_Bench_config.json"
TEST_DATASET_NAMES=("AIME-2024")
model_names=("qwen-2.5-32b-instruct")

run_qa() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_qa.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}

run_aggregation() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_aggregation.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}


run_revise() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_revise.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}

run_evaluation() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_evaluation.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}

run_planning() {
  local model_name=$1 dataset_name=$2
  python X-MAS-Bench/infer_planning.py --model_name $model_name --model_config $config_path --test_dataset_name $dataset_name
}

run_function() {
  local model_name=$1 dataset_name=$2
  run_qa "$model_name" "$dataset_name"
  run_aggregation "$model_name" "$dataset_name"
  run_planning "$model_name" "$dataset_name"
  run_revise "$model_name" "$dataset_name"
  run_evaluation "$model_name" "$dataset_name"
}

for dataset_name in "${TEST_DATASET_NAMES[@]}"; do
  for model_name in "${model_names[@]}"; do
    run_function "$model_name" "$dataset_name" &
  done
done

wait

