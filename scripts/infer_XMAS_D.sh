model_api_config=./configs/model_config.json
model_name=gpt-4o-mini-2024-07-18

# ==================================
TEST_DATASET_NAMES=("GSM-Hard" "MATH-500" "AQUA-RAT" "AIME_2024" "MBPP-Plus" "MBPP" "HumanEval" "HumanEval-Plus" "MedQA" "MedMCQA" "PubMedQA" "FinanceBench" "FinQA" "FPB" "SciEval" "SciKnowEval" "SciBench" "GPQA" "GPQA-Diamond" "MMLU-Pro" "MMLU")


METHOD_NAME_LIST=(
    hmas
    llm_debate
    dylan
    agentverse
)

for test_dataset_name in "${TEST_DATASET_NAMES[@]}"; do
    for method_name in "${METHOD_NAME_LIST[@]}"; do
        python XMAS-Bench/inference_XMAS.py --method_name $method_name --model_name $model_name --test_dataset_name $test_dataset_name --model_api_config $model_api_config
    done
done

wait


