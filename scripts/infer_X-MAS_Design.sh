model_api_config=./configs/X-MAS_Design_config.json
model_name=qwen-2.5-32b-instruct

# ==================================
# TEST_DATASET_NAMES=("GSM-Hard" "MATH-500" "AQUA-RAT" "AIME-2024" "MBPP-Plus" "MBPP" "HumanEval" "HumanEval-Plus" "MedQA" "MedMCQA" "PubMedQA" "FinanceBench" "FinQA" "FPB" "SciEval" "SciKnowEval" "SciBench" "GPQA" "GPQA-Diamond" "MMLU-Pro" "MMLU")
TEST_DATASET_NAMES=("AIME-2024")

METHOD_NAME_LIST=(
    x_mas_proto
    # llm_debate
    # dylan
    # agentverse
)

for test_dataset_name in "${TEST_DATASET_NAMES[@]}"; do
    for method_name in "${METHOD_NAME_LIST[@]}"; do
        python X-MAS-Design/inference_mas.py --method_name $method_name --model_name $model_name --test_dataset_name $test_dataset_name --model_api_config $model_api_config
    done
done

wait


