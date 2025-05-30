model_api_config=./configs/X-MAS_Design_config.json
TEST_DATASET_NAMES=("AIME-2024")
# TEST_DATASET_NAMES=("GSM-Hard" "MATH-500" "AQUA-RAT" "AIME-2024" "MBPP-Plus" "MBPP" "HumanEval" "HumanEval-Plus" "MedQA" "MedMCQA" "PubMedQA" "FinanceBench" "FinQA" "FPB" "SciEval" "SciKnowEval" "SciBench" "GPQA" "GPQA-Diamond" "MMLU-Pro" "MMLU")

METHOD_NAME_LIST=(
    x_mas_proto
    llm_debate
    # dylan
    # agentverse
)

for test_dataset_name in "${TEST_DATASET_NAMES[@]}"; do
    for method_name in "${METHOD_NAME_LIST[@]}"; do
        python X-MAS-Design/inference_mas.py --model_api_config $model_api_config --method_name $method_name --test_dataset_name $test_dataset_name
    done
done

wait

# ==================================
# If you want to run llm_debate with your own config or settings, you can uncomment the following lines and modify them accordingly.
# ==================================

# model_api_config=./configs/X-MAS_Design_config.json

# method_config_name=config_math

# test_dataset_name="math-500"

# method_name="llm_debate"


# python X-MAS-Design/inference_mas.py --model_api_config $model_api_config --method_name $method_name --method_config_name $method_config_name --test_dataset_name $test_dataset_name


# wait
