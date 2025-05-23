from .xmasp import XMASP_MAIN
from .mas_base import MAS
from .llm_debate import LLMDebate, LLMDebate_MATH, LLMDebate_GSM, LLMDebate_MMLU
from .DyLAN import DyLAN_Main, DyLAN_MMLU, DyLAN_Humaneval, DyLAN_MATH
from .agentverse import Agentverse_HumanEval, Agentverse_MGSM, Agentverse_MAIN

method2class = {
    "xmasp": XMASP_MAIN,
    "llm_debate_math": LLMDebate_MATH,
    "llm_debate_gsm": LLMDebate_GSM,
    "llm_debate_mmlu": LLMDebate_MMLU,
    "llm_debate": LLMDebate,
    "dylan":DyLAN_Main,
    "dylan_mmlu":DyLAN_MMLU,
    "dylan_humaneval":DyLAN_Humaneval,
    "dylan_math":DyLAN_MATH,
    "agentverse_humaneval": Agentverse_HumanEval,
    "agentverse_mgsm": Agentverse_MGSM,
    "agentverse": Agentverse_MAIN
}

def get_method_class(method_name, dataset_name=None):
    # lowercase the method name
    method_name = method_name.lower()
    
    all_method_names = method2class.keys()
    matched_method_names = [sample_method_name for sample_method_name in all_method_names if method_name in sample_method_name]
    
    if len(matched_method_names) > 0:
        if dataset_name is not None:
            # lowercase the dataset name
            dataset_name = dataset_name.lower()
            # check if there are method names that contain the dataset name
            matched_method_data_names = [sample_method_name for sample_method_name in matched_method_names if sample_method_name.split('_')[-1] in dataset_name]
            # matched_method_data_names = [sample_method_name for sample_method_name in matched_method_names if dataset_name in  sample_method_name]
            if len(matched_method_data_names) > 0:
                method_name = matched_method_data_names[0]
                if len(matched_method_data_names) > 1:
                    print(f"[WARNING] Found multiple methods matching {dataset_name}: {matched_method_data_names}. Using {method_name} instead.")
    else:
        raise ValueError(f"[ERROR] No method found matching {method_name}. Please check the method name.")
    print(f"[INFO] Using method: {method2class[method_name]}")
    return method2class[method_name]