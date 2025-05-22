import os
import importlib

module_list = [
    "0_cot_code",
    "1_cot_general",
    "2_5cot_sc_code",
    "3_5cot-sc_general",
    "4_reflection",
    "5_3llmdebate_code",
    "6_llmDebate_general",
    "7_sc-ensemble_math_aflow",
    "8_ensemble-test-fix_aflow",
    "9_ensemble-format_aflow",
    "10_quality-diversity",
    "11_step-back-abstraction_adas",
    "12_dynamic_agent",
    "13_heuristic-simulation-refine_adas",
    "14_priority-refine_adas",
    "15_questioning_adas",
    "16_test-and-refine_adas",
    "18_medical_medagent",
    "19_financial_reflection",
    "20_financial_2reflection",
    "21_physics_mechagent",
]

mas_func_dict = {}

for module_name in module_list:
    try:
        module = importlib.import_module(module_name)
        
        # 检查模块是否有 forward 函数
        if hasattr(module, "forward"):
            # 调用 forward 函数
            forward_func = getattr(module, "forward")
            if callable(forward_func):
                print(f"Success")
                mas_func_dict[module_name] = forward_func
            else:
                print(f"'forward' in {module_name} is not callable.")
        else:
            print(f"No 'forward' function in {module_name}.")
    except Exception as e:
        print(f"Failed to load or execute {module_name}: {e}")
  

