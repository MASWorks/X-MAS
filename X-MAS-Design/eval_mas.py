import json
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
import os
import random
import re
import sys
import requests


import argparse

def get_answer_prompt(task, operated_text):
    message_to_check = "===Problem: " + task + f"\n\n===Solution: {operated_text}"

    prompt = f"""You are a helpful AI assistant tasked with extracting the final answer from a provided solution.

**Input:**
1. A problem statement, prefixed with "===Problem: <problem>".
2. A solution to the problem, prefixed with "===Solution:".

**Problem and Solution:**
{message_to_check}

**Instructions:**
- Carefully analyze the solution and extract the final answer in reply: "The answer is <answer extracted> in reply".
- If the solution does not contain a final answer (e.g., only reasoning, code without execution, or incomplete information), respond with: "The reply doesn't contain an answer."
- Ensure that the extracted answer is exactly as presented in the solution. Do not infer or use external knowledge. Do not execute the code yourself.
- Remember, Never execute the code yourself! Never doing any computation yourself! Just extract and output the existing answer!
"""
    return prompt

def get_eval_prompt(task, operated_text, ground_truth):
        message_to_check = "===Problem: " + str(task) + f"\n\n===Ground truth answer: "+  str(ground_truth) + f"\n\n===Reply: {str(operated_text)}"
        
        prompt = f"""You are a helpful AI assistant. You will use your coding and language skills to verify the answer.
You are given:
    1. A problem, which is going to start like "===Problem: <problem>".
    2. A ground truth answer, which is going to start like "===Ground truth answer:".
    3. A reply with the answer to the problem, which are going to start like "===Reply:".
Please do the following:
1. Extract the answer in reply: "The answer is <answer extracted> in reply".
2. Check whether the answer in reply matches the ground truth answer. When comparison is not obvious (for example, 3*\\sqrt(6) and 7.348), you may compare by calculation, allowing a small margin of error.
3. After everything is done, please give each reply a comment like the following options:
    - "The answer is correct."
    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The reply doesn't contain an answer." 
Here are the promblem, the ground truth answer and the reply:
{message_to_check}
    """
        return prompt

from tenacity import retry, wait_exponential, stop_after_attempt, RetryError

def handle_retry_error(retry_state):
    # print('here ')
    return None

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
def call_llm(prompt, temperature, model_url_list, model_name):
    if "gpt" in model_name:
        payload_dict = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0    # for evaluation, we don't need temperature
        }
        payload = json.dumps(payload_dict)
        headers = {
            'Authorization': '',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': '',
            'Connection': 'keep-alive'
            }
        result = requests.request("POST", "http://./v1/chat/completions", headers=headers, data=payload)
        response = result.json()["choices"][0]["message"]["content"]
    else:
        model_url = random.choice(model_url_list)
        llm = OpenAI(base_url=f"{model_url}", api_key="EMPTY")
        completion = llm.chat.completions.create(
            model=f"{model_name}",
            messages=[{"role": "user", "content": prompt}],
            stop=['<|eot_id|>'],
            temperature=temperature,
            max_tokens=2048
        )
        response = completion.choices[0].message.content
    return response

def eval_llm_v2(item, model_url_list, model_name, source_data, temperature=0.0):
    prompt = get_answer_prompt(item['query'], item['generated_output'])
    # print(prompt)
    # print('------------')
    retries = 0
    while retries < 5:
        try:
            answer = call_llm(prompt, temperature, model_url_list, model_name)
            # print(answer)
            break
        except Exception as e:
            retries += 1
            if retries == 5:
                print(f"After 5 retries, request failed with error: {e}")
                return "calling llm error", None


    prompt = get_eval_prompt(item['query'], answer, item['gt'])
    retries = 0
    while retries < 5:
        try:
            response = call_llm(prompt, temperature, model_url_list, model_name)
            break
        except Exception as e:
            retries += 1
            if retries == 5:
                print(f"After 5 retries, request failed with error: {e}")
                return "calling llm error", None

    if "The answer is correct." in response:
        score = 2
    elif "The answer is approximated but should be correct." in response:
        score = 1
    else:
        score = 0
    return response, score

def eval_llm(item, model_url_list, model_name, source_data, temperature=0.0):
    prompt = get_eval_prompt(item['query'], item['generated_output'], item['gt'])
    retries = 0
    while retries < 5:
        try:
            response = call_llm(prompt, temperature, model_url_list, model_name)
            break
        except Exception as e:
            retries += 1
            if retries == 5:
                print(f"After 5 retries, request failed with error: {e}")
                return "calling llm error", None

    if "The answer is correct." in response:
        score = 2
    elif "The answer is approximated but should be correct." in response:
        score = 1
    else:
        score = 0
    return response, score

def eval_mmlupro(item, model_url_list, model_name, source_data):

    def extract_answer(text):
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            # print("1st answer extract failed\n" + text)
            return extract_again(text)

    def extract_again(text):
        match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
        if match:
            return match.group(1)
        else:
            return extract_final(text)

    def extract_final(text):
        pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return None
    
    extracted_pred = extract_answer(item['generated_output'])
    extracted_gt = extract_answer(item['gt'])

    option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    response = f"Predicted: {extracted_pred} | Correct: {extracted_gt}"
    if extracted_pred is None:      # No answer extracted, then randomly select one
        random.seed(12345)
        x = random.randint(0, item["num_choices"] - 1)
        extracted_pred = option_list[x]
        response = f"No answer extracted. Random: {extracted_pred} | Correct: {extracted_gt}"
    if str(extracted_pred) == str(extracted_gt):
        score = 2
    else:
        score = 0    
    
    return response, score

def extract_code_with_llm(text, model_url_list, model_name, temperature=0.1):
    prompt = f"""Extract the final code solution from the following content. Only output the code, without any additional explanation or content. Do not modify any part of the code.

Content to be extracted:
{text}
"""
    retries = 0
    while retries < 3:
        try:
            response = call_llm(prompt, temperature, model_url_list, model_name)
            code = re.sub(r"^```(?:\w+)?\n?|```$", "", response, flags=re.MULTILINE).strip()
            return code
        except Exception as e:
            retries += 1
            if retries == 3:
                print(f"After 3 retries, request failed with error: {e}")
                return None

def extract_code_function_llm(query, solution, model_url_list, model_name, temperature=0.1):
    prompt = f"""You are given a **Problem** and a **Solution**. The **Problem** asks for a code function. Extract the final code function from the **Solution**.
**Problem:**
{query}

**Solution:**
{solution}

Please follow the following rules:
- Only output the code function that exists in the **Solution**, without any additional explanation or content.
- Do not modify any part of the code function.
- Remove parts like 'example use' or 'test cases'.
- If the **Solution** does not contain a code function, respond with: "The reply doesn't contain a code function."
"""
    retries = 0
    while retries < 3:
        try:
            response = call_llm(prompt, temperature, model_url_list, model_name)
            code = re.sub(r"^```(?:\w+)?\n?|```$", "", response, flags=re.MULTILINE).strip()
            return code
        except Exception as e:
            retries += 1
            if retries == 3:
                print(f"After 3 retries, request failed with error: {e}")
                return None

def execute_code(code, temp_dir="mas_workspace_1/mas_workspace_2/mas_workspace_3", timeout=10):
    """
    Executes a given code string in a temporary directory and captures print statements 
    in the output. Cleans up the directory after execution.

    Args:
        code (str): A string containing Python code. The code is expected to define a 
                    variable named 'output' whose value will be retrieved and returned.
        temp_dir (str): The directory in which the code will be executed.
        timeout (int): Maximum time (in seconds) allowed for code execution.

    Returns:
        str: The value of the 'output' variable and captured print statements as a string.
             If 'output' is not defined, returns "None".
             If there is an error during execution, returns the error message as a string.
             If execution times out, returns "Execution Time Out".
    """
    if not code:
        return "Empty code. No output."

    # Ensure the temp directory exists
    original_dir = os.getcwd()
    temp_dir_path = os.path.join(original_dir, temp_dir)
    os.makedirs(temp_dir_path, exist_ok=True)
    
    def execute(queue):
        try:
            # Change to the temp directory
            os.chdir(temp_dir_path)
            
            # Local dictionary to store variables during code execution
            local_context = {}
            
            # Capture print output
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                exec(code, {}, local_context)
                captured_output = buf.getvalue()
            
            # Retrieve the 'output' variable
            output = local_context.get("output", "None")
            
            # Combine output and captured print statements
            result = f"Final output:{output}\nPrint during execution:{captured_output}\n".strip()
            queue.put(result)  # Send the result back via the queue
        except Exception as e:
            queue.put(f"Error: {str(e)}")  # Send error message to the queue
        finally:
            os.chdir(original_dir)  # Restore the original directory

    # Create a queue for inter-process communication
    queue = multiprocessing.Queue()

    # Create a separate process to execute the code
    process = multiprocessing.Process(target=execute, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # If the process is still running after the timeout, terminate it
        process.terminate()
        process.join()
        return "Execution Time Out"

    # Retrieve the result from the queue
    try:
        result = queue.get_nowait()  # Get the result from the queue
    except multiprocessing.queues.Empty:
        result = "None"  # Default result if the queue is empty

    # Clean up the temp directory
    shutil.rmtree(temp_dir_path, ignore_errors=True)
    
    return result



def test_code_get_feedback(code, test_cases, temp_dir="mas_workspace_1/mas_workspace_2/mas_workspace_3", timeout=20):
    """
    Test the given code against a list of test cases in a specified directory with a time limit and provide feedback.

    Args:
        code (str): The Python code to be tested, typically a function definition.
        test_cases (list of str): A list of test cases, where each test case is an assert statement represented as a string.
        temp_dir (str): The directory in which the code will be executed.
        timeout (int): Maximum time (in seconds) allowed for testing all test cases.

    Returns:
        tuple: A tuple containing:
            - int: The number of test cases that passed.
            - str: Feedback detailing errors or a success message.
    """
    if not code:
        return 0, "Empty code! This might be due to the code not being provided in the correct format (wrapped with triple backticks ```), causing extraction to fail."

    if not test_cases:
        return 0, "No test case provided!"

    # Ensure the temp directory exists
    original_dir = os.getcwd()
    temp_dir_path = os.path.join(original_dir, temp_dir)
    os.makedirs(temp_dir_path, exist_ok=True)

    def execute_tests(queue):
        """
        Worker function to execute the code and test cases.
        Sends the result back via a multiprocessing.Queue.
        """
        correct_count = 0
        feedback = ""
        shared_context = {}  # Shared context for exec() calls

        try:
            # Change to the temp directory
            os.chdir(temp_dir_path)

            # Execute the provided code to define the function or variables
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                exec(code, shared_context)

            # print(shared_context)


            for assert_str in test_cases:
                try:
                    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                        exec(assert_str, shared_context)  # Use the shared context for test cases
                    correct_count += 1
                except AssertionError:
                    feedback += f"Assertion failed: {assert_str}\n\n"
                except Exception as e:
                    feedback += f"Execution error in: {assert_str} -> {e}\n\n"

            # If all test cases pass
            if correct_count == len(test_cases):
                feedback = "All assertions passed successfully."

        except Exception as e:
            queue.put((0, f"Function definition error: {e}"))
            return

        finally:
            # Restore the original directory after all assertions
            os.chdir(original_dir)

        # Send results back to the main process
        queue.put((correct_count, feedback))

    # Create a multiprocessing.Queue for inter-process communication
    queue = multiprocessing.Queue()

    # Create a subprocess to run the test cases
    process = multiprocessing.Process(target=execute_tests, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # If the process is still running after the timeout, terminate it
        process.terminate()
        process.join()
        return 0, "Execution Time Out"

    # Retrieve results from the queue
    try:
        result = queue.get_nowait()
    except multiprocessing.queues.Empty:
        result = (0, "No feedback available.")

    # Clean up the temp directory content
    shutil.rmtree(temp_dir_path, ignore_errors=True)
    
    return result

def eval_mbpp(item, model_url_list, model_name, source_data):
    code = extract_code_function_llm(item['query'], item["generated_output"], model_url_list, model_name)
    item['extracted_code'] = code
    test_cases = item["test_cases"]
    if not code:
        print("No valid code extracted")
        return "No valid code extracted", 0

    correct_cnt, feedback = test_code_get_feedback(code, test_cases)

    # print("="*50, "\n>>Query:\n", item["query"])
    # print("="*50, "\n>>Code:\n", code)
    # print(">>Test: ", test_cases)
    # print(">> Feedback: ", feedback)
    # print(">> Pass Rate: ", correct_cnt/len(test_cases)*100)
            
    return feedback, correct_cnt/len(test_cases)

def eval_mbppplus(item, model_url_list, model_name, source_data):
    code = extract_code_function_llm(item['query'], item["generated_output"], model_url_list, model_name)
    item['extracted_code'] = code
    test_code = item["test_code"]
    function_signature = item["function_signature"]
    if not code:
        print("No valid code extracted")
        return "No valid code extracted", 0
    test_code += f"""{code}"""
    test_code += f"""
output = 0
for i, (inp, exp) in enumerate(zip(inputs, results)):
    try:
        assertion({function_signature}(*inp), exp, 0)
        output += 1
    except AssertionError:
        pass
output /= len(inputs)
"""

    feedback, success_rate = execute_code(test_code, timeout=120)
    success_rate = 0 if success_rate is None else success_rate
    if 'Error' in feedback or 'error' in feedback:
        print(feedback)

    return feedback, success_rate

def eval_fullstackbench(item, model_url_list, model_name, source_data):
    code = extract_code_function_llm(item['query'], item["generated_output"], model_url_list, model_name)
    item['extracted_code'] = code
    test_cases = item["test_cases"]
    if not code:
        print("No valid code extracted")
        return "No valid code extracted", 0
    code += "\nmy_solution = Solution()\n"
    correct_cnt, feedback = test_code_get_feedback(code, test_cases)
    if 'Error' in feedback:
        print(feedback)
    return feedback, correct_cnt/len(test_cases)

def eval_livecode_functional(item, model_url_list, model_name, source_data):
    code = extract_code_function_llm(item['query'], item["generated_output"], model_url_list, model_name)
    item['extracted_code'] = code
    code = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n\n" + code
    
    test_cases = item["private_test_cases"]
    func_name = item["func_name"]
    asserts = []
    for test_case in test_cases:
        asserts.append(f"solution = Solution()\nassert solution.{func_name}({test_case['input']}) == {test_case['output']}")
    if not code:
        print("No valid code extracted")
        return "No valid code extracted", 0
    correct_cnt, feedback = test_code_get_feedback(code, test_cases)
    if 'Error' in feedback:
        print(feedback)
    return feedback, correct_cnt/len(test_cases)

def eval_humaneval(item, model_url_list, model_name, source_data):
    code = extract_code_function_llm(item['query'], item["generated_output"], model_url_list, model_name)
    item['extracted_code'] = code
    test_code = "from typing import *\n"
    test_code += item['test']
    function_name = item['entry_point']

    if not code:
        print("No valid code extracted")
        return "No valid code extracted", 0
    test_code += f"""
{code}"""
    test_code += f"""
output = 0
try:
    check({function_name})
    output = 2
except Exception as e:
    pass
"""
    feedback, score = execute_code(test_code, timeout=120)
    if ('time' in feedback and 'out' in feedback) or ('Error' in feedback) or ('error' in feedback):
        score = 0
    
    if 'Error' in feedback:
        print(feedback)
    return feedback, score

def eval_humanevalplus(item, model_url_list, model_name, source_data):
    code = extract_code_function_llm(item['query'], item["generated_output"], model_url_list, model_name)
    item['extracted_code'] = code
    if not code:
        print("No valid code extracted")
        return "No valid code extracted", 0
    test_code = "from typing import *\n"
    test_code += item['test']
    test_code += f"""
{code}"""
    entry_point = item['entry_point']
    test_code += f"""
output = 0
try:
    check({entry_point})
    output = 2
except Exception as e:
    pass
"""
    feedback, score = execute_code(test_code, timeout=120)
    if ('time' in feedback and 'out' in feedback) or ('Error' in feedback) or ('error' in feedback):
        score = 0
    
    if 'Error' in feedback:
        print(feedback)
    return feedback, score

def eval_evoeval(item, model_url_list, model_name, source_data):
    code = extract_code_function_llm(item['query'], item["generated_output"], model_url_list, model_name)
    item['extracted_code'] = code
    test_code = item["test"]
    if not code:
        print("No valid code extracted")
        return "No valid code extracted", 0
    test_code = "from typing import *\n"
    test_code += item['test']
    test_code += f"""
{code}"""
    entry_point = item['entry_point']
    test_code += f"""
output = 0
try:
    check({entry_point})
    output = 2
except Exception as e:
    pass
"""
    print(test_code)
    feedback, score = execute_code(test_code, timeout=120)
    if ('time' in feedback and 'out' in feedback) or ('Error' in feedback) or ('error' in feedback):
        score = 0
    
    if 'Error' in feedback:
        print(feedback)
    return feedback, score

def eval_bigcodebench(item, model_url_list, model_name, source_data):
    code = extract_code_function_llm(item['query'], item["generated_output"], model_url_list, model_name)
    if not code:
        print("No valid code extracted")
        return "No valid code extracted", 0
    test = item["test"]
    test_code = f"""
{code}

{test}

def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCases)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    return result.wasSuccessful()
    
if run_tests():
    output = 2
else:
    output = 0
"""
    feedback, score = execute_code(test_code, timeout=120)
    return feedback, score

def eval_eval(item, model_url_list, model_name, source_data):

    extracted_eval = item.get('generated_output')
    extracted_checker = source_data.get('gt_score')
    extracted_gt = item.get('gt', "N/A")  
    
    if (
        (extracted_eval == "True" and extracted_checker == 2) or  
        (extracted_eval == "False" and extracted_checker in {1, 0})  
    ):
        score = 2  
    else:
        score = 0  
    
    response = f"evaluate: {extracted_eval} | checker: {extracted_checker} | Correct: {extracted_gt}"

    return response, score

def evaluate_one_sample(item, model_url_list, model_name, source_data):
    
    if "evaluate_query" in item:
        eval_func = eval_eval
    elif item['source'] == 'MBPP':
        eval_func = eval_mbpp
    elif item['source'] == "MBPP-Plus":
        eval_func = eval_mbppplus
    elif item['source'] == 'MMLU-Pro-1':
        eval_func = eval_mmlupro
    elif item['source'] == 'FullStackBench':
        eval_func = eval_fullstackbench
    elif item['source'] == 'LiveCodeBench':
        eval_func = eval_livecode_functional
    elif item['source'] == 'HumanEval':
        eval_func = eval_humaneval
    elif item['source'] == 'HumanEval-Plus':
        eval_func = eval_humanevalplus
    elif item['source'] in ["EvoEval_difficult", "EvoEval_creative", "EvoEval_subtle", "EvoEval_combine"]:
        eval_func = eval_evoeval
    else:
        eval_func = eval_llm_v2
    
    return eval_func(item, model_url_list, model_name, source_data)  # eval_content, score

def load_source_data(source_dir):
    """加载 JSON 文件并构建查询映射表"""
    source_map = {}
    
    with open(source_dir, "r") as f:
        source_datas = json.load(f)

    for source_data in source_datas:
        query = source_data.get('query')
        if query:
            source_map[query] = source_data
            
    print(f"Loaded {len(source_map)} source data items.")
    return source_map

def get_evaluation(eval_data, model_url_list, model_name, dataset_name, infer_name, max_workers=4, sequential=False):
    """
    批量评分函数，使用并行处理来加速评分过程。
    
    :param prompts: 要评分的字符串列表
    :param max_workers: 最大并行线程数
    :return: 返回评分列表
    """
    # print(eval_data)
    if "evaluate" in infer_name:
        source_dir = f"./X-MAS-Design/results/{dataset_name}/qwen2.5-32b-instruct_direct_eval.json"
        source_map = load_source_data(source_dir)
    else:
        source_map = {}
    
    eval_content_list, scores = [None] * len(eval_data), [None] * len(eval_data)

    if sequential:
        for i in tqdm(range(len(eval_content_list)), "Sequential Evaluating"):
            query = eval_data[i].get("query")
            source_data = source_map.get(query, {})
            eval_content_list[i], scores[i] = evaluate_one_sample(eval_data[i], model_url_list, model_name, source_data)
    else:
        # 使用ThreadPoolExecutor进行并行化
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 将所有评分任务提交给executor
            future_to_index = {executor.submit(evaluate_one_sample, item, model_url_list, model_name, source_map.get(item.get("query"), {})): idx for idx, item in enumerate(eval_data)}
            
            # 等待所有任务完成并收集结果
            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index), desc="Parallel Evaluating"):
                idx = future_to_index[future]
                try:
                    eval_content, score = future.result()
                    # print(eval_content, score)
                    eval_content_list[idx] = eval_content
                    scores[idx] = score
                except Exception as exc:
                    print(f"Error occurred for prompt at index {idx}: {exc}")
                    print(f)
                    eval_content_list[idx] = "Error"
                    scores[idx] = 0
    return eval_content_list, scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama-3.1-70b-instruct", help="the LLM for judgement")
    parser.add_argument("--model_config", type=str, default="config_.json")
    parser.add_argument("--dataset_names", type=str, nargs='+', default=["MATH", "GSM8K", "AQUA-RAT", "MedMCQA"])
    parser.add_argument("--eval_mode", type=str, choices=["test", "train", "bench-test"], required=True)
    parser.add_argument("--infer_name", type=str, default="mas_3_5cot-sc_general_infer.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    print("="*50)
    print(json.dumps(vars(args), indent=4))
    
    # ================== Define the model list ==================
    with open(args.model_config, "r") as f:
        config = json.load(f)
        model_dict = config["model_dict"]
        worker_dict = config["worker_dict"]
    model_list = model_dict[args.model_name]
    model_url_list = [item[1] for item in model_list]
    max_workers = worker_dict[args.model_name] * len(model_list)
    print(f">> {len(model_url_list)} models will be used for evaluation")

    # ============== main ==============
    for i, dataset_name in enumerate(args.dataset_names):
        if dataset_name in ["IFEval"]:
            continue

        print('-'*20 + f"\n>> Evaluating {i}-th dataset: {dataset_name}")
        if args.eval_mode == "bench-test":
            infer_path = f"./X-MAS-Design/results/{dataset_name}/{args.infer_name}"


        save_eval_path = infer_path.replace(".jsonl", "_eval.json")

        eval_data, existing_eval_data = [], []

        try:
            print(infer_path)
            eval_data = []
            with open(infer_path, "r") as f:
                tmp = f.readlines()
            # eval_data = [json.loads(line) for line in tmp]

            for line in tmp:
                try:
                    eval_data.append(json.loads(line))
                except Exception as e:
                    print(line)
                    print(f"{e}")

            print(f">> Before filtering: {len(eval_data)} samples")

            if os.path.exists(save_eval_path):
                with open(save_eval_path, "r") as f:
                    existing_eval_data = json.load(f)

                # 获取已评估过的样本的 query 和 mas_name 的组合，mas_name 不存在时只用 query
                evaluated_pairs = {
                    (item['query'], item['mas_name']) if 'mas_name' in item else item['query']
                    for item in existing_eval_data if 'gt_score' in item
                }

                # 筛选出那些没有被评估过的样本
                eval_data = [
                    item for item in eval_data
                    if ('mas_name' in item and (item['query'], item['mas_name']) not in evaluated_pairs) or
                    ('mas_name' not in item and item['query'] not in evaluated_pairs)
                ]

            print(f">> After filtering: {len(eval_data)} samples")

            eval_data = eval_data[1:3] if args.dry_run else eval_data

            print(f">> Running Loaded {len(eval_data)} samples")

            eval_content_list, score_list = get_evaluation(eval_data, model_url_list, args.model_name, dataset_name, args.infer_name, max_workers, args.sequential)

            # mapping the response back to the original query
            for i, eval_content, score in zip(range(len(eval_data)), eval_content_list, score_list):
                
                # 将评分加入到responses中
                eval_data[i]['eval_content'] = eval_content
                eval_data[i]['gt_score'] = score
                existing_eval_data.append(eval_data[i])

            print(f">> Finished evaluating {len(eval_data)} samples")

            with open(save_eval_path, "w") as f:
                json.dump(existing_eval_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error occurred during evaluation: {e}")
            continue
