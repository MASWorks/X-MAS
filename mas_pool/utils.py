import json
import re
import ast
import os
import openai
import multiprocessing
import shutil
import random
import io                
import contextlib
import sys
import requests
import tiktoken
from datetime import datetime

try:
    sys.path.append("/mnt/petrelfs/yerui/mac/Apptainer")
    from apptainer_exe import ApptainerClient
except ImportError:
    print("Failed to import ApptainerClient. Skipping this part.")
except Exception as e:
    print(f"An unexpected error occurred when import ApptainerClient: {e}")

default_model_list = ["biomedgpt-lm-7b", "huatuogpt-o1-7b"]

def proxy_on():
    proxy_addr = "http://yerui:L5OUlLMt9QmwNSMmSUaY80oZYgVZ9BmIg88suLAi7Ql47PaLmxJf0hcpWKeB@10.1.20.50:23128/"
    os.environ['HTTP_PROXY'] = proxy_addr
    os.environ['HTTPS_PROXY'] = proxy_addr
    os.environ['http_proxy'] = proxy_addr
    os.environ['https_proxy'] = proxy_addr      

def proxy_off():
    os.environ['HTTP_PROXY'] = ""
    os.environ['HTTPS_PROXY'] = ""
    os.environ['http_proxy'] = ""
    os.environ['https_proxy'] = ""

def parse_to_json(input_str):
    """
    Attempts to parse the input string into a JSON object.
    If direct parsing fails, extracts the first '{}' block and tries parsing it as JSON.
    
    Args:
        input_str (str): The input string to be parsed.
        
    Returns:
        dict: Parsed JSON object if successful.
        None: None if parsing fails.
    """
    try:
        # Attempt direct parsing
        return json.loads(input_str)
    except json.JSONDecodeError:
        # If direct parsing fails, search for the first '{}' block
        match = re.search(r'\{.*?\}', input_str, re.DOTALL)
        if match:
            json_fragment = match.group(0)
            try:
                # Attempt to parse the extracted block
                return json.loads(json_fragment)
            except json.JSONDecodeError:
                # Return none if the extracted block cannot be parsed
                return None
        else:
            # Return none if no '{}' block is found
            return None
        
def extract_code(text):
    """
    Extract code enclosed by triple backticks (```).

    Args:
        text (str): The input text containing code enclosed by triple backticks.

    Returns:
        str: Extracted code without language descriptors. An empty string if no matches found.
    """
    # Match content enclosed by triple backticks
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Extract the first match and strip surrounding whitespace
        match = matches[0].strip()
        # Split by lines
        lines = match.split("\n")
        # Check if the first line is a language descriptor (e.g., 'python', 'cpp', etc.)
        if len(lines) > 1 and lines[0].strip().lower() in {
            "python", "cpp", "java", "javascript", "c", "c++", "bash", "html", "css", "json", "sql"
        }:
            # Remove the first line if it is a language descriptor
            lines = lines[1:]
        code =  "\n".join(lines).strip()  # Join the remaining lines

        try:
            # Parse the code to check if it's valid Python syntax
            ast.parse(code)
            return code  # Code is valid and executable
        except (SyntaxError, ValueError):
            return ""  # Code is invalid or not executable
    
    return ""  # Return empty string if no matches found

from tenacity import retry, wait_exponential, stop_after_attempt, RetryError

def handle_retry_error(retry_state):
    return None


def save_obj_to_jsonl(query:str, response_text:str, prompt:str, save_path:str):
    try:
        
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(response_text)
        token_cnt = len(tokens)
        save_obj = {
            'query':query,
            'token':token_cnt,
            'prompt':prompt,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        save_obj = {
            'query':query,
            'token':-1,
            'prompt':prompt,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    if save_path:
        with open(save_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(save_obj) + '\n')



import logging
class LLM():

    def __init__(self, model_list):
        self.model_list = model_list
        ## 用于记录 call 模型次数和token数
        self.save_path = None
        self.query = None

        if len(self.model_list[0]) == 2:
            self.model_name, self.model_url = random.choice(self.model_list)
            self.api_key = "EMPTY"
        elif len(self.model_list[0]) == 3:
            self.model_name, self.model_url, self.api_key = random.choice(self.model_list)
        else:
            raise ValueError("Invalid model list format.")
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
    def call_llm(self, prompt, temperature=0.5):
        
        retries = 0
        while retries < 3:
            try:

                if self.model_name == "deepseek-reasoner":
                    proxy_on()
                    llm = openai.OpenAI(base_url="https://api.deepseek.com", api_key="sk-35134a687cf6409183a3bc4f5350722c")
                    completion = llm.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role":"user", "content":prompt}
                        ],
                        max_tokens=8192,
                        stream=False
                    )
                    proxy_off()

                    response = completion.choices[0].message.content
                    if "r1" in self.model_name:
                        try:
                            response = response.split("</think>")[-1]
                        except:
                            pass
                    ############### 添加 写入文件的逻辑 ######################

                    response_text = completion.choices[0].message.content
                    
                    save_obj_to_jsonl(self.query, response_text, prompt, self.save_path)
                    return response

                if ("gpt" not in self.model_name and "o1" not in self.model_name) or self.model_name in default_model_list:
                    llm = openai.OpenAI(base_url=f"{self.model_url}", api_key=self.api_key)
                    if 'qwq' in self.model_name or 'r1' in self.model_name:
                        max_tokens = 8192
                        timeout = 600
                    else:
                        max_tokens = 2048
                        timeout = 180
                    try:
                        completion = llm.chat.completions.create(
                            model=f"{self.model_name}",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            stop=['<|eot_id|>'],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=timeout
                        )
                        # response = completion.choices[0].message.content
                        response = completion
                    except Exception as e:
                        response = f"Error occurred: {str(e)}"
                        print(response)
                        return response
                    
                    ############### 添加 写入文件的逻辑 ######################
                    response_text = completion.choices[0].message.content
                    save_obj_to_jsonl(self.query, response_text, prompt, self.save_path)
                    return response
                else:
                    proxy_on()
                    payload_dict = {
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    }
                    if "o1" not in self.model_name:
                        payload_dict["temperature"] = temperature
                    if "o1" not in self.model_name:
                        payload_dict["max_completion_tokens"] = 4096
                    # else:
                    #     payload_dict["max_completion_tokens"] = 16384
                    payload = json.dumps(payload_dict)
                    headers = {
                        'Authorization': 'sk-7cEFlTBXX4sPi8Xx6a32DdA4B5A74b85Aa2893896f2c01Bc',
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json',
                        'Accept': '*/*',
                        'Host': '47.88.65.188:8405',
                        'Connection': 'keep-alive'
                        }
                    result = requests.request("POST", "http://47.88.65.188:8405/v1/chat/completions", headers=headers, data=payload)
                    print(result.json())
                    # if "o1" in self.model_name:
                    print(result)
                    response = result.json()["choices"][0]["message"]["content"]

                    proxy_off()
                    ############### 添加 写入文件的逻辑 ######################
                    response_text = result.json()["choices"][0]["message"]["content"]
                    save_obj_to_jsonl(self.query, response_text, prompt, self.save_path)

                    return response
            except Exception as e:
                retries += 1
                logging.error(f"{retries}-th request failed with error: {e}")
                if retries == 3:
                    logging.error(f"After 3 retries, request failed with error: {e}")
                    return None

    def multi_turn_conversation(self, prompt, messages, temperature=0.5):
        
        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        retries = 0
        while retries < 3:
            try:
                if self.model_name == "deepseek-reasoner":
                    proxy_on()
                    llm = openai.OpenAI(base_url="https://api.deepseek.com", api_key="sk-35134a687cf6409183a3bc4f5350722c")
                    response = llm.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=messages,
                        max_tokens=8192,
                        stream=False
                    )

                    messages.append(
                        {
                            "role": "assistant",
                            "content": completion.choices[0].message.content
                        }
                    )
                    proxy_off()
                    ############### 添加 写入文件的逻辑 ######################
                    response_text = response.choices[0].message.content, messages
                    save_obj_to_jsonl(self.query, response_text, prompt, self.save_path)
                    ########################################################

                    return response.choices[0].message.content, messages

                elif "gpt" not in self.model_name and "o1" not in self.model_name:
                    
                    llm = openai.OpenAI(base_url=f"{self.model_url}", api_key=self.api_key)
                    completion = llm.chat.completions.create(
                        model=f"{self.model_name}",
                        messages=messages,
                        stop=['<|eot_id|>'],
                        temperature=temperature,
                        max_tokens=2048,
                        timeout=60
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": completion.choices[0].message.content
                        }
                    )

                    ############### 添加 写入文件的逻辑 ######################
                    response_text = completion.choices[0].message.content, messages
                    save_obj_to_jsonl(self.query, response_text, prompt, self.save_path)
                    ########################################################
                    
                    return completion.choices[0].message.content, messages
                else:
                    proxy_on()
                    payload_dict = {
                        "model": self.model_name,
                        "messages": messages
                    }
                    if "o1" not in self.model_name:
                        payload_dict["temperature"] = temperature
                    if "o1" not in self.model_name:
                        payload_dict["max_completion_tokens"] = 4096
                    payload = json.dumps(payload_dict)
                    headers = {
                        'Authorization': 'sk-hBgybDTlbHcb3du371F28825975e46D294D75bC26b5dBd2c',
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json',
                        'Accept': '*/*',
                        'Host': '47.88.65.188:8405',
                        'Connection': 'keep-alive'
                        }
                    result = requests.request("POST", "http://47.88.65.188:8405/v1/chat/completions", headers=headers, data=payload)
                    response = result.json()["choices"][0]["message"]["content"]
                    
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response
                        }
                    )
                    proxy_off()

                    ############### 添加 写入文件的逻辑 ######################
                    response_text = result.json()["choices"][0]["message"]["content"]
                    save_obj_to_jsonl(self.query, response_text, prompt, self.save_path)
                    ########################################################

                    return response, messages
            except Exception as e:
                retries += 1
                if retries == 3:
                    logging.error(f"After 3 retries, request failed with error: {e}")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "The LLM failed to give a response."
                        }
                    )
                    return None, messages
    
'''
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
'''

def execute_code(code, temp_dir="mas_workspace_1/mas_workspace_2/mas_workspace_3", timeout=30):
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

    container = ApptainerClient("../Apptainer/sandbox")
    # container = ApptainerClient("/home/ubuntu/DATA3/ruige/python-sandbox.sif")

    captured_output, output = container.exec_code(code, timeout=timeout)

    
    result = f"Final output:{output}\nPrint during execution:{captured_output}\n".strip()
    return result


def test_code_get_feedback(code, test_cases, temp_dir="mas_workspace_1/mas_workspace_2/mas_workspace_3", timeout=30):

    container = ApptainerClient("../Apptainer/sandbox")
    # container = ApptainerClient("/home/ubuntu/DATA3/ruige/python-sandbox.sif")

    result = container.test_code_with_testcases(code, test_cases, timeout)

    return result   # tuple (passed test cases, feedback)


# def test_code_get_feedback(code, test_cases, temp_dir="mas_workspace_1/mas_workspace_2/mas_workspace_3", timeout=20):
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


def websearch(query):
    """
    Search the internet given the query and return a list of passages.
    
    Args:
        query (str): a query or keyword for web search.
    Return:
        list: a list of searched passages(str)

    """
    return []

FUNCTION_SIGNATURE_DESIGNER_PROMPT = """Problem Description: {}

Task:
Given the problem description, write a Python function signature that matches the problem's requirements, \
including appropriate argument types. The function signature must include a brief and clear docstring that describes the function's purpose, its parameters, \
and the return value.

Your output must be formatted as a JSON object with two fields:
1. "think": Describe your reasoning and approach to solving the problem.
2. "function": Provide the function signature, including the docstring.

Use the following example as a guide for formatting:
{{
  "think": "Your reasoning process here.",
  "function": "def calculate_sum(a: int, b: int) -> int:\\n    \\\"\\\"\\\"\\n    Calculate the sum of two integers.\\n\\n    Parameters:\\n    a (int): The first integer.\\n    b (int): The second integer.\\n\\n    Returns:\\n    int: The sum of the two integers.\\n    \\\"\\\"\\\""
}}

Ensure the function signature and docstring are concise and directly aligned with the problem statement. You should output only the function signature so avoid including the function implementation. Avoid adding any text or explanations outside of the "think" field.

Please adhere strictly to the JSON format. Provide only the JSON object as the output.
"""

TEST_DESIGNER_PROMPT = """Problem Description: {problem}
Function Signature:
{function}

Task:
As a tester, your task is to create comprehensive test cases given the problem description and the function signature. \
These test cases should encompass Basic, Edge, and Large Scale scenarios to ensure the code's robustness, reliability, and scalability, in the format \
of assert statements. Remember to import necessary libs in each assert assert statements if necessary.

Your output must be formatted as a JSON object with four fields:
1. "think": Describe your reasoning and approach to solving the problem.
2. "basic": Several basic test cases to verify the fundamental functionality of the function under normal conditions.
3. "edge": Several edge test cases to evaluate the function's behavior under extreme or unusual conditions.
4. "large scale": Several large-scale test cases to assess the function's performance and scalability with large data samples.

**Instructions**:
- Implement a comprehensive set of test cases following the guidelines above.
- Ensure each test case is well-documented with comments explaining the scenario it covers.
- Pay special attention to edge cases as they often reveal hidden bugs.
- For large-scale tests, focus on the function's efficiency and performance under heavy loads.

Use the following example as a guide for formatting:
{{
  "think": "Describe your reasoning and approach here.",
  "basic": [
    "# An ordinary case\\nassert sum([3,5]) == 8",
    "# An ordinary case\\nassert sum([2,7,3]) == 12",
    ...
  ],
  "edge": [
    "# Test with empty input list\\nassert sum([]) == 0",
    "# Test with single-element input\\nassert sum([7]) == 7",
    ...
  ],
  "large scale": [
    "# Test with large input list\\nlarge_list = [i for i in range(100)]\\nassert sum(large_list) == 4950",
    ...
  ]
}}

Please adhere strictly to the JSON format. Use '\\n' to represent line breaks in multi-line strings. Provide only the JSON object as the output. Do not add any text or explanations outside the JSON object. All comments must be included inside the JSON object as part of the strings. Do not place comments outside of the JSON structure to ensure proper parsing.
"""


def get_function_signature(llm, taskInfo):
    """
    Generate a Python function signature based on the problem description.

    Args:
        taskInfo (str): The problem description.

    Returns:
        str: The function signature with an appropriate docstring.
    """
    # Generates an instruction prompt by formatting the FUNCTION_SIGNATURE_DESIGNER_PROMPT with the task information
    function_signature_designer_instruction = FUNCTION_SIGNATURE_DESIGNER_PROMPT.format(taskInfo)
    # Calls the large language model (LLM) with the generated instruction
    answer = llm.call_llm(function_signature_designer_instruction)
    # Parses the LLM's response into a dictionary
    answer_dict = parse_to_json(answer)
    # Extracts and returns the function signature from the response
    if answer_dict and "function" in answer_dict.keys():
        return answer_dict["function"]
    return ""

# Function to generate test cases from the problem description and function signature
def get_test_cases(llm, taskInfo, function_signature):
    """
    Generate test cases based on the problem description and function signature.

    Args:
        taskInfo (str): The problem description.
        function_signature (str): The Python function signature.

    Returns:
        list: A list of test cases combining basic, edge, and large-scale scenarios.
    """
    # Generates an instruction prompt by formatting the TEST_DESIGNER_PROMPT with the task information and function signature
    test_designer_instruction = TEST_DESIGNER_PROMPT.format(problem=taskInfo, function=function_signature)
    # Calls the LLM with the generated instruction
    answer = llm.call_llm(test_designer_instruction, temperature=0.3)
    # Parses the LLM's response into a dictionary
    answer_dict = parse_to_json(answer)
    # Combines and returns the basic, edge, and large-scale test cases from the response
    if answer_dict and "basic" in answer_dict.keys() and "edge" in answer_dict.keys() and "large scale" in answer_dict.keys():
        return answer_dict["basic"] + answer_dict["edge"] + answer_dict["large scale"]
    # return an empty list if parse fails
    return []         

def extract_code_solution(solution):
    """
    Extract the code solution from the provided solution string.

    Args:
        solution (str): The solution string containing the code snippet.

    Returns:
        str: The extracted code snippet.
    """
    # Extract the code snippet enclosed by custom tags
    code_pattern = r"<Code Solution>\s*(.*?)\s*</Code Solution>"
    match = re.search(code_pattern, solution, re.DOTALL)
    if match:
        code = match.group(1)
        # Remove code block tags if present
        code = re.sub(r"^```(?:\w+)?\n?|```$", "", code, flags=re.MULTILINE).strip()
        if code:
            return code
        return ""
    return ""

def generate_and_extract_code(llm, prompt, temperature=None, max_attempts=3):
        """
        Generate a response from the LLM and extract the contained code with retry logic.

        This function attempts to generate a response from the LLM containing a code snippet.
        It first extracts the portion of the response wrapped within custom tags (e.g., <Code Solution>). 
        Then remove possible code block tags (e.g., ```python).
        Returns both the full response and the extracted code. 
        If no valid code is found after multiple attempts, it returns the last response and an empty string for the code.

        Args:
            prompt (str): The instruction to send to the LLM to generate a response with code.
            temperature (float, optional): Sampling temperature for the LLM, controlling randomness in the output.
            max_attempts (int): Maximum number of attempts to fetch a response with valid code. Default is 3.
            
        Returns:
            tuple:
                str: The full LLM response.
                str: The extracted code snippet, or an empty string if no valid code is detected.
        """
        attempts = 0  # Track the number of attempts
        tag_pattern = r"<Code Solution>\s*(.*?)\s*</Code Solution>" # Regular expression pattern to extract content within custom tags
        if "<Code Solution>" not in prompt:
            prompt += """\n\nWrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your function code here
</Code Solution>"""
        
        while attempts < max_attempts:
            # Generate response using the LLM
            if temperature:
                llm_response = llm.call_llm(prompt, temperature=temperature)
            else:
                llm_response = llm.call_llm(prompt)
                
            code = extract_code_solution(llm_response)
            if code:
                return llm_response, code
            
            attempts += 1  # Increment attempts and retry if no valid code is detected
        
        # Return the last LLM response and an empty code snippet after exhausting all attempts
        return llm_response, ""