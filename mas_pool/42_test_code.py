from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi agent system for solving coding tasks.
        Steps:
            1. An agent gives an initial code attempt.
            2. A test agent writes test code to test the solution.
            3. A reasoning agent reflects on the code based on the feedback from tests and provides an improved code solution.
        """
        # Initial instruction for the reasoning agent to think step by step and provide the first code attempt.
        cot_initial_instruction = f"""**Task:**
{taskInfo}

Please think step by step and then solve the task by writing python code. Your code should only contain the function definition and not the test cases.

Wrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your function code here
</Code Solution>
""" # ask for specific output format for easier extraction
        
        
        # Call `generate_and_extract_code` to generate answer using the initial instruction and extract the code
        _, code = generate_and_extract_code(llm=self.llm, prompt=cot_initial_instruction)
        test_generation_instruction = f"""**Task:**
{taskInfo}

**Code Solution:**
{code}

The **Code Solution** is an attempt to solve the **Task**. Now, you need to write test code to test the solution. Follow the guidelines below:
1. Your test code will be directly appended to the Code Solution and executed afterward. It should be executable in sequence without additional setup.
2. It should include a variety of test cases, such as normal cases, edge cases, and cases with large inputs.
3. Use either assert statements or print to output the result of each test case. However, use try-catch blocks to ensure that the failure of one test case does not cause the entire testing process to stop.
4. Print test messages and results for each test case to provide clear feedback on the test results.
5. You need to calculate the accuracy of the tests, storing the result in the `output` variable. This variable must be defined at the global scope and contain the final computation result.

Wrap your final test code in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your test code here
</Code Solution>
"""     
        # Generate test code
        _, test_code = generate_and_extract_code(llm=self.llm, prompt=test_generation_instruction)
        # Execute the code and test code, and get the test result
        output = execute_code(code+'\n'+test_code)

        # Refine the initial code solution based on the test results
        refine_instruction = f"""**Task:**
{taskInfo}

**Code Solution:**
{code}

**Test Code:**
{test_code}

**Test Result:**
{output}

The **Code Solution** is the initial attempt to solve the **Task**. The **Test Code** is the code used to test the solution. The **Test Result** is the output of the test code execution.
You need to carefully consider the test results. Modify and refine the code solution to improve the accuracy of the solution. Then output your final solution to the **Task**. Your final response should be complete as if you are solving the task.

Wrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your function code here
</Code Solution>
"""
        refine_response = self.llm.call_llm(refine_instruction)
        # return the reflected response
        return refine_response