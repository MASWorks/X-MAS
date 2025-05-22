from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system designed to address specialized problems.
        Steps:
            1.An agent provides the initial ideas to solve the problem.
            2.An agent generates python code to solve the problem based on the initial problem-solving idea.
            3.A testing agent devises test cases to evaluate the outcome of the problem.
            4.An improvement agent proposes new ideas based on the test results.
            5.A reflective agent offers a new solution based on the revised idea and test outcomes.
        """
        initial_idea_instruction = f"""**Task:
{taskInfo}

Please think step by step and then provide the ideas to solve the problem.
"""
        initial_idea = self.llm.call_llm(initial_idea_instruction)


        # Initial instruction for the idea agent to think step by step and provide the first solution idea.
        initial_code_instruction = f"""**Task:**
{taskInfo}

**Ideas to solve the problem:**
{initial_idea}

Please think step by step and then solve the task by writing python code. Your code should only contain the function definition and not the test cases.

Wrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your function code here
</Code Solution>
""" # ask for specific output format for easier extraction
        
        
        # Call `generate_and_extract_code` to generate answer using the initial instruction and extract the code
        _, code = generate_and_extract_code(llm=self.llm, prompt=initial_code_instruction)
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
        _, test_code = generate_and_extract_code(llm=self.llm, prompt=test_generation_instruction)
        output = execute_code(code+test_code)

        refine_idea_instruction = f"""**Task:**
{taskInfo}

**Code Solution:**
{code}

**Test Code:**
{test_code}

**Test Result:**
{output}

The **Code Solution** is the initial attempt to solve the **Task**. The **Test Code** is the code used to test the solution. The **Test Result** is the output of the test code execution.
You need to carefully consider the test results. Modify and refine the ideas to solve the task.
"""
        refine_idea_response = self.llm.call_llm(refine_idea_instruction)

        final_output_instruction = f"""**Task:**
{taskInfo}

**Code Solution:**
{code}

**Test Code:**
{test_code}

**Test Result:**
{output}

**New Ideas:**
{refine_idea_response}

The **Code Solution** is the initial attempt to solve the **Task**. The **Test Code** is the code used to test the solution. The **Test Result** is the output of the test code execution. The **New Ideas:** is new attempts to solve the problem.
You need to carefully consider the test results. Use the new ideas to provide the correct answer to the given task.
"""
        final_output_response = self.llm.call_llm(final_output_instruction, temperature=0.3)


        return final_output_response
