from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi agent system for solving coding tasks.
        Steps:
            1. An agent gives an initial code attempt.
            2. A reasoning agent reflects on the code based on the feedback from tests and provides an improved code solution.
        """
        # Call `get_function_signature` to generate the function signature based on the task information
        function_signature = get_function_signature(llm=self.llm, taskInfo=taskInfo)
        
        # Call `get_test_cases` to generate a list of test cases for the function
        test_cases = get_test_cases(llm=self.llm, taskInfo=taskInfo, function_signature=function_signature)

        # Initial instruction for the reasoning agent to think step by step and provide the first code attempt.
        cot_initial_instruction = (
            f"Task:\n{taskInfo}.\n\n"
            f"Function Signature:\n{function_signature}\n\n"
            "Please think step by step and then solve the task by writing python code. Ensure the function adheres to the provided signature.\n"
            "Wrap your final code solution in <Code Solution> and </Code Solution>. For example:\n"
            "<Code Solution>\n"
            "Your function code here\n"
            "</Code Solution>\n"
        ) # ask for specific function signature for test and specific output format for easier extraction 
        
        # Call `generate_and_extract_code` to generate answer using the initial instruction and extract the code
        answer, code = generate_and_extract_code(llm=self.llm, prompt=cot_initial_instruction)
        
        # Call `test_code_get_feedback` function to test the code on test cases and get feedback 
        correct_count, feedback = test_code_get_feedback(code=code, test_cases=test_cases)

        # Return the answer if all test cases pass
        if correct_count == len(test_cases):
            return answer
        
        # Instruction for the reasoning agent to reflect on previous attempts and feedback to improve the code in subsequent iterations.
        reflection_instruction = (
            f"**Task:**\n{taskInfo}\n\n"
            f"**Function Signature:**\n{function_signature}\n\n"
            f"**Previous Attempt:**\n{answer}\n\n"
            f"**Feedback:**\n{feedback}\n\n"
            "**Previous Attempt** provides an initial solution to the task. **Feedback** is the result of testing the code on multiple test cases.\n"
            "Your need to carefully consider what went wrong in the previous attempt. Modify and refine the previous attempts and try to solve the task better.\n"
            "Your final response should be complete as if you are directly answering the problem. Ensure the function adheres to the provided signature.\n\n"
            "Wrap your final code solution in <Code Solution> and </Code Solution>. For example:\n"
            "<Code Solution>\n"
            "Your function code here\n"
            "</Code Solution>\n"
        )
        
        reflection_response = self.llm.call_llm(reflection_instruction)

        # return the reflected response
        return reflection_response