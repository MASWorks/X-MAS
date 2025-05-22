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
            3. Iterate over the reflection process for a maximum number of attempts to refine the code.
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
        
        # Instruction for the reasoning agent to reflect on previous attempts and feedback to improve the code in subsequent iterations.
        cot_reflect_instruction = (
            f"Task:\n{taskInfo}\n\n"
            f"Function Signature:\n{function_signature}\n\n"
            "Given previous attempts and feedback, carefully consider where you went wrong in your latest attempt. Using insights from previous attempts, try to solve the task better. \n"
            "Ensure the function adheres to the provided signature.\n"
            "Wrap your final code solution in <Code Solution> and </Code Solution>. For example:\n"
            "<Code Solution>\n"
            "Your function code here\n"
            "</Code Solution>\n"
        )
        
        N_max = 3  # Maximum number of attempts to refine the code

        # Iterate through the feedback loop for up to N_max attempts, improving the code with each iteration.
        for _ in range(N_max):
            # Call `test_code_get_feedback` function to test the code on test cases and get feedback 
            correct_count, feedback = test_code_get_feedback(code=code, test_cases=test_cases)

            # Break out if all test cases pass
            if correct_count == len(test_cases):
                break
            
            # Update the reflection instruction with the answer and feedback from the previous attempt.
            prompt = cot_reflect_instruction + f"Attempt:\n{answer}\nFeedBack:\n{feedback}\n\n"
            
            # Call `generate_and_extract_code` with reflection instruction and update the answer and code
            answer, code = generate_and_extract_code(llm=self.llm, prompt=prompt)

        # return the answer in the last round
        return answer