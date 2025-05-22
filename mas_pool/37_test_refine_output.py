from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    # Main forward function to generate solutions, refine them, and return the best one
    def forward(self, taskInfo):
        """
        Design a multi-agent system for code generation.
        
        Steps:
            1. The code agent generates a code solution based on the task description.
            2. The test agent generates test code to validate the correctness of the implementation.
            3. The code agent refines the code based on the test result.
            4. Iterate between the test agent and the code agent to refine the code and test it.
        """
        # Initialize the history messages for code agent and test agent for multi-turn conversation
        code_agent_history_messages = []
        test_agent_history_messages = []

        # Instruction for the code agent to write the code
        implementation_instruction = f"""Problem Description: {taskInfo}

Task:
Based on the provided problem, write Python code that solves the problem.

In your output, you should first describe your reasoning and approach to solving the problem. Then provide the code.

Wrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your code here
</Code Solution>
"""
        # Call the llm to generate a code solution and update the history messages
        code_response, code_agent_history_messages = self.llm.multi_turn_conversation(prompt=implementation_instruction, messages=code_agent_history_messages)
        # Extract the code from the response
        code = extract_code_solution(code_response)
        
        # Refine the code through testing, maximum 3 iterations
        for _ in range(3):
            # Instruction for the test agent to write the test code
            test_instruction = f"""Problem Description: {taskInfo}

Solution:
{code}

Task:
Based on the problem description and the provided solution, write test code to validate the correctness of the implementation.

Requirements:
- Include the code implementation from the solution directly within your test code, placing it in the appropriate location.
- Generate and include specific test cases along with their expected results. Ensure that these test cases comprehensively validate the solution. Use a mechanism similar to `try-catch` to handle each test case separately, ensuring that the testing process continues even if one or more test cases fail.
- Provide complete and self-contained code with necessary imports and setups. Ensure that all variables in the code are defined and the entire script can be directly executed without errors.
- Include clear print statements to display the results of the tests and compare them against the expected results in a way that is easy to understand.
- Wrap your final test code in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your code here
</Code Solution>
"""
            # Call the llm to generate test code and update the history messages
            test_response, test_agent_history_messages = self.llm.multi_turn_conversation(prompt=test_instruction, messages=test_agent_history_messages)
            # Extract the code from the response
            test_code = extract_code_solution(test_response)
            # Execute the test code to get the test result
            test_result = execute_code(test_code)
            
            # The instruction for the code agent to refine the code based on the test result
            refine_instruction = f"""Problem Description: {taskInfo}

Test code:
{test_code}

Test result:
{test_result}

Task:
Based on the test result, refine your previous code solution to improve its performance. Ensure that the output format strictly adheres to the requirements in the problem description and is not altered by the changes made for testing purposes. Adjustments for testing will be addressed separately.

Wrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your code here
</Code Solution>
"""
            # Call the llm to refine the code and update the history messages
            code_response, code_agent_history_messages = self.llm.multi_turn_conversation(prompt=refine_instruction, messages=code_agent_history_messages)
            # Extract the code for next iteration
            code = extract_code_solution(code_response)

        # The final instruction for the code agent to submit the final refined code solution
        final_instruction = f"""Problem Description: {taskInfo}

Based on previous attempts and test results, provide the final code solution that addresses the problem requirements.
"""
        # Call the llm to get the final refined code solution
        final_response, _ = self.llm.multi_turn_conversation(prompt=final_instruction, messages=code_agent_history_messages)
        # Return the final response
        return final_response