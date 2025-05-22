from utils import *
import re

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
        initial_instruction = f"Problem: {taskInfo}\n\nPlease think step by step and then solve the problem."
        response = self.llm.call_llm(initial_instruction)

        max_attempts = 3
        for _ in range(max_attempts):
            response, is_correct = self.self_refine(taskInfo, response)
            if is_correct:
                return response
        
        return response
    
    def self_refine(self, taskInfo, previous_response):
        """
        Self-refine to improve the previous response.

        Args:
            taskInfo (str): The task information.
            previous_response (str): The previous response.
        
        Returns:
            A tuple of two elements:
                - The refined response.
                - A boolean indicating whether the previous response is correct.
        """
        self_refine_instruction = f"""**Problem:**
{taskInfo}

**Response:**
{previous_response}

The **Problem** is the problem to be solved. The **Response** is an attempt to the problem.
You need to carefully go through semantically the response, check the correctness of every step of the attempt. Then you need to refine the response and provide the final solution to the problem.

Output Format:
Your output should include two parts:
1. The feedback on the previous response. If you think the given response is completely correct, output <<OK>> in your feedback. (Wrap your feedback in <Feedback> and </Feedback>)
2. Your final response to the problem. (Wrap your final response in <Final Response> and </Final Response>)

For example:
<Feedback>
Your feedback here
</Feedback>

<Final Response>
Your final response here
</Final Response>
"""
        self_refine_response = self.llm.call_llm(self_refine_instruction)
        def extract_feedback_and_response(self_refine_response):
            feedback_pattern = r"<Feedback>\s*(.*?)\s*</Feedback>"
            response_pattern = r"<Final Response>\s*(.*?)\s*</Final Response>"
            feedback_match = re.search(feedback_pattern, self_refine_response, re.DOTALL)
            response_match = re.search(response_pattern, self_refine_response, re.DOTALL)
            feedback = feedback_match.group(1) if feedback_match else self_refine_response
            response = response_match.group(1) if response_match else self_refine_response
            return feedback, response
        feedback, response = extract_feedback_and_response(self_refine_response)
        if "<<OK>>" in feedback:
            return response, True
        else:
            return response, False