from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving general tasks.
        Steps:
            1. An agent thinks step by step and gives the solution.
        """
        # Instruction for the Chain-of-Thought (CoT) approach
        cot_instruction = f"Task: {taskInfo}\n\nPlease think step by step and then solve the task."
        
        # Call the llm to generate answer according to the instruction
        answer = self.llm.call_llm(cot_instruction)
        
        # Return the final solution
        return answer