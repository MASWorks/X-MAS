from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving coding tasks.
        Steps:
            1. An agent thinks step by step and writes the code.
        """
        # Instruction for the Chain-of-Thought (CoT) approach with code generation
        cot_instruction = f"Task: {taskInfo}\n\nPlease think step by step and then solve the task by writing the code."
        
        # Call the llm to generate answer according to the instruction
        answer = self.llm.call_llm(cot_instruction)
        
        # Return the final solution
        return answer