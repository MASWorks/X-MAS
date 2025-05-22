from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, task_description):
        """
        A muti-agent system for solving scientific problems.
        
        Steps:
            1. Identify the relevant principles (mathematics, physics, chemistry, or biology) required to solve the problem.
            2. Use the identified principles to solve the task step by step.
        """
        # Step 1: Identify the principles involved in solving the task
        # Purpose: Understand the underlying concepts and principles (math, physics, chemistry, or biology) necessary for solving the task.
        principle_instruction = (
            f"Problem: {task_description}\n\n"
            f"Task:\nGiven the problem, what are the math, physics, chemistry, or biology principles and concepts involved in solving this task?\n\n"
            f"First think step by step. Then list all involved principles and explain them."
        )
        # Call the llm to generate the principles and their explanations
        principle_response = self.llm.call_llm(principle_instruction)

        # Step 2: Solve the task based on the identified principles
        # Purpose: Leverage the identified principles to reason through and solve the problem step by step.
        cot_instruction = (
            f"Problem: {task_description}\n\n"
            f"Principles:\n{principle_response}\n\n"
            f"Task:\nGiven the problem and the involved principles behind the problem, think step by step and then solve the task."
        )
        # Call the llm to generate the solution to the task
        final_solution = self.llm.call_llm(cot_instruction)

        # Step 3: Return the final solution
        return final_solution