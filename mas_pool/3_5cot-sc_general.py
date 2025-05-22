from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving general tasks.
        
        Steps:
            1. 5 agents think step by step and solve the task independently.
            2. A final decision-making agent reasons over the solutions and provides the final solution.
        """

        # Step-by-step instruction for each chain-of-thought agent to reason and generate answer
        cot_instruction = f"Task: {taskInfo}\n\nPlease think step by step and then solve the task."

        # Set the number of solutions to generate; using 5 for variety and diversity
        N = 5
        # Call the llm to generate each solution independently
        cot_results = [self.llm.call_llm(cot_instruction) for _ in range(N)]  

        # Use a final decision-making agent to reason over the solutions and provide the final solution
        final_solution = self.get_final_solution(taskInfo=taskInfo, solutions=cot_results)
       
        # Return the final solution
        return final_solution

    def get_final_solution(self, taskInfo, solutions):
        """
        Based on the given task, aggregate solutions and generate a final solution.

        Args:
            taskInfo (str): A description of the task that needs to be completed.
            solutions (list): A list containing solutions for the task.

        Returns:
            str: The final solution aggregating the solutions.
        """

        # Initialize the instruction text with a general guideline
        instruction = f"Task:\n{taskInfo}\n\n"

        # Append each solution to the instruction
        for i, solution in enumerate(solutions):
            instruction += f"Solution {i+1}:\n{solution}\n\n"  # Number each solution for clarity

        # Add the final prompt to encourage reasoning over the solutions and provide a final answer
        instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task."
        
        # Call the LLM to generate the final solution, using a low temperature for accuracy
        final_solution = self.llm.call_llm(prompt=instruction, temperature=0.3)
        
        return final_solution