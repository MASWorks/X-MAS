from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving general tasks.
        
        Steps:
            1. 5 agents solve the task independently.
            2. A final decision-making agent reasons over the solutions and provides the final solution.
        """

        # Step-by-step instruction for each agent to reason and generate answer
        instruction = f"Task: {taskInfo}\n\nPlease solve the task."

        # Set the number of solutions to generate; using 5 for variety and diversity
        N = 5
        # Call the llm to generate each solution
        cot_results = [self.llm.call_llm(instruction) for _ in range(N)]  

        # Get the instruction for the final decision-making agent based on all generated solutions
        final_decision_instruction = self.get_final_decision_instruction(taskInfo, cot_results)

        # Call the llm to process the final decision-making instruction and generate the final answer
        final_decision_result = self.llm.call_llm(final_decision_instruction)

        # Return the final solution
        return final_decision_result

    def get_final_decision_instruction(self, taskInfo, cot_results):
        """
        Format an instruction for final decision-making based on a given task description and a list of solutions.

        Args:
            taskInfo (str): A description of the task that needs to be completed.
            cot_results (list): A list containing solutions or reasoning steps for the task.

        Returns:
            str: A formatted instruction that includes the task description, each solution, and a prompt for final decision-making.
        """

        # Initialize the instruction text with a general guideline
        instruction = f"Task:\n{taskInfo}\n\n"

        # Append each solution from cot_results to the instruction
        for i, result in enumerate(cot_results):
            instruction += f"Solution {i+1}:\n{result}\n\n"  # Number each solution for clarity

        # Add the final prompt to encourage reasoning over the solutions and provide a final answer
        instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task."
        
        # Return the complete instruction text
        return instruction