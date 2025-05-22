from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving general tasks.
        Steps:
            1. A primary agent generates an initial solution based on the task description.
            2. A series of subsequent agents generate alternative solutions, each based on previous attempts, fostering diversity and creativity.
            3. A final decision-making agent reasons over all previous solutions and generate the final solution.
        """
        # Initial instruction for the reasoning agent to think step by step and provide the first attempt.
        cot_initial_instruction = (
            f"Task:\n{taskInfo}.\n\n"
            "Please think step by step and then solve the task."
        )
        
        # Instruction for generating alternative solution based on previous attempts. This explores diversity.
        cot_QD_instruction = (
            f"Task:\n{taskInfo}\n\n"
            "Given previous attempts, try to come up with another interesting way to solve the task."
        )

        # Final decision instruction: Reason over all generated solutions and provide the best solution.
        final_decision_instruction = (
            f"Task:\n{taskInfo}\n\n"
            "Given all the above solutions, reason over them carefully and provide a final answer."
        )

        # Initialize possible answers as a list
        possible_answers = []

        # Generate the initial answer
        initial_answer = self.llm.call_llm(cot_initial_instruction)
        # Store the answer for diversity response
        possible_answers.append(initial_answer)

        previous_answer = initial_answer  # Initially, set the previous answer the inital answer. 

        N_max = 3  # Maximum number of attempts to generate diverse solutions        

        # Generate multiple diverse solutions based on previous attempts
        # Unlike repeated questioning, we generate new solutions by exploring alternatives based on prior attempts
        for i in range(N_max):
            # Append the previous attempts to the instruction to create new solutions
            cot_QD_instruction += f"Attempt {i+1}:\n{previous_answer}\n\n"
            # Use the new quality diversity instruction to genearte a new solution
            answer = self.llm.call_llm(cot_QD_instruction)
            # Store the new solution
            possible_answers.append(answer)
            # set previous answer to be answer for iterations
            previous_answer = answer

        # Format the final decision-making instruction with all previous solutions
        for i, solution in enumerate(possible_answers):
            final_decision_instruction += f"Solution {i+1}:\n{solution}\n\n"

        # Call the LLM for reasoning to generate the final answer based on previous answers, with a low temperature to ensure accuracy
        final_answer = self.llm.call_llm(final_decision_instruction, temperature=0.1)

        return final_answer  # Return the final solution

    