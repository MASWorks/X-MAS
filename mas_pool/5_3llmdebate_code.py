from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi agent system for solving coding tasks.
        Steps:
            1. An initial agent solves the task by writing the code.
            2. Three agents(Coding Artist, AlgorithmDeveloper, ComputerScientist) reflect on existing solutions and provide different views, simulating a debate process.
            3. Iterate over the debate process for a maximum number of rounds to refine the code.
            4. A final decision-making agent reason over the solutions and generates the final code.
        """
        # Step-by-step instruction for the initial chain-of-thought agent to reason and generate code
        initial_instruction = f"Task:\n{taskInfo}\n\nPlease think step by step and then solve the task by writing the code."
        # Call the llm to generate the initial answer based on the initial prompt
        initial_answer = self.llm.call_llm(prompt=initial_instruction)
        
        # Instruction for debating and updating the solution based on other agents' solutions
        debate_instruction = f"Problem:\n{taskInfo}\n\nGiven solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer by writing the code."
        
        # Initialize the list of debate agents and their respective roles
        debate_agents = ["Coding Artist", "AlgorithmDeveloper", "ComputerScientist"]
        role_descriptions = [
            # Description of the "Coding Artist" role, ask for the elegance of the code
            "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility.",
            # Description of the "Algorithm Developer" role, foucusing on algorithms
            "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems.",
            # Description of the "Computer Scientist" role, foucsing on high performance
            "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solving real problems."
        ]    

        max_round = 2  # Maximum number of debate rounds

        # Initialize a list to hold solutions for each round of debate
        all_results = [[] for _ in range(max_round)]
        
        # Execute multiple rounds of debate where agents refine their solutions
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0:
                    # In the first round, provide role description along with the initial solution for agents to refine
                    prompt = f"Role Description:\n{role_descriptions[i]}\n\n{debate_instruction}\n\nSolutions:\n{initial_answer}"
                else:
                    # In later rounds, include the previous round's solutions in the prompt to refine further
                    prompt = f"Role Description:\n{role_descriptions[i]}\n\n{debate_instruction}\n\n"
                    for j in range(len(all_results[r-1])):
                        prompt += f"Solution {j+1}:\n{all_results[r-1][j]}\n\n"
                        
                # Call the llm to generate a new answer based on the role and existing solutions, using a high temperature for diversity.
                answer = self.llm.call_llm(prompt=prompt, temperature=0.8)
                # Store the generated answer in the list of results for the current round
                all_results[r].append(answer)
        
        # Instruction for final decision-making based on all debates and solutions
        final_decision_instruction = f"Task:\n{taskInfo}\n\nGiven all the thinking and answers, reason over them carefully and provide a final answer by writing the code."
        
        # Make the final decision based on the results of the last round of debate
        for i in range(len(all_results[max_round-1])):
            # Add all solutions from the last round to the final decision-making instruction
            final_decision_instruction +=  f"Solution {i+1}:\n{all_results[max_round-1][i]}\n\n"
        
        # Call the llm to produce the final decision, with a low temperature to ensure accuracy
        answer = self.llm.call_llm(prompt=final_decision_instruction, temperature=0.1)
        
        return answer  # Return the final answer