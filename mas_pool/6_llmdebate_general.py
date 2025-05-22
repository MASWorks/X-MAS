from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving scientific tasks.
        
        Steps:
            1. An initial agent generates a solution using a general prompt.
            2. Expert agents in different roles(Mathematics, Physics, Chemistry, Biology Expert) reflect on existing solutions and provide different views based on their domain expertise, simulating a debate process. 
            3. Iterate over the debate process for a maximum number of rounds to refine the solution.
            4. A Science Generalist agent integrates the solutions from domain-specific experts and provides a final answer.
        """

        # Define expert roles and their detailed descriptions for task-specific reasoning
        expert_roles = ['Mathematics Expert', 'Physics Expert', 'Chemistry Expert', 'Biology Expert']
        role_descriptions = [
            # Mathematics Expert: Focus on numerical accuracy, advanced equations, and logical consistency.
            "You are a Mathematics Expert skilled in solving complex equations, mathematical modeling, and logical problem-solving. Given the problem, provide detailed calculations, step-by-step reasoning, and a precise mathematical solution using advanced mathematical concepts where necessary.",
            # Physics Expert: Provide insights into physical laws, mechanics, and systems analysis.
            "You are a Physics Expert specializing in mechanics, electromagnetism, thermodynamics, and quantum physics. Given the problem, use your expertise to analyze physical systems, apply relevant laws, and explain phenomena with accurate calculations and clear conceptual reasoning.",
            # Chemistry Expert: Analyze chemical processes, molecular structures, and reaction details.
            "You are a Chemistry Expert proficient in chemical reactions, molecular structures, and laboratory analysis. Provide insights into chemical processes, detailed explanations of reactions, and any necessary calculations for solving chemistry-related problems.",
            # Biology Expert: Address biological principles, cellular processes, and ecological interactions.
            "You are a Biology Expert knowledgeable in cellular biology, genetics, ecology, and physiology. Use your expertise to analyze biological systems, explain processes at the molecular and ecological levels, and provide thorough explanations grounded in biological principles.",
        ]

        # Generate the initial solution using a general prompt
        initial_prompt = (
            f"Task:\n{taskInfo}\n\n"
            f"Given the task, please think step by step and provide your answer."
        )
        # Call the llm to generate solution
        initial_solution = self.llm.call_llm(initial_prompt)

        max_round = 2  # Maximum number of debate rounds
        # Initialize a list to hold solutions for each round of debate
        all_results = [[] for _ in range(max_round)]
        # Instruction for debating and updating the solution based on other agents' solutions
        debate_instruction = f"Problem:\n{taskInfo}\n\nGiven solutions to the problem from other agents, reason over them carefully and consider their reasoning as additional advice. Make full use of your domain knowledge and expertise to provide a unique perspective on the problem. Please think carefully and offer an updated response."
        
        # Execute multiple rounds of debate where agents refine their solutions
        for r in range(max_round):
            for i in range(len(expert_roles)):
                if r == 0:
                    # In the first round, provide role description along with the initial solution for agents to refine
                    prompt = f"Role Description:\n{role_descriptions[i]}\n\n{debate_instruction}\n\nSolutions:\n{initial_solution}"
                else:
                    # In later rounds, include the previous round's solutions in the prompt to refine further
                    prompt = f"Role Description:\n{role_descriptions[i]}\n\n{debate_instruction}\n\n"
                    for j in range(len(all_results[r-1])):
                        prompt += f"Solution {j+1}:\n{all_results[r-1][j]}\n\n"
                        
                # Call the llm to generate a new answer based on the role and existing solutions, using a high temperature for diversity.
                answer = self.llm.call_llm(prompt=prompt, temperature=0.8)
                # Store the generated answer in the list of results for the current round
                all_results[r].append(answer)

        # Compile the solutions from each expert role for the final decision-making
        compiled_solutions = ""
        for i, expert in enumerate(expert_roles):
            compiled_solutions += f"{expert} Solution:\n{all_results[max_round-1][i]}\n\n"
            
        # Final decision-making prompt by the Science Generalist to integrate all answers from experts and gibe a final answer
        final_decision_prompt = (
            f"Problem:\n{taskInfo}\n\n"
            f"Role Description:\nYou are a Science Generalist with a broad understanding across multiple scientific domains, including math, physics, chemistry, and biology. For a given problem, synthesize knowledge from different fields, provide interdisciplinary insights, and ensure a comprehensive and balanced explanation.\n\n"
            f"{compiled_solutions}"
            f"Task:\nGiven all the thinking and answers to the problem provided by other agents, reason over them carefully and provide a final answer to the problem."
        )
        # call the llm to generate the final answer
        final_solution = self.llm.call_llm(final_decision_prompt)
        # return the final solution
        return final_solution