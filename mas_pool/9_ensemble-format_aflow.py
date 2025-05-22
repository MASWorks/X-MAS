from utils import * 

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        Design a multi-agent system to produce direct and concise answer.
        
        Steps:
            1. Generate multiple solutions as candidates.
            2. Use an ensemble mechanism to select the most reliable solution.
            3. Format the selected solution into a concise response.
        """
        solutions = []  # List to store generated solutions.

        # The instruction for generating answers.
        answer_generation_instruction = f"""Task:
{taskInfo}

Think step by step and solve the problem. First describe your reasoning and approach. Then provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.

For example:
Thinking:
Your reasoning and approach here.
Answer:
Your final answer here.

"""
        for _ in range(5):  # Generate five solutions for ensemble evaluation.
            # Call `call_llm` to generate a response.
            answer = self.llm.call_llm(answer_generation_instruction, temperature=0.6)
            solutions.append(answer)

        # Use the ensemble mechanism to select the best answer.
        best_answer = self.sc_ensemble(taskInfo=taskInfo, solutions=solutions)

        # Format the best answer into a concise response.
        format_answer_instruction = f"""Task:
{taskInfo}

Answer:
{best_answer}

Given the question and the answer, format the final answer to be concise, \
accurate, and directly addressing the question. Ensure the answer is a clear, \
brief statement without additional explanation or reasoning. If the answer is a \
name, profession, or short phrase, provide only that information without forming a \
complete sentence.

For example:
- If the answer is a person's name, just provide the name.
- If the answer is a profession, state only the profession.
- If the answer is a short phrase, give only that phrase.

Do not include any prefixes like "The answer is" or "The profession is". Just provide the answer itself.
"""
        format_solution = self.llm.call_llm(format_answer_instruction, temperature=0.1)

        return format_solution  # Return the final formatted solution.

    # Function to implement the ensemble voting mechanism for selecting the best solution.
    def sc_ensemble(self, taskInfo, solutions, max_retries=3):
        """
        Self-Consistency Ensemble to select the most reliable solution.

        Args:
            taskInfo (str): The description of the task.
            solutions (list): The solutions to be evaluated.
            max_retries (int): Maximum number of retries if solution extraction fails. Default is 3.

        Returns:
            str: The final chosen solution after Self-Consistency Ensemble.
        """
        import re

        # Prepare the solutions in a numbered format for evaluation
        solutions_str = ""
        for i, solution in enumerate(solutions):
            solutions_str += f"[[{i+1}]]\n{solution}\n\n"

        # Prompt for evaluating multiple generated solutions and selecting the most reliable one.
        ensemble_instruction = f"""Given the question described as follows: {taskInfo}
Several solutions have been generated to address the given question. They are as follows:
{solutions_str}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

Provide a detailed explanation of your thought process. Then give your choice of the most reliable answer. Output only the ID (<<1>>, <<2>>]], etc.) corresponding to the most consistent solution. Wrap your final choice with <<>> and without including any additional text or explanation in it.

For example:
<<1>>
"""

        retries = 0
        while retries < max_retries:
            # Call the LLM to generate an answer, a ralatively low temperature to improve accuracy
            answer = self.llm.call_llm(ensemble_instruction, temperature=0.3)

            # Attempt to extract the selected solution ID from the response
            numbers = re.findall(r'\<\<(\d+)\>\>', answer)
            if numbers:  # If at least one match is found
                try:
                    index = int(numbers[0]) - 1  # Convert to zero-based index
                    if 0 <= index < len(solutions):
                        return solutions[index]  # Return the selected solution
                except ValueError:
                    pass  # In case the match is not a valid number

            # Increment retry counter if extraction fails
            retries += 1

        # If all retries fail, return the first solution as a fallback
        return solutions[0]