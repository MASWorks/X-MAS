from utils import *  

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving math problems.
        Steps:
            1. A code-generation agent which generate code to help solve the problem, and a refine agent answering the problem based on the output of the code.
            2. A detail agent providing detailed solution.
            3. A basic agent providing solution by thinking step by step.
            4. A final ensembler which evaluates all the solutions and determine the most reliable one.
        """
        # Get the refinement solution by generating code and answering the problem based on the output of the code.
        refinement_solution = self.coding_and_refinement(taskInfo)

        # Generate a detailed solution.
        # Prompt for providing a more detailed and explanatory solution, suitable for educational purposes.
        DETAILED_SOLUTION_PROMPT = r"""
Provide a comprehensive, step-by-step solution to the given mathematical problem. Your
response should include:
1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \boxed{} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.
Ensure your solution is rigorous, easy to follow, and educational for someone learning
the concept.
"""
        detail_solution_instruction = DETAILED_SOLUTION_PROMPT + f"\n\nproblem:\n{taskInfo}\n"
        # Call the llm to generate answer
        detail_solution = self.llm.call_llm(detail_solution_instruction)

        # Combine the refined solution and the detailed solution into a list.
        solutions = [refinement_solution, detail_solution]

        # Generate additional solutions for ensemble voting.
        # Prompt for generating a basic, step-by-step solution.
        GENERATE_SOLUTION_PROMPT = r"""
Please solve the given mathematical problem step by step. Follow these guidelines:
1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Present the final answer enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.
Your solution should be thorough, mathematically sound, and easy to understand.
"""
        generate_solution_instruction = GENERATE_SOLUTION_PROMPT + f"\n\nproblem:\n{taskInfo}\n"
        for _ in range(2):  # Generate two additional solutions.
            # Call the llm to generate answer
            answer = self.llm.call_llm(generate_solution_instruction)
            solutions.append(answer)
        
        # Select the most reliable solution using the ensemble process.
        final_solution = self.sc_ensemble(taskInfo=taskInfo, solutions=solutions)

        return final_solution

    def coding_and_refinement(self, taskInfo):
        """
        Generate code to help solve the problem and refine the solution based on the output of the code.
        Args:
            taskInfo (str): The description of the task.

        Returns:
            str: The refined solution based on the code output.
        """
        # The instruction to generate Python code to help solve the problem based on the provided task information.
        code_generation_instruction = f"""
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {taskInfo}

Your code should:
1. Implement the calculation steps described in the problem.
2. For every intermediate result in the calculation process, print the result along with a clear, descriptive message that explains what the intermediate result represents.
3. Store the final calculation result in a variable named `output`. This variable should contain the final result of the computation and be defined at the global scope.
4. Ensure that `output` contains the result in a basic data type such as a string, integer, or float.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.

Wrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your function code here
</Code Solution>
""" # ask for specific output format for easier extraction

        # Call `generate_and_extract_code` to generate answer and extract the code
        answer, code = generate_and_extract_code(llm=self.llm, prompt=code_generation_instruction)
        # Execute the generated Python code and obtain the output.
        output = execute_code(code)
        
        # Refine the solution using the code output.
        REFINE_ANSWER_PROMPT = r""" 
Given the mathematical problem and the output from the code execution, please provide
a well-formatted and detailed solution. Follow these guidelines:
1. Begin with a clear statement of the problem.
2. Explain the approach and any formulas or concepts used.
3. Show step-by-step calculations, using LaTeX notation for mathematical expressions.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer, enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.
Your response should be comprehensive, mathematically rigorous, and easy to follow.
"""
        refine_instruction = REFINE_ANSWER_PROMPT + f"\n\nproblem:\n{taskInfo}\n\nCode Solution:\n{answer}\n\nCode Execution Output:\n{output}\n\n"
        # Call the llm to generate answer
        refinement_solution = self.llm.call_llm(refine_instruction)
        
        return refinement_solution
        
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