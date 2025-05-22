from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving scientific problems.
        
        Steps:
            1. 5 agents think step by step and solve the task independently.
            2. Each agent verifies the correctness of the calculations by writing Python code. Then modify the solution based on the verification results.
            2. A final decision-making agent reasons over the 5 modified solutions and provides the final solution.
        """

        # Step-by-step instruction for each chain-of-thought agent to reason and generate answer
        cot_instruction = f"""You are an expert in solving scientific problems.
Problem: {taskInfo}

Please think step by step, first clarify the formula and concepts involved in the problem. Then solve the problem step by step."""

        # Set the number of solutions to generate; using 5 for variety and diversity
        N = 5
        # Call the llm to generate each solution independently
        cot_results = [self.llm.call_llm(cot_instruction) for _ in range(N)]  

        # Verify the correctness of the calculations by writing Python code and modify the solution based on the verification results
        modified_solutions = [self.code_verify(taskInfo, solution) for solution in cot_results]

        # Use a final decision-making agent to reason over the solutions and provide the final solution
        final_solution = self.get_final_solution(taskInfo=taskInfo, solutions=modified_solutions)
       
        # Return the final solution
        return final_solution

    def code_verify(self, taskInfo, solution):
        """
        Verify the correctness of the calculations by writing Python code and modify the solution based on the verification results.
        Args:
            taskInfo (str): The description of the task.
            solution (str): The initial solution to be verified.

        Returns:
            str: The modified solution based on the verification results.
        """
        # Instruction for verifying the correctness of the calculations by writing Python code
        verfiy_instruction = f"""**Problem:**
{taskInfo}

**Solution:**
{solution}

The **Solution** is an initial attempt to solve the problem. Your task is to verify the correctness of this solution by writing Python code that validates the calculations. Make sure your code can be directly executed. Follow these instructions:

1. Use the initial Solution as a reference for your calculations. Your Python code should align with the logic and steps outlined in the **Solution**.
2. Use the values mentioned directly in the **Problem** for your calculation.
2. Include detailed comments and explanations within the code to clarify the implementation. Print relevant intermediate results to enhance clarity. 
3. Ensure the final result is stored in a variable named `output`. This variable must be defined at the global scope and contain the final computation result.
4. Wrap your Python code in <Code Solution> and </Code Solution> tags for easy identification.

Output Format:
<Code Solution>
Your code here
</Code Solution>"""
        # Call `generate_and_extract_code` to generate answer and extract the code
        _, code = generate_and_extract_code(llm=self.llm, prompt=verfiy_instruction)
        # Call `execute_code` to execute the code and get the output
        output = execute_code(code)
        # Modify the initial solution based on the verification results
        modify_instruction = f"""**Problem:**
{taskInfo}

**Solution:**
{solution}

**Verification Code:**
{code}

**Code Execution Result:**
{output} 

The **Problem** is the problem to be solved. The **Solution** is an initial attempt to solve the problem. The **Verification Code** is the Python code to verify the correctness of the initial solution. The **Code Execution Result** is the output of the verification code.

**Task:**
Judge the correctness of the initial solution based on the verification results. 
If correct, you can use the initial solution as the final answer.
If incorrect, modify the initial solution and provide a final answer to the problem.

In your output, you should give the final solution to the **Problem**. As like you are answering the question directly."""
        modified_solution = self.llm.call_llm(modify_instruction)
        return modified_solution

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
        instruction += "Given all the above solutions, reason over them carefully and provide a final answer to the task. Identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution. In your output, you should give the final solution to the **Problem**. As like you are answering the question directly."
        
        # Call the LLM to generate the final solution, using a low temperature for accuracy
        final_solution = self.llm.call_llm(prompt=instruction, temperature=0.3)
        return final_solution