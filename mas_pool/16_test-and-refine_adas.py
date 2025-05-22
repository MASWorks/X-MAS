from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    # Main forward function to generate solutions, refine them, and return the best one
    def forward(self, taskInfo):
        """
        Design a multi-agent system for code generation.
        
        Steps:
            1. A strategy agent to provide high-level strategy guilelines.
            2. A implementation agent to write code based on the strategy.
            3. A refinement agent to iteratively refine the soution based on the test feedbacks.
            4. A top solution is selected through testing and comparison as the final output.
        """
        
        # Calls `get_function_signature` to generate the function signature based on the task information
        function_signature = get_function_signature(llm=self.llm, taskInfo=taskInfo)

        # Calls `get_test_cases` to generate a list of test cases for the function
        test_cases = get_test_cases(llm=self.llm, taskInfo=taskInfo, function_signature=function_signature)

        # Get the strategy for the task
        strategy = self.get_strategy(taskInfo=taskInfo)

        # Instruction for the implementation agent to write the code based on the strategy
        implementation_instruction = f"""Problem Description: {taskInfo}

Strategy: 
{strategy}

Function Signature:
{function_signature}

Task:
Based on the provided problem and strategy, write a Python function that solves the problem. Ensure the function adheres to the provided signature.

In your output, you should first describe your reasoning and approach to solving the problem. Then provide the code.

Wrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your function code here
</Code Solution>
"""
        # Call `generate_and_extract_code` to genearte a code implementation
        answer, code = generate_and_extract_code(llm=self.llm, prompt=implementation_instruction)

        # List to store potential solutions with their feedback and correct count
        possible_solutions = []
        
        # Defines the maximum number of refinement attempts
        N_max = 3
        
        # Loop through the refinement process up to the maximum allowed attempts
        for _ in range(N_max):
            # Calls `test_code_get_feedback` function to test the code implementation with the generated test cases
            correct_count, feedback = test_code_get_feedback(code, test_cases)
            # Appends the current implementation, feedback, and correctness count to the solutions list
            possible_solutions.append({"answer": answer, "feedback": feedback, "correct_count": correct_count})
            # If the implementation passes all test cases, break the loop
            if correct_count == len(test_cases):
                break
            
            # Instruction for refining the implementation based on the feedback
            refine_instruction = f"""Problem Description: {taskInfo}

Solution:
{answer}

Feedback:
{feedback}

Task:
Based on the provided problem, solution and feedback, refine the code to improve its performance. Don't change the function signature.

In your output, you should first describe your reasoning and approach to solving the problem. Then provide the code.

Wrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your function code here
</Code Solution>
"""
            # Call `generate_and_extract_code` to refine the code implementation
            answer, code = generate_and_extract_code(llm=self.llm, prompt=refine_instruction)

        # Sorts all the generated solutions by their correctness score in descending order
        sorted_answers = sorted(possible_solutions, key=lambda x: x['correct_count'], reverse=True)
        # Selects the solution with the highest correctness score as the top solution
        top_solution = sorted_answers[0]

        # Returns the best solution
        return top_solution["answer"]

    def get_strategy(self, taskInfo, temperature=None, max_retries=3):
        """
        Generate a response from the LLM and extract the strategy enclosed in double square brackets ([[ ]]).

        Args:
            taskInfo (str): The task description.
            temperature (float, optional): Sampling temperature for the LLM, controlling the 
                randomness or diversity of the response. If not specified, a default value is used.
            max_retries (int): Maximum number of attempts to extract valid content from the LLM response. 
                Default is 3.

        Returns:
            str: The strategy extracted from the LLM response that is enclosed in [[ ]]. 
                If no valid content is found after the maximum retries, return the whole response.
        """
        # Get the instruction prompt for generating high-level strategies based on the problem description
        strategy_instruction = f"""Problem Description: {taskInfo}

Task:
Given the problem description, please generate high-level strategies and detailed explanations for transforming the input grid to the output grid.

In your output, you should first describe your reasoning and approach to solving the problem. Then provide your strategy wrapped in <Strategy> and </Strategy>.

For example:
<Strategy>
Your strategy here
</Strategy>
"""
        attempts = 0  # Track the number of retry attempts
        while attempts < max_retries:
            # Call the LLM to generate the response with the given prompt and temperature
            if temperature:
                response = self.llm.call_llm(strategy_instruction, temperature=temperature)
            else:
                response = self.llm.call_llm(strategy_instruction)

            # Pattern to match content wrapped in <Strategy> tags
            pattern = r"<Strategy>\s*(.*?)\s*</Strategy>"
            match = re.search(pattern, response)

            # Extract and validate the matched content
            if match:
                strategy = match.group(1).strip()
                return strategy

            # Increment the retry counter if no valid match is found
            attempts += 1

        # Return the whole response if all retries are exhausted without finding a valid match
        return response