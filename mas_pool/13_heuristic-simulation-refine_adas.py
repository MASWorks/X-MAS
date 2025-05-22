from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)
        
        # Pre-defined prompt template
        
        # The prompt template for chain-of-thought reasoning, asking for output in json format for easier extraction.
        self.COT_PROMPT = """Problem Description: {problem}

Function Signature:
{function}

Task:
Given the problem description and the function signature, please think step by step and then solve the task by writing the python code. Ensure the function adheres to the provided signature.

In your output, you should first describe your reasoning and approach to solving the problem. Then provide your code.
"""

        # The prompt template for domain-speciic heuristics and rules.
        self.HEURISTIC_PROMPT = """Problem Description: {problem}

Solution:
{solution}

Task:
Given the task and solution, please apply domain-specific heuristics and rules to refine the solution for improved performance.

In your output, you should first describe your reasoning and approach to solving the problem. Then provide your domain-specific heuristic wrapped in [[]].

For example:
[[your domain-specific heuristic]]

"""

        # The prompt template for validation and get feedback
        self.VALIDATE_PROMPT = """Problem Description: {problem}

Solution:
{solution}

Task:
Given the task and solution, please validate the solution using domain-specific simulation and provide feedback.

In your output, you should first describe your reasoning and approach to solving the problem. Then provide your feedback wrapped in [[]].

For example:
[[your feedback]]

"""

        # The prompt template for refinement, given heuristics and simulation feedbacks
        self.REFINE_PROMPT = """Problem Description: {problem}

Solution:
{solution}

Heuristic:
Here are some domain-specific heuristic and rules.
{heuristic}

Feedback:
Feedback comes from domain-specific simulation. 
{feedback}

Task:
Based on all above, refine the Solution to improve its performance on the Problem. Don't change the format and the function signature of the Solution.

In your output, you should first describe your reasoning and approach to solving the problem. Then provide your code.

Wrap your final code solution in <Code Solution> and </Code Solution>. For example:
<Code Solution>
Your function code here
</Code Solution>
"""

        # The prompt template for final decisions
        self.FINAL_DECISION_PROMPT = """Problem Description: {problem}

Reference Solutions:
These are a few reference solutions and their test feedback respectively.
{solutions}

Task:
Given all the above solutions, reason over them carefully and provide a final answer by writing the python code.

In your output, you should first describe your reasoning and approach to solving the problem. Then provide your code.
"""

    def forward(self, taskInfo):
        """
        Desgin a multi-agent system for code tasks.

        Heuristic Agent: provide domain-specific heuristics and rules.
        Validation Agent: provide feedback with domain-specific simulation.
        Refinement Agent: provide refinement given heuristics and simulation feedbacks.
        Final Decision Agent: synthesize all the information and provide a final decision.
        """

        # call the function to get function signature
        function_signature = get_function_signature(llm=self.llm, taskInfo=taskInfo)
        # call the function to get test cases
        test_cases = get_test_cases(llm=self.llm, taskInfo=taskInfo, function_signature=function_signature)

        # Step 1: Generate initial candidate solutions using multiple LLM agents
        cot_instruction = self.COT_PROMPT.format(problem=taskInfo, function=function_signature)
        num_candidates = 5  # Number of initial candidates

        initial_solutions = []
        for i in range(num_candidates):
            # Call the to generate cot responses
            answer = self.llm.call_llm(cot_instruction, temperature=0.7)
            initial_solutions.append({
                'answer': answer
            })

        # Step 2: Get domain-specific heuristics and rules to refine the generated solutions
        heuristic_solutions = []
        for sol in initial_solutions:
            # format the heuristic instruction
            heuristic_instruction = self.HEURISTIC_PROMPT.format(problem=taskInfo, solution=sol["answer"])
            # Call the llm to generate response and extract heuristic content
            sol["heuristics"] = self.get_bracketed_content(heuristic_instruction, temperature=0.6)
            # store the heuristic
            heuristic_solutions.append(sol)

        # Step 3: Get feedbacks with domain-specific validation
        validated_solutions = []
        for sol in heuristic_solutions:
            # format the validate instruction
            validate_instruction = self.VALIDATE_PROMPT.format(problem=taskInfo, solution=sol["answer"])
            # Call the llm to generate response and extract feedback content
            sol['simulation_feedback'] = self.get_bracketed_content(validate_instruction, temperature=0.5)
            # store the feedback
            validated_solutions.append(sol)

        # Step 4: Refine the solutions using heuristics and feedback from simulations. 
        refined_solutions = []
        for sol in validated_solutions:
            # format the refine instruction with initial answer, heuristic and feedback
            refine_instruction = self.REFINE_PROMPT.format(problem=taskInfo, solution=sol["answer"], heuristic=sol["heuristics"], feedback=sol["simulation_feedback"])
            # Call `generate_and_extract_code` prefefined in utils to generate the refined answer and extract code
            refined_answer, code = generate_and_extract_code(llm=self.llm, prompt=refine_instruction, temperature=0.5)
            
            # call `test_code_get_feedback` function to get test feedback and the correct count
            correct_count, test_feedback = test_code_get_feedback(code, test_cases)
            # store the results
            refined_solutions.append(
                {
                    "answer": refined_answer,
                    "feedback": test_feedback,
                    "correct_count": correct_count
                }
            )
        # sort answers based on the correct count in the test
        sorted_answers = sorted(refined_solutions, key=lambda x: x['correct_count'], reverse=True)

        # Select the top solutions
        top_solutions = sorted_answers[:3]

        # Iteratively format top solutions
        top_solutions_format = ""
        for i,solution in enumerate(top_solutions):
            top_solutions_format += f"[{i+1}]\nsolution:\n{solution['answer']}\nfeedback:{solution['feedback']}\n\n"

        # Use the final decision agent to integrate all information and generate the final answer
        final_decision_instruction = self.FINAL_DECISION_PROMPT.format(problem=taskInfo, solutions = top_solutions_format)
        # Call the llm to generate the final asnwer, using a low temperature for accuracy
        final_response = self.llm.call_llm(final_decision_instruction,temperature=0.1)
        return final_response

    def get_bracketed_content(self, prompt, temperature=None, max_retries=3):
        """
        Call the llm and extract content enclosed in double square brackets ([[ ]]) from the response.

        Args:
            prompt (str): The instruction or query to send to the LLM.
            temperature (float, optional): Sampling temperature for the LLM, controlling the randomness of the response. Default is None.
            max_retries (int): Maximum number of retries if the content is not found. Default is 3.

        Returns:
            str: The first content match enclosed in [[ ]] from the LLM response, or "None" if no match is found.
        """
        attempts = 0  # Track the number of retry attempts

        while attempts < max_retries:
            # Call the llm to generate the response with given prompt and temperature
            if temperature:
                llm_response = self.llm.call_llm(prompt, temperature=temperature)
            else:
                llm_response = self.llm.call_llm(prompt)

            # Pattern to match content enclosed in double square brackets ([[ ]])
            pattern = r'\[\[(.*?)\]\]'
            matches = re.findall(pattern, llm_response)

            if matches:  # If a match is found, return the first match
                return matches[0]

            # Increment retry counter if no match is found
            attempts += 1

        # Return "None" if all retries are exhausted without a match
        return llm_response
    
    def generate_and_extract_code(self, prompt, temperature=None, max_attempts=3):
        """
        Generate a response from the LLM and extract the contained code with retry logic.

        This function attempts to generate a response from the LLM containing a code snippet.
        It extracts the code snippet from the response and returns both the full response and the extracted code. 
        If no valid code is found after multiple attempts, it returns the last response and an empty string for the code.

        Args:
            prompt (str): The instruction to send to the LLM to generate a response with code.
            temperature (float, optional): Sampling temperature for the LLM, controlling randomness in the output.
            max_attempts (int): Maximum number of attempts to fetch a response with valid code. Default is 3.
            
        Returns:
            tuple:
                str: The full LLM response.
                str: The extracted code snippet, or an empty string if no valid code is detected.
        """
        attempts = 0  # Track the number of attempts
        
        while attempts < max_attempts:
            # Generate response using the LLM
            if temperature:
                llm_response = self.llm.call_llm(prompt, temperature=temperature)
            else:
                llm_response = self.llm.call_llm(prompt)
            
            # Extract the code snippet from the response
            code_snippet = extract_code(llm_response)
            
            if code_snippet:  # If a valid code snippet is found, return the response and the code
                return llm_response, code_snippet
            
            attempts += 1  # Increment attempts and retry if no valid code is detected
        
        # Return the last LLM response and an empty code snippet after exhausting all attempts
        return llm_response, ""