from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving coding tasks.
        Steps:
            1. A primary agent generates an initial solution based on the task description.
            2. A series of subsequent agents generate alternative solutions, each based on previous attempts, fostering creativity.
            3. The best solutions are selected through testing and comparison.
            4. A final decision-making agent reasons over the top solutions and generate the final solution.
        """
        # Calls `get_function_signature` to generate the function signature based on the task information
        function_signature = get_function_signature(llm=self.llm, taskInfo=taskInfo)

        # Calls `get_test_cases` to generate a list of test cases for the function
        test_cases = get_test_cases(llm=self.llm, taskInfo=taskInfo, function_signature=function_signature)

        # Initial instruction for the reasoning agent to think step by step and provide the first code attempt.
        cot_initial_instruction = (
            f"Task:\n{taskInfo}.\n\n"
            f"Function Signature:\n{function_signature}\n\n"
            "Please think step by step and then solve the task by writing python code. Ensure the function adheres to the provided signature.\n"
            "Wrap your final code solution in <Code Solution> and </Code Solution>. For example:\n"
            "<Code Solution>\n"
            "Your function code here\n"
            "</Code Solution>\n"
        )# fix the output format for easier extraction
        
        # Call `generate_and_extract_code` to generate the answer and extract the code
        initial_answer, code = generate_and_extract_code(llm=self.llm, prompt=cot_initial_instruction)

        possible_answers = []

        # Store the answer and code for subsequent test
        possible_answers.append(
            {
                "answer": initial_answer,
                "code": code
            }
        )
        
        # Instruction for generating alternative solution based on previous attempts. This explores diversity.
        cot_QD_instruction = (
            f"Task:\n{taskInfo}\n\n"
            f"Function Signature:\n{function_signature}\n\n"
            "Given previous attempts, try to come up with another interesting way to solve the task by writing the python code.\n"
            "Ensure the function adheres to the provided signature.\n"
            "Wrap your final code solution in <Code Solution> and </Code Solution>. For example:\n"
            "<Code Solution>\n"
            "Your function code here\n"
            "</Code Solution>\n"
        )
        
        previous_answer = initial_answer  # Initially, set the previous answer the inital answer. 

        N_max = 3  # Maximum number of attempts to generate diverse solutions        

        # Generate multiple diverse solutions based on previous attempts
        # Unlike repeated questioning, we generate new solutions by exploring alternatives based on prior attempts
        for i in range(N_max):
            # Append the previous attempts to the instruction to create new solutions
            cot_QD_instruction += f"Attempt {i+1}:\n{previous_answer}\n\n"
            # Call `generate_and_extract_code` to genearte a new solution
            answer, code = generate_and_extract_code(llm=self.llm, prompt=cot_QD_instruction)
            # Store the new solution
            possible_answers.append(
                {
                    "answer": answer,
                    "code": code
                }
            )
            # set previous answer to be answer for iterations
            previous_answer = answer

        # Final decision instruction: Reason over all generated solutions and provide the best code solution.
        final_decision_instruction = (
            f"Task:\n{taskInfo}\n\n"
            f"Function Signature:\n{function_signature}\n\n"
            "Given all the solutions and their feedbacks from test, reason over them carefully and provide a final answer by writing the python code.\n"
            "Ensure the function adheres to the provided signature."
        )

        top_k = 2   # The number of top answers to be given to the final decision-making agent
        # Use the test_and_sort function to test the generated solutions and choose the top k best answers
        top_solutions = self.test_and_sort(taskInfo=taskInfo, test_cases=test_cases, solutions=possible_answers, top_k=top_k)

        # Add the top 2 solutions to the final decision-making instruction
        for i, solution in enumerate(top_solutions):
            final_decision_instruction += f"Solution {i+1}:\n{solution['answer']}\nFeedback:\n{solution['feedback']}\n\n"

        # Call the LLM for reasoning to generate the final answer based on the top k answers, with a low temperature to ensure accuracy
        final_answer = self.llm.call_llm(final_decision_instruction, temperature=0.1)

        return final_answer  # Return the final best solution
    
    def test_and_sort(self, taskInfo, test_cases, solutions, top_k):
        """
        This function tests the generated code solutions against test cases and selects the best solutions based on the scores.
        
        Args:
            taskInfo (str): The task description that the code is trying to solve.
            test_cases (list): Test cases used to test the solutions.
            solutions (list): A list of code solutions to test.
            top_k (int): The number of best solutions to return based on their performance.

        Returns:
            top_k_answers (list): A list of the top K code solutions based on their testing performance.
        """
        all_results = []
        for solution in solutions:
            # get the code part of the solution for test
            code = solution["code"]
            # Call `test_code_get_feedback` to test the code
            correct_count, feedback = test_code_get_feedback(code, test_cases=test_cases)
            # Save the solution, correct count and feedback
            all_results.append(
                {
                    "answer": solution["answer"],
                    "code": solution["code"],
                    "correct_count": correct_count,
                    "feedback": feedback
                }
            )
        # sort the solutions with the correct count
        sorted_solutions = sorted(all_results, key=lambda x: x["correct_count"], reverse=True)

        # Select the top K solutions based on their scores
        top_k_solutions = [solution for solution in sorted_solutions[:top_k]]

        return top_k_solutions  # Return the top K best solutions