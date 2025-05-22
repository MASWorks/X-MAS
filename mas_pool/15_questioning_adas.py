from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving general tasks.
        
        Steps:
            1. A questioning agent to generate Socratic questions to better understand and solve the task.
            2. A agent to answer the Socratic questions.
            3. A final decision agent to provide the final answer based on the answer.
        """
        # Step 1: Generate Socratic questions
        # Instruction for generating Socratic questions
        socratic_question_instruction = (
            f"Task:\n{taskInfo}\n\n"
            "What are some important questions one should ask to better understand and solve this task? "
            "Please generate a list of Socratic questions."
        )
        # Call the llm to generate Socratic questions
        questioning_response = self.llm.call_llm(socratic_question_instruction)

        # Step 2: Answer the Socratic questions
        # Instruction for answering the Socratic questions
        answering_instruction = (
            f"Task:\n{taskInfo}\n\n"
            f"Questions:\n{questioning_response}\n\n"
            "Given the task and these important Socratic questions, please answer each question to help solve the task.\n"
            "Each answer should follow the format:\n"
            "Question: <Insert the question here>\n"
            "Answer: <Insert the answer here>\n\n"
        )
        # Call the llm to generate answers
        answering_response = self.llm.call_llm(answering_instruction)

        # Step 3: use the final decision agent to provide the final answer
        # Instruction for final decision-making based on the answers
        final_decision_instruction = (
            f"Task:\n{taskInfo}\n\n"
            f"Answers:\n{answering_response}\n\n"
            "Given the answers to the Socratic questions, carefully reason over them and provide a final answer to the task."
        )
        # Call the llm to generate the final answer
        final_decision_response = self.llm.call_llm(final_decision_instruction, temperature=0.1)

        return final_decision_response