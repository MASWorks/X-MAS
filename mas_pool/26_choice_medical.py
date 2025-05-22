from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

        # Pre-defined prompt template
        
        # Define the prompt template for the LLM to answer medical multiple-choice questions.
        self.prompt = """You are a medical expert. Answer the following multiple choice question from the medical domain based on following instructions.

1. Output a brief explanation summarizing and providing context to the question under the heading 'Explanation' in about 5 sentences.
2. Select the correct option and provide the correct option under the heading 'Answer'.
3. Always select one of the four options provided as the answer.
4. If the options are ambiguous or the question does not have enough context, select the one that best answers the question.

Question: {question}

"""
    def forward(self, taskInfo):
        """
        Use the predefined prompt to guide the LLM in accurately answering a multiple-choice question in the medical domain.
        """        
        # Generate a response by formatting the prompt with the input question.
        answer = self.llm.call_llm(self.prompt.format(question=taskInfo))
        
        # Return the final solution
        return answer