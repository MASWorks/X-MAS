from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving financial problems.
        
        Steps:
            1. A financial analyst agent to analyze the information, perform calculations, and provide an initial answer.
            2. A data extraction critique agent to evaluate the accuracy of data extraction and provide a critique.
            3. A calculation critique agent to evaluate the correctness of the calculations and provide a critique.
            4. A revision agent to refine the initial answer based on feedback from both critique agents.
        """
        # Get the prompt for the financial analyst agent
        fin_analyst_prompt = self.get_fin_analyst_agent_prompt(taskInfo=taskInfo)
        # Call llm to get the analysis response from the LLM
        analysis_response = self.llm.call_llm(fin_analyst_prompt)

        # Generate the prompt for the data extraction critique agent
        extraction_critic_prompt = self.get_extraction_critic_agent_prompt(taskInfo=taskInfo, response=analysis_response)
        # Call llm to get the response from the LLM
        extraction_critic_response = self.llm.call_llm(extraction_critic_prompt)

        # Generate the prompt for the calculation critique agent
        calculation_critic_prompt = self.get_calculation_critic_agent_prompt(taskInfo=taskInfo, response=analysis_response)
        # Call llm to get the response from the LLM
        calculation_critic_response = self.llm.call_llm(calculation_critic_prompt)

        # Generate the prompt for the revision agent using the responses from both critique agents
        revision_agent_prompt = self.get_revision_agent_prompt(
            taskInfo=taskInfo,
            initial_response=analysis_response,
            extraction_critique=extraction_critic_response,
            calculation_critique=calculation_critic_response
        )
        # Call the llm to get the final refined answer
        final_answer = self.llm.call_llm(revision_agent_prompt)

        return final_answer


    def get_fin_analyst_agent_prompt(self, taskInfo):
        """
        Generates the prompt for the financial analyst agent.

        Args:
            taskInfo (str): The financial problem or question to be analyzed.

        Returns:
            str: The prompt for the financial analyst agent.
        """
        prompt = (
            f"Problem:\n{taskInfo}\n\n"
            "You are a financial analysis agent specializing in interpreting earnings reports and financial statements. "
            "Your task is to answer specific financial questions based on the given context from financial reports.\n"
            "When answering questions:\n"
            "Carefully read and analyze the provided financial information.\n"
            "Extract the relevant data points needed to answer the question from the table or text provided.\n"
            "Perform any necessary calculations.\n"
            "Remember to be precise in your calculations and clear in your step-by-step explanation.\n"
            "Maintain a professional and objective tone in your response.\n"
            "Use only the information provided in the context. Do not introduce external information.\n"
            "Provide the answer in the unit specified in the question (million, percentage, or billion).\n"
            "If no unit is specified, use the most appropriate unit based on the context and question.\n"
        )
        return prompt


    def get_extraction_critic_agent_prompt(self, taskInfo, response):
        """
        Generates the prompt for the data extraction critique agent.

        Args:
            taskInfo (str): The financial problem or question to be analyzed.
            response (str): The initial response provided by the financial analyst agent.

        Returns:
            str: The prompt for the data extraction critique agent.
        """
        prompt = (
            f"Problem:\n{taskInfo}\n\n"
            f"Response:\n{response}\n\n"
            "You are a meticulous financial analyst and critic. Your task is to review the response provided by another agent regarding "
            "financial data extraction and provide feedback on its accuracy and completeness. Pay close attention to the following aspects:\n"
            "Question Comprehension: Does the response correctly understand the original question?\n"
            "Data Extraction: Are all relevant numbers accurately extracted from the provided text/tables?\n\n"
            "Focus only on these two aspects. Do not evaluate calculations or provide additional analysis."
        )
        return prompt


    def get_calculation_critic_agent_prompt(self, taskInfo, response):
        """
        Generates the prompt for the calculation critique agent.

        Args:
            taskInfo (str): The financial problem or question to be analyzed.
            response (str): The initial response provided by the financial analyst agent.

        Returns:
            str: The prompt for the calculation critique agent.
        """
        prompt = (
            f"Problem:\n{taskInfo}\n\n"
            f"Response:\n{response}\n\n"
            "You are a meticulous financial analyst and critic. Your task is to review the response provided by another agent regarding "
            "financial calculations and provide feedback on its accuracy and completeness. Pay close attention to the following aspects:\n"
            "Calculation Steps: Confirm that all calculation steps are correct.\n"
            "Calculation Accuracy: Verify the accuracy of all calculations, including intermediate and final results.\n"
            "Unit Consistency: Ensure the final answer’s unit matches what the question requires."
        )
        return prompt


    def get_revision_agent_prompt(self, taskInfo, initial_response, extraction_critique, calculation_critique):
        """
        Generates the prompt for the revision agent.

        Args:
            taskInfo (str): The financial problem or question to be analyzed.
            initial_response (str): The initial response provided by the financial analyst agent.
            extraction_critique (str): The critique provided by the data extraction critique agent.
            calculation_critique (str): The critique provided by the calculation critique agent.

        Returns:
            str: The prompt for the revision agent.
        """
        prompt = (
            f"Problem:\n{taskInfo}\n\n"
            f"Initial response:\n{initial_response}\n\n"
            f"Data Extraction Critique:\n{extraction_critique}\n\n"
            f"Calculation Critique:\n{calculation_critique}\n\n"
            "Based on the critiques provided by the data extraction and calculation agents, your task is to revise the initial response. "
            "Fix any mistakes in data extraction and refine the calculations based on the feedback. Provide a clear and accurate final answer."
        )
        return prompt