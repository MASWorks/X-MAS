from utils import *
import re

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    # Main Function
    def forward(self, taskInfo):
        """
        A Multi-Agent System for Solving Physics Problems

        Steps:
            1. Use a categorization agent to determine which physics domains are relevant to the question.
            2. Utilize domain-specific experts to provide detailed analysis for the question.
            3. Synthesize the expert analyses into a single report using a synthesis agent.
            4. Generate the final answer based on the final synthesized report.
        """
        # Step 1: Determine relevant physics domains for the task
        domain_categorization_prompt = self.get_question_domains_prompt(taskInfo)
        # prompt the llm with domain categorization prompt and extract relevant domains
        relevant_domains = self.get_domains(domain_categorization_prompt)

        # Step 2: Gather analysis from domain-specific experts
        analysis_prompts = [self.get_question_analysis_prompt(taskInfo, domain) for domain in relevant_domains]  # get the prompts for analysis
        # Call the llm to get response for expert analyses
        expert_analyses = [self.llm.call_llm(prompt) for prompt in analysis_prompts]

        # Step 3: Synthesize a comprehensive report from expert analyses
        synthesis_prompt = self.get_synthesized_report_prompt(expert_analyses)
        # Call the llm to get the synthesized report
        synthesized_report = self.llm.call_llm(synthesis_prompt)

        # Step 4: Generate the final answer based on the final report
        final_decision_prompt = self.get_final_answer_prompt(problem=taskInfo, syn_report=synthesized_report)
        # Call the llm to get the final answer
        final_answer = self.llm.call_llm(final_decision_prompt)

        return final_answer

    # Supporting Functions

    def get_question_domains_prompt(self, question):
        """
        Generate a prompt to categorize a physical scenario into relevant subfields of physics.

        Args:
            question (str): The physical scenario or question.

        Returns:
            str: A formatted prompt for the LLM.
        """
        NUM_QD = 5  # Number of physics subfields to categorize into
        question_domain_format = "Physics Field: " + " | ".join(["Field" + str(i+1) for i in range(NUM_QD)])  # output format
        question_domain_prompt = (
            "You are a physics expert who specializes in categorizing a specific physical problem into specific areas of physics. "
            "You need to complete the following steps:\n"
            f"1. Carefully read the physical problem presented in the question: '''{question}'''. \n"
            "2. Based on the physical problem, classify the question into five different subfields of physics. \n"
            f"3. You should output in exactly the same format as:\n'''\n{question_domain_format}\n'''."
        )
        return question_domain_prompt

    def get_domains(self, prompt, temperature=None, max_retries=3):
        """
        Query the LLM for physics fields based on a prompt, with retry logic.

        Args:
            prompt (str): The input prompt to send to the language model.
            temperature (float, optional): Sampling temperature for the LLM, controlling the randomness of the output. 
                                            Defaults to None.
            max_retries (int, optional): Maximum number of attempts to query the LLM in case of unsuccessful responses. 
                                            Defaults to 3.

        Returns:
            list: A list of physics fields extracted from the LLM's response. If no valid response is obtained after all retries, a default list of common physics fields is returned.
        """
        attempts = 0  # Track the number of attempts

        while attempts < max_retries:
            # Generate response using the LLM
            if temperature:
                response = self.llm.call_llm(prompt, temperature=temperature)
            else:
                response = self.llm.call_llm(prompt)

            # Search the pattern in the response and return a list if found
            match = re.search(r'Physics Field:\s*(.+)', response)
            if match:
                fields = match.group(1).split('|')
                return [field.strip() for field in fields]
            
            attempts += 1  # Increment attempts and retry if no valid fields are detected
        
        # Return the default general physics fields
        return ["Classical Mechanics", "Quantum Mechanics", "Thermodynamics", "Electromagnetism", "Optics"]

    def get_question_analysis_prompt(self, question, domain):
        """
        Generate a prompt to request domain-specific analysis for a physical scenario.

        Args:
            question (str): The physical scenario or question.
            domain (str): The domain of the expert providing the analysis.

        Returns:
            str: A formatted prompt for the LLM.
        """
        return (
            f"You are a physics expert in the domain of {domain}. "
            "From your area of specialization, you will analyze and interpret the physical problem or scenario presented. "
            f"Please meticulously examine the physical scenario outlined in this question: '''{question}'''.\n"
            "Drawing upon your physics expertise, interpret the problem being depicted. "
            "Subsequently, identify and highlight the aspects of the issue that you find most significant or noteworthy."
        )

    def get_synthesized_report_prompt(self, question_analyses):
        """
        Generate a prompt to synthesize analyses from multiple experts into a single report.

        Args:
            question_analyses (list): A list of analyses from domain-specific experts.

        Returns:
            str: A formatted prompt for the LLM.
        """
        question_analyses_text = "\n".join([f"Report {i + 1}:\n{analysis}" for i, analysis in enumerate(question_analyses)])
        syn_report_format = "Key Knowledge: [extracted key knowledge] \nTotal Analysis: [synthesized analysis] \n"
        return (
            "You are a physics decision maker who excels at summarizing and synthesizing reports from various domain experts. "
            "Here are the reports:\n"
            f"{question_analyses_text}\n\n"
            "Steps:\n"
            "1. Extract key knowledge from the reports.\n"
            "2. Derive a comprehensive analysis based on the extracted knowledge.\n"
            "3. Produce a refined synthesized report in the following format:\n"
            f"'''\n{syn_report_format}'''."
        )

    def get_final_answer_prompt(self, problem, syn_report):
        """
        Generate a prompt to provide the final decision based on the synthesized report.

        Args:
            problem (str): The original problem or question.
            syn_report (str): The final synthesized report.

        Returns:
            str: A formatted prompt for the LLM.
        """
        return (
            f"Problem:\n{problem}\n\n"
            f"Synthesized Report:\n{syn_report}\n\n"
            "You are a final decision expert. Based on the report, provide the final answer to the problem."
        )