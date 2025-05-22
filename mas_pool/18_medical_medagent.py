from utils import *
import re

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    # Main Function
    def forward(self, taskInfo):
        """
        A multi-agent System for solving medical problems

        Steps:
            1. Use a categorization agent to determine which medical domains are relevant to the question.
            2. Utilize domain-specific experts to provide detailed analysis for the question.
            3. Synthesize the expert analyses into a single report using a synthesis agent.
            4. Iteratively improve the synthesized report based on feedback from domain experts who disagree.
            5. Generate the final answer based on the final synthesized report.
        """
        # Step 1: Determine relevant medical domains for the task
        domain_categorization_prompt = self.get_question_domains_prompt(taskInfo)
        # prompt the llm with domain categorization prompt and extract relevant domains
        relevant_domains = self.get_domains(domain_categorization_prompt)

        # Step 2: Gather analysis from domain-specific experts
        analysis_prompts = [self.get_question_analysis_prompt(taskInfo, domain) for domain in relevant_domains] # get the prompts for analysis
        # Call the llm to get response for expert analyses
        expert_analyses = [self.llm.call_llm(prompt) for prompt in analysis_prompts]  

        # Step 3: Synthesize a comprehensive report from expert analyses
        synthesis_prompt = self.get_synthesized_report_prompt(expert_analyses)
        # Call the llm to get the synthesized report
        synthesized_report = self.llm.call_llm(synthesis_prompt)

        # Step 4: Iteratively improve the synthesized report through feedback
        max_iterations = 3  # Maximum number of iterative optimization rounds
        for _ in range(max_iterations):
            # Collect feedback from experts on the synthesized report
            consensus_prompts = [self.get_consensus_prompt(domain, syn_report=synthesized_report) for domain in relevant_domains]
            # Call the llm with the consensus prompt and extract the result
            consensus_results = [self.get_consensus_result(prompt) for prompt in consensus_prompts]

            # Identify experts who disagree with the report
            disagreeing_domains = [domain for domain, decision in zip(relevant_domains, consensus_results) if not decision]
            if not disagreeing_domains:  # If all experts agree, stop iteration
                break

            # Gather revision suggestions from disagreeing experts
            revision_prompts = [self.get_consensus_opinion_prompt(domain, syn_report=synthesized_report) for domain in disagreeing_domains]
            # Call the llm to get revision suggestions
            revision_suggestions = [self.llm.call_llm(prompt) for prompt in revision_prompts]
            # Format the suggestions into a dictionary
            revision_advice = {
                domain: suggestion for domain, suggestion in zip(disagreeing_domains, revision_suggestions)
            }

            # Update the synthesized report based on revisions
            synthesized_report = self.llm.call_llm(self.get_revision_prompt(syn_report=synthesized_report, revision_advice=revision_advice))

            # Update relevant domains to focus on remaining disagreements for iterations
            relevant_domains = disagreeing_domains

        # Step 5: Generate the final answer based on the final report
        final_decision_prompt = self.get_final_answer_prompt(problem=taskInfo, syn_report=synthesized_report)
        # Call the llm to get the final answer
        final_answer = self.llm.call_llm(final_decision_prompt)

        return final_answer


    # Supporting Functions

    def get_question_domains_prompt(self, question):
        """
        Generate a prompt to categorize a medical scenario into relevant subfields of medicine.

        Args:
            question (str): The medical scenario or question.

        Returns:
            str: A formatted prompt for the LLM.
        """
        NUM_QD = 5  # Number of medical subfields to categorize into
        question_domain_format = "Medical Field: " + " | ".join(["Field" + str(i+1) for i in range(NUM_QD)]) # output format
        question_domain_prompt = (
            "You are a medical expert who specializes in categorizing a specific medical scenario into specific areas of medicine. "
            "You need to complete the following steps:\n"
            f"1. Carefully read the medical scenario presented in the question: '''{question}'''. \n"
            "2. Based on the medical scenario, classify the question into five different subfields of medicine. \n"
            f"3. You should output in exactly the same format as:\n'''\n{question_domain_format}\n'''."
        )
        return question_domain_prompt

    def get_domains(self, prompt, temperature=None, max_retries=3):
        """
        Query the LLM for medical fields based on a prompt, with retry logic.

        Args:
            prompt (str): The input prompt to send to the language model.
            temperature (float, optional): Sampling temperature for the LLM, controlling the randomness of the output. 
                                            Defaults to None.
            max_retries (int, optional): Maximum number of attempts to query the LLM in case of unsuccessful responses. 
                                        Defaults to 3.

        Returns:
            list: A list of medical fields extracted from the LLM's response. If no valid response is obtained after all retries, a default list of common medical fields is returned.
        """
        attempts = 0   # Track the number of attempts

        while attempts < max_retries:
            # Generate response using the LLM
            if temperature:
                response = self.llm.call_llm(prompt, temperature=temperature)
            else:
                response = self.llm.call_llm(prompt)

            # Search the pattern in the response and return a list if found
            match = re.search(r'Medical Field:\s*(.+)', response)
            if match:
                fields = match.group(1).split('|')
                return [field.strip() for field in fields]
            
            attempts += 1  # Increment attempts and retry if no valid fields are detected
        
        # Return the default general medical fields
        return ["Internal Medicine", "Surgery", "Orthopedics", "Pediatrics", "Neurology"]


    def get_question_analysis_prompt(self, question, domain):
        """
        Generate a prompt to request domain-specific analysis for a medical scenario.

        Args:
            question (str): The medical scenario or question.
            domain (str): The domain of the expert providing the analysis.

        Returns:
            str: A formatted prompt for the LLM.
        """
        return (
            f"You are a medical expert in the domain of {domain}. "
            "From your area of specialization, you will scrutinize and diagnose the symptoms presented by patients in specific medical scenarios. "
            f"Please meticulously examine the medical scenario outlined in this question: '''{question}'''.\n"
            "Drawing upon your medical expertise, interpret the condition being depicted. "
            "Subsequently, identify and highlight the aspects of the issue that you find most alarming or noteworthy."
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
            "You are a medical decision maker who excels at summarizing and synthesizing reports from various domain experts. "
            "Here are the reports:\n"
            f"{question_analyses_text}\n\n"
            "Steps:\n"
            "1. Extract key knowledge from the reports.\n"
            "2. Derive a comprehensive analysis based on the extracted knowledge.\n"
            "3. Produce a refined synthesized report in the following format:\n"
            f"'''\n{syn_report_format}'''."
        )


    def get_consensus_prompt(self, domain, syn_report):
        """
        Generate a prompt to ask a domain expert whether they agree with a synthesized report.

        Args:
            domain (str): The domain of the expert.
            syn_report (str): The synthesized report.

        Returns:
            str: A formatted prompt for the LLM.
        """
        return (
            f"You are a medical expert specialized in the {domain} domain. "
            f"Here is a medical report:\n{syn_report}\n\n"
            "Please read the report and decide whether your opinions are consistent with this report. "
            "Respond only with: [YES or NO]. Wrap your output with [].\n"
            "For example:\n[YES]\n"
        )

    def get_consensus_result(self, prompt, temperature=None, max_retries=3):
        """
        Call the llm to generate a response to the given prompt and extracts a decision from the response in the form of [YES] or [NO]. 
        If multiple attempts fail to extract a valid decision, the function defaults to returning `True`.

        Args:
            prompt : str
                The input text prompt to be passed to the language model (LLM) for generating a decision.
            temperature : float, optional
                The temperature setting for the LLM, controlling the randomness of its outputs. 
            max_retries : int, optional
                The maximum number of attempts to extract a valid decision from the LLM's response. Defaults to 3.

        Returns:
            bool
                Returns `True` if the extracted decision is [YES], `False` if the extracted decision is [NO], 
                or defaults to `True` if no valid decision is extracted after `max_retries` attempts.
        """
        attempts = 0  # Track the number of attempts
        while attempts < max_retries:
            # Generate response using the LLM
            if temperature:
                response = self.llm.call_llm(prompt, temperature=temperature)
            else:
                response = self.llm.call_llm(prompt)

            # Use regular expression to search for [YES] or [NO] in the response
            match = re.search(r'\[(YES|NO)\]', response)
            if match:
                # Return True if [YES] is found, False if [NO] is found
                if match.group(1) == "YES":
                    return True
                if match.group(1) == "NO":
                    return False

            # Increment the attempt counter if no valid decision is found
            attempts += 1

        # Default to True if no valid response is extracted after all retries
        return True

    def extract_decision(self, input_string):
        """
        Extract a decision (YES or NO) from the LLM response.

        Args:
            input_string (str): The response string.

        Returns:
            bool: True for 'YES', False for 'NO', None if invalid format.
        """
        match = re.search(r'\[(YES|NO)\]', input_string)
        return True if match and match.group(1) == "YES" else False if match else None


    def get_consensus_opinion_prompt(self, domain, syn_report):
        """
        Generate a prompt to request revision suggestions from a disagreeing expert.

        Args:
            domain (str): The domain of the expert.
            syn_report (str): The synthesized report.

        Returns:
            str: A formatted prompt for the LLM.
        """
        return (
            f"Here is a medical report:\n{syn_report}\n\n"
            f"As a medical expert specialized in {domain}, please provide detailed revisions to improve this report. "
            "Output in the format: '''Revisions: [proposed revision advice] '''"
        )


    def get_revision_prompt(self, syn_report, revision_advice):
        """
        Generate a prompt to revise the synthesized report based on expert feedback.

        Args:
            syn_report (str): The original synthesized report.
            revision_advice (dict): A dictionary of revision advice from experts.

        Returns:
            str: A formatted prompt for the LLM.
        """
        revisions_text = "\n\n".join([f"Advice from {domain}:\n{advice}" for domain, advice in revision_advice.items()])
        return (
            f"Original report:\n{syn_report}\n\n"
            f"Revisions:\n{revisions_text}\n\n"
            "Based on these revisions, produce a refined report in the format:\n"
            "'''Total Analysis: [revised analysis] '''"
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