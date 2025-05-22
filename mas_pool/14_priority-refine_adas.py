from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for reading comprehension and passage understanding.
        
        Steps:
            1. Utilize three specialized agents to analyze the passage and provide insights from different perspectives:
                - Numerical reasoning agent for quantitative insights.
                - Linguistic analysis agent for language patterns and key phrases.
                - Contextual understanding agent for causal relationships and contextual insights.
            2. Employ an attention agent to evaluate and prioritize the insights based on their importance and need for refinement.
                Iteratively refine the prioritized insights by redirecting tasks to the corresponding specialized agents.
                Dynamically update priorities after each refinement iteration to ensure focus on unresolved issues.
            3. Use a synthesis agent to combine all refined insights into a comprehensive and coherent final answer.
        """

        # Step 1: Instructions for generating initial insights
        numerical_insight_prompt = (
            f"Task:\n{taskInfo}\n\n"
            "Analyze the passage and question for any numerical reasoning required and provide your insights."
        )
        linguistic_insight_prompt = (
            f"Task:\n{taskInfo}\n\n"
            "Analyze the passage and question for linguistic patterns and provide your insights."
        )
        contextual_insight_prompt = (
            f"Task:\n{taskInfo}\n\n"
            "Analyze the passage and question for contextual understanding and provide your insights."
        )

        # Generate initial insights from specialized agents
        numerical_insight = self.llm.call_llm(numerical_insight_prompt)
        linguistic_insight = self.llm.call_llm(linguistic_insight_prompt)
        contextual_insight = self.llm.call_llm(contextual_insight_prompt)

        # Store insights in a list for easier management and access
        insights = [numerical_insight, linguistic_insight, contextual_insight]

        # Step 2: Instructions for refining insights
        numerical_refine_prompt = (
            "Task:\n{taskInfo}\n\n"
            "Previous Answer:\n{answer}\n\n"
            "Given the task and the previous answer, further improve and refine the answer. Focus on numerical reasoning."
        )
        linguistic_refine_prompt = (
            "Task:\n{taskInfo}\n\n"
            "Previous Answer:\n{answer}\n\n"
            "Given the task and the previous answer, further improve and refine the answer. Focus on linguistic patterns."
        )
        contextual_refine_prompt = (
            "Task:\n{taskInfo}\n\n"
            "Previous Answer:\n{answer}\n\n"
            "Given the task and the previous answer, further improve and refine the answer. Focus on contextual understanding."
        )

        # Map agent roles to their respective insight index and refinement instruction
        agent_prompts = {
            "numerical": (0, numerical_refine_prompt),
            "linguistic": (1, linguistic_refine_prompt),
            "contextual": (2, contextual_refine_prompt)
        }

        # Step 3: Attention Agent prompt to evaluate and prioritize insights
        attention_prompt_template = (
            "Task:\n{taskInfo}\n\n"
            "Numerical Insight:\n{numerical}\n\n"
            "Linguistic Insight:\n{linguistic}\n\n"
            "Contextual Insight:\n{contextual}\n\n"
            "Evaluate the importance and completeness of each component's insights. "
            "Decide which component most needs refinement, and enclose your choice in [[ ]]. "
            "Options: numerical, linguistic, contextual, satisfactory (if all components are already complete).\n"
            "For example:\n[[linguistic]]\n\n"
        )

        # Step 4: Iterative refinement process
        for _ in range(3):  # Maximum of 3 refinement iterations
            # Format the attention prompt with current insights
            attention_prompt = attention_prompt_template.format(
                taskInfo=taskInfo,
                numerical=insights[0],
                linguistic=insights[1],
                contextual=insights[2]
            )
            # prompt with the attention prompt and extract the choice, a low temperature for accuracy
            refinement_choice = self.get_priority_choice(attention_prompt, temperature=0.3)

            if refinement_choice == "satisfactory":  # If no further refinement is needed
                break
            # Identify the corresponding insight index and refinement instruction
            insight_index, refine_prompt_template = agent_prompts[refinement_choice]
            # Format the refinement prompt with the current answer
            refinement_prompt = refine_prompt_template.format(
                taskInfo=taskInfo,
                answer=insights[insight_index]
            )
            # Call the specialized agent for refinement
            refined_insight = self.llm.call_llm(refinement_prompt)
            # Update the insight with the refined version
            insights[insight_index] = refined_insight

        # Step 5: Synthesis Agent to generate the final answer
        synthesis_prompt = (
            f"Task:\n{taskInfo}\n\n"
            f"Numerical Insight:\n{insights[0]}\n\n"
            f"Linguistic Insight:\n{insights[1]}\n\n"
            f"Contextual Insight:\n{insights[2]}\n\n"
            "Combine the insights from other agents to form a thorough understanding and provide your final answer."
        )
        # Call the llm to generate the final answer
        final_answer = self.llm.call_llm(synthesis_prompt)

        return final_answer

    import re
    def get_priority_choice(self, prompt, temperature=None, max_retries=3):
        """
        Generate a response from the LLM and extract content enclosed in double square brackets ([[ ]]) \
        that matches predefined valid choices.

        Args:
            prompt (str): The instruction or query to send to the LLM for generating a response.
            temperature (float, optional): Sampling temperature for the LLM, controlling the 
                randomness or diversity of the response. If not specified, a default value is used.
            max_retries (int): Maximum number of attempts to extract valid content from the LLM response. 
                Default is 3.

        Returns:
            str: A valid choice extracted from the LLM response that is enclosed in [[ ]]. 
                If no valid choice is found after the maximum retries, the function defaults to "satisfactory".
        """
        attempts = 0  # Track the number of retry attempts
        available_choices = [
            "numerical",
            "linguistic",
            "contextual",
            "satisfactory"
        ]
        while attempts < max_retries:
            # Call the LLM to generate the response with the given prompt and temperature
            if temperature:
                response = self.llm.call_llm(prompt, temperature=temperature)
            else:
                response = self.llm.call_llm(prompt)

            # Pattern to match content enclosed in double square brackets ([[ ]])
            pattern = r'\[\[(.*?)\]\]'
            match = re.search(pattern, response)

            # Extract and validate the matched content
            if match:
                choice = match.group(1).strip().lower()
                if choice in available_choices:
                    return choice

            # Increment the retry counter if no valid match is found
            attempts += 1

        # Return "satisfactory" if all retries are exhausted without finding a valid match
        return "satisfactory"