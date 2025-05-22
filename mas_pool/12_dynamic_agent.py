from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving scientific problems.
        
        Steps:
            1. A routing agent selects the most appropriate expert based on the task.
            2. The chosen expert agent generates the solution based on the domain expertise.
        """
        # Step 1: Determine the most suitable expert for the given task
        choice = self.get_routing_choice(taskInfo=taskInfo, temperature=0.3)
        
        # Step 2: Define the role-specific prompts for each expert
        role_descriptions = {
            # Mathematics Expert: Handles problems requiring mathematical modeling and calculations.
            "Mathematics Expert": "You are a Mathematics Expert skilled in solving complex equations, mathematical modeling, and logical problem-solving. Given the problem, provide detailed calculations, step-by-step reasoning, and a precise mathematical solution using advanced mathematical concepts where necessary.",
            
            # Physics Expert: Specializes in mechanics, electromagnetism, and other physical sciences.
            "Physics Expert": "You are a Physics Expert specializing in mechanics, electromagnetism, thermodynamics, and quantum physics. Given the problem, use your expertise to analyze physical systems, apply relevant laws, and explain phenomena with accurate calculations and clear conceptual reasoning.",
            
            # Chemistry Expert: Focuses on chemical processes, reactions, and molecular analysis.
            "Chemistry Expert": "You are a Chemistry Expert proficient in chemical reactions, molecular structures, and laboratory analysis. Provide insights into chemical processes, detailed explanations of reactions, and any necessary calculations for solving chemistry-related problems.",
            
            # Biology Expert: Tackles problems related to biological systems, genetics, and ecology.
            "Biology Expert": "You are a Biology Expert knowledgeable in cellular biology, genetics, ecology, and physiology. Use your expertise to analyze biological systems, explain processes at the molecular and ecological levels, and provide thorough explanations grounded in biological principles.",
            
            # Science Generalist: Integrates knowledge across all fields to provide a holistic solution.
            "Science Generalist": "You are a Science Generalist with a broad understanding across multiple scientific domains, including math, physics, chemistry, and biology. For a given problem, synthesize knowledge from different fields, provide interdisciplinary insights, and ensure a comprehensive and balanced explanation."
        }
        
        # Step 3: Generate the problem-solving prompt based on the selected expert
        # The prompt includes the task description and the corresponding role description
        problem_solving_prompt = (
            f"Problem:\n{taskInfo}\n\n"
            f"{role_descriptions[choice]}\n"
            f"Let's think step by step."
        )
        
        # Step 4: Call the llm to generate the final answer
        final_answer = self.llm.call_llm(problem_solving_prompt)
        
        # Step 5: Return the final solution provided by the selected expert
        return final_answer
    
    def get_routing_choice(self, taskInfo, temperature=None, max_retries=3):
        """
        Determines the most suitable expert for a given task by querying the LLM.

        Args:
            task_description (str): The description of the task to be routed.
            max_retries (int): Maximum number of attempts to resolve the expert choice.

        Returns:
            str: The selected expert's name. Defaults to "Science Generalist" if no valid expert is chosen after retries.
        """
        # List of valid expert roles
        available_experts = [
            "Mathematics Expert",
            "Physics Expert",
            "Chemistry Expert",
            "Biology Expert",
            "Science Generalist"
        ]
        
        # Initialize retry counter
        retry_count = 0

        while retry_count < max_retries:
            # Generate the routing prompt by inserting the task description
            routing_prompt = f"""Task: {taskInfo}

Given the task, please choose an Expert to answer the question. Choose from: \
Mathematics Expert, Physics Expert, Chemistry Expert, Biology Expert, or Science Generalist.

First describe your reasoning and explain your choice. Then provide your final answer. Wrap your final answer with [[]].

For example:
[[Physics Expert]]
"""

            # Call the LLM with the routing prompt
            if temperature:
                routing_response = self.llm.call_llm(routing_prompt, temperature=temperature)
            else:
                routing_response = self.llm.call_llm(routing_prompt)
            
            # Extract the expert choice from the response
            # Expect the expert choice to be wrapped in double square brackets [[ ]]
            expert_choice_matches = re.findall(r'\[\[(.*?)\]\]', routing_response)
            
            if expert_choice_matches:
                selected_expert = expert_choice_matches[0].strip()
                
                # Validate the selected expert
                if selected_expert in available_experts:
                    return selected_expert
            
            # Increment retry counter
            retry_count += 1
        return "Science Generalist"