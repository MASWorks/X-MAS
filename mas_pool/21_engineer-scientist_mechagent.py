from utils import *  
import re 

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving mechanics problem.
        Steps:
            1. A planner to provide step-by-step plan.
            2. For each step, either a engineer or scientist is called to finish the step.
                The engineer generates code and provide the output.
                The scientist focus on providing formulation and theoretical analyses.
            3. An final decision agent to provide the final response.
        """
        # Get the plan steps from the planner
        steps = self.get_plan_steps(taskInfo=taskInfo)

        # Initialize the existing insights to store outputs from all steps
        existing_insights = ""
        index = 1  # Step counter for numbering

        # Process each step in the generated plan
        for step in steps:
            current_step = step["description"]
            if not existing_insights:
                existing_insights = "None"  # Default value if no insights are available
                
            if step["role"] == "Engineer":
                # If the role is Engineer, generate output based on the engineer's prompt
                output = self.get_engineer_output(taskInfo=taskInfo, current_step=current_step, existing_insights=existing_insights)
            else:
                # If the role is Scientist, generate output based on the scientist's prompt
                scientist_prompt = (
                    f"Problem:\n{taskInfo}\n\n"
                    f"Existing insights:\n{existing_insights}\n\n"
                    f"Current step:\n{current_step}\n\n"
                    "A solving plan has been provided for the problem. "
                    "As a Scientist, your task is to focus on the current step and address the mechanics problem.\n"
                    "Existing insights represent the progress made so far based on the previously executed steps of the solving plan. "
                    "Use this context to ensure that your formulation aligns with the results and assumptions established up to this point.\n"
                    "The current step specifies the exact aspect of the problem you need to analyze or refine.\n\n"
                    "You should:\n"
                    "1. Figuring out the geometry of the mesh.\n"
                    "2. Defining the boundary conditions.\n"
                    "3. Establishing the constitutive law of materials and identifying related material properties.\n"
                    "4. Formulating the mechanics problem clearly.\n\n"
                )
                # Call llm to generate a response
                output = self.llm.call_llm(scientist_prompt)
            
            # Append the step's description and output to existing insights
            existing_insights += f"Step {index}:{step['description']}\n{output}\n\n"
            index += 1  # Increment step counter

        # Get the final decision agent's prompt using the accumulated insights
        final_decision_prompt = (
            f"Problem:\n{taskInfo}\n\n"
            f"Existing insights:\n{existing_insights}\n\n"
            "As the Final Decision Agent, your task is to review the problem and the existing insights generated from previous steps. "
            "Based on these insights, provide a comprehensive and concise final answer to the problem. "
            "Your answer should:\n"
            "1. Summarize the key steps and insights.\n"
            "2. Address any unresolved aspects of the problem.\n"
            "3. Clearly state the final conclusions or solutions.\n\n"
            "Ensure that your response is clear, complete, and addresses all aspects of the task."
        )
        # Call llm to generate the final response
        final_response = self.llm.call_llm(final_decision_prompt)
        return final_response

    def get_plan_steps(self, taskInfo, temperature=None, max_retries=3):
        """
        Call the LLM to generate a response containing steps and extract the steps from the response using a retry mechanism.
        If all retry attempts fail, a default set of steps is returned, assigning the given task to both an Engineer and a Scientist.

        Args:
            prompt : str
                The input text prompt to be passed to the llm for generating steps.
            temperature : float, optional
                The temperature setting for the LLM, controlling the randomness of its outputs. 
            max_retries : int, optional
                The maximum number of attempts to extract valid steps from the LLM's response. Defaults to 3.

        Returns:
            list[dict]
                A list of dictionaries where each dictionary contains 'role' and 'description' keys.
                Returns an empty list if no valid steps are extracted after `max_retries` attempts.
        """
        # The prompt for the planner agent
        prompt = (
            f"Problem:\n{taskInfo}\n\n"
            "You are a Planner. Given the problem, your task is to suggest a detailed plan consisting of specific steps. "
            "Each step must be assigned explicitly to either an engineer or a scientist based on their roles:\n\n"
            "- The Engineer writes code and generates outputs.\n"
            "- The Scientist defines the geometry, boundary conditions, material properties, and formulates the mechanics problem.\n\n"
            "Your plan must:\n"
            "1. Break down the problem into clear, executable steps.\n"
            "2. Explicitly assign each step to either the Engineer or the Scientist.\n"
            "3. Ensure the plan is concise and complete to address all aspects of the problem.\n\n"
            "Output format:\n"
            "Step 1: the description of step1 (Scientist)\n"
            "Step 2: the description of step2 (Engineer)\n"
        )
        
        attempts = 0  # Track the number of attempts
        step_pattern = r"Step\s+\d+:\s+(.*?)\s+\((Scientist|Engineer)\)"     # The expected output format pattern
        
        while attempts < max_retries:
            # Generate response using the LLM
            if temperature:
                response = self.llm.call_llm(prompt, temperature=temperature)
            else:
                response = self.llm.call_llm(prompt)
            
            # Use regex to extract steps from the response
            matches = re.findall(step_pattern, response)
            if matches:
                # Convert matches into a list of dictionaries
                steps = [{"role": role, "description": description.strip()} for description, role in matches]
                return steps
            
            # Increment the attempt counter if no valid steps are found
            attempts += 1

        # Return an default setting if no valid steps are extracted after all retries
        return [
            {
                "role": "Engineer",
                "description": taskInfo
            },
            {
                "role": "Scientist",
                "description": taskInfo
            }
        ]

    def get_engineer_output(self, taskInfo, current_step, existing_insights):
        """
        Generates and executes the Engineer's output based on the current step, with a retry mechanism in case the extracted code is invalid.

        Args:
            taskInfo (str): The overall task description.
            current_step (str): The specific step for the Engineer.
            existing_insights (str): The progress and results of previously executed steps.
            max_retries (int, optional): The maximum number of attempts to extract valid code from the LLM's response. Defaults to 3.

        Returns:
            str: The output generated after executing the Engineer's code.
                Return an error message if no valid code is extracted after retries.
        """
        # Get the Engineer's prompt
        # Prompt describing the Engineer's role and expected output format
        prompt = (
            f"Problem:\n{taskInfo}\n\n"
            f"Existing insights:\n{existing_insights}\n\n"
            f"Current step:\n{current_step}\n\n"
            "A solving plan has been provided for the problem. "
            "As an Engineer, your task is to focus on the current step and write Python code to solve the task in the current step. "
            "Existing insights represent the progress made so far based on the previously executed steps of the solving plan. "
            "Use this context to ensure that your code aligns with the results and assumptions established up to this point.\n"
            "The current step specifies the exact task you need to focus on and implement.\n"
            "Requirements:\n"
            "1. Ensure the code is self-contained and ready to execute, including all necessary imports and dependencies.\n"
            "2. Store the final calculation result in a variable named `output`. This variable must contain the final result of the computation.\n"
            "3. For every intermediate result in the calculation process, print the result along with a clear, descriptive message that explains what the intermediate result represents.\n"
            "4. Write only one complete code block per response. The code should not require additional modifications or dependencies.\n"
            "Wrap your final code solution in <Code Solution> and </Code Solution>. For example:\n"
            "<Code Solution>\n"
            "Your function code here\n"
            "</Code Solution>\n"
        )
        # Call `generate_and_extract_code` predefined in utils to generate the response and extract the code
        response, code = generate_and_extract_code(llm=self.llm, prompt=prompt)
        
        if code:  # If valid code is extracted, execute and return the output
            # Call `execute_code` function predefined in utils to execute the code and return the output
            output = execute_code(code)
            # return the combined response and the code execution result
            return f"Solution:\n{response}\n\nCode Execution Result:\n{output}"
        else:
            # If no valid code is extracted after retries, return the response and an error message
            return f"Solution:\n{response}\n\nCode Execution Result:\nInvalid code. No output."