from utils import *
import re

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        """
        A multi-agent system for solving complex scientific problems.
        
        Steps:
            1. Explain the key concepts and formulas involved in the problem.
            2. Develop a step-by-step strategy to solve the problem.
            3. Solve each step of the problem.
                - Scientist: Provide a detailed explanation for the step.
                - Engineer: Write Python code to implement the solution.
                - Aggregate the solutions and provide a final solution.
            4. Aggregate the solutions of each step and provide a final solution.
        """

        concepts_instruction = f"""You are an expert in solving scientific problems. Please follow the instructions below:
**Problem:**
{taskInfo}

**Instructions:**

Explain the Concepts:

- Identify all key concepts, terms, and principles involved in the problem.
- Provide formulas, laws, or theories involved in the problem.
- Highlight assumptions, constraints, or special conditions relevant to the problem.


Output Requirements:

- Focus exclusively on the **Explain the Concepts**. You don't need to provide a solution.
- Use clear and concise language.
"""
        concepts = self.llm.call_llm(concepts_instruction)
        steps = self.get_strategy(taskInfo, concepts)
        step_solutions = ""
        for i, _ in enumerate(steps):
            solution = self.solve_a_step(taskInfo, concepts, steps, i+1)
            steps[i]["solution"] = solution
            step_solutions += f"<<Step {i+1}>>\n{steps[i]['step_content']}\nSolution: {solution}\n\n"


        final_instruction = f"""**Problem:**
{taskInfo}

**Concepts:**
{concepts}

**Steps:**
{step_solutions}

You are an expert in solving complex scientific problems. The **Problem** is the scientific problem to be solved. The **Concepts** provides an overview of the key concepts and formulas involved in the problem. The **Steps** section includes the solutions to each step of the problem.
Your task is to reason over each step and provide a final solution to the **Problem**.
"""
        final_response = self.llm.call_llm(final_instruction)
        return final_response
    
    def get_strategy(self, taskInfo, concepts):
        """
        Develop a step-by-step strategy to solve the problem.
        Args:
            taskInfo (str): The description of the task.
            concepts (str): The explanation of the key concepts and formulas involved in the problem.
        
        Returns:
            list: A list of steps with their content.
        """
        strategy_instruction = f"""**Problem:**
{taskInfo}

**Concepts:**
{concepts}

You are an expert in solving complex scientific problems. The given **Problem** is challenging and requires a detailed, step-by-step solution. The **Concepts** provides an overview of the key concepts and formulas involved in the problem.

Your task is to develop a clear and comprehensive step-by-step strategy to solve the problem. Each step should be well-defined, detailed, and aim to achieve a specific sub-goal that contributes to solving the overall problem. Follow these guidelines:
 
1. Focus on the main steps. Each step should involve specific calculation or derivation that progresses toward solving the problem.
2. Determine the number of steps based on the complexity of the problem. Don't assign too many steps.
3. Each step should specify the target of the calculation.
4. Use logical sequencing, ensuring each step builds upon the previous ones. 
5. **Only focus on developing the strategy; you don't need to do the calculations of provide the solution to each step.**
6. Start each step with <<Step X>>, where X is the step number. Conclude the solution with <<End>>.

Output Format:
<<Step 1>>
Description of Step 1

<<Step 2>>
Description of Step 2]

...  

<<End>>
"""
        strategy_response = self.llm.call_llm(strategy_instruction)
        def extract_steps(strategy_response):
            pattern = re.compile(
                r'(?i)<<\s*step\s*(\d+)\s*>>\s*(.*?)\s*(?=<<\s*step\s*\d+\s*>>|<<\s*end\s*>>|$)',
                re.DOTALL
            )

            matches = pattern.findall(strategy_response)

            steps = []
            for step_num, step_content in matches:
                steps.append(
                    {
                        "step_num": step_num,
                        "step_content": step_content.strip()
                    }
                )

            return steps
        steps = extract_steps(strategy_response)
        return steps
    
    def solve_a_step(self, taskInfo, concepts, steps, step_num):
        solved_steps = ""
        for i in range(step_num-1):
            solved_steps += f"<<Step {i+1}>>\n{steps[i]['step_content']}\nSolution: {steps[i]['solution']}\n\n"
        if not solved_steps:
            solved_steps = "No previous steps. This is the first step."

        scientist_instruction = f"""**Problem:**
{taskInfo}

**Concepts:**
{concepts}

**Solved Steps:**
{solved_steps}

**Current Step:**
<<Step {step_num}>>
{steps[step_num-1]['step_content']}

You are an expert in solving complex scientific problems.
The **Problem** is the scientific problem to be solved. The **Concepts** provides an overview of the key concepts and formulas involved in the problem. The **Solved Steps** section includes the solutions to the previous steps.
Your task is to focus exclusively on solving the **Current Step**. Follow these guidelines:

1. Use results or insights from the Solved Steps if necessary to support your solution.
2. Provide a detailed and comprehensive explanation for the Current Step, including any relevant calculations, formulas, or theoretical reasoning.
3. Do not extend or speculate on future steps; your focus is strictly on addressing the Current Step."""

        scientist_response = self.llm.call_llm(scientist_instruction)

        engineer_instruction = f"""**Problem:**
{taskInfo}

**Concepts:**
{concepts}

**Solved Steps:**
{solved_steps}

**Current Step:**
<<Step {step_num}>>
{steps[step_num-1]['step_content']}

You are an expert in solving complex scientific problems by writing code.
The **Problem** is the scientific problem to be solved. The **Concepts** provides an overview of the key concepts and formulas involved in the problem. The **Solved Steps** section includes the solutions to the previous steps.
Your task is to focus exclusively on solving the **Current Step**. Follow these guidelines:

1. Write Python code to implement the solution for the Current Step.
2. Include detailed comments and explanations within the code to clarify the implementation. Print relevant intermediate results if necessary.
3. Wrap your code solution in <Code Solution> and </Code Solution> tags.
4. Ensure the final result is stored in a variable named `output`. This variable must be defined at the global scope and contain the final computation result.

Output Format:
<Code Solution>
Your code here
</Code Solution>
"""
        engineer_response, code = generate_and_extract_code(llm=self.llm, prompt=engineer_instruction)
        execution_result = execute_code(code)
        engineer_response += f"\n\nCode Execution Result:\n{execution_result}"

        aggregate_instruction = f"""**Problem:**
{taskInfo}

**Concepts:**
{concepts}

**Solved Steps:**
{solved_steps}

**Current Step:**
<<Step {step_num}>>
{steps[step_num-1]['step_content']}

**Scientist's Solution:**
{scientist_response}

**Engineer's Solution:**
{engineer_response}

The **Problem** is the scientific problem to be solved. The **Concepts** provides an overview of the key concepts and formulas involved in the problem. The **Solved Steps** section includes the solutions to the previous steps.
The **Scientist's Solution** is the solution for the **Current Step** provided by the scientist. The **Engineer's Solution** is the Python code implementation for the **Current Step** provided by the engineer.
Your task is to reason over both solutions, identify the correct and reliable parts of each solution, and provide a brief final solution for the **Current Step**. Focus on the current step only.
"""
        aggregate_response = self.llm.call_llm(aggregate_instruction)
        return aggregate_response