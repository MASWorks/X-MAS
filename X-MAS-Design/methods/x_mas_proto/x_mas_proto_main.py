import os
import re
from typing import List, Dict, Any, Set, Tuple

from ..mas_base import MAS
from ..utils import load_config
from .prompt_main import *

# Define the NEWMAS class which inherits from MAS and implements the inference method
class X_MAS_PROTO_MAIN(MAS):
    def __init__(self, general_config, method_config_name = "config_main"):
        super().__init__(general_config)
        self.method_config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", f"{method_config_name}.yaml"))
        
        self.max_revise_rounds = self.method_config['max_revise_rounds']
        self.model_names = self.method_config['model_names']

        self.cnt_d_agents = self.method_config['cnt_d_agents']
        self.cnt_e_agents = self.method_config['cnt_e_agents']
        self.cnt_agents = self.cnt_d_agents + self.cnt_e_agents

        self.history = []
    
    def inference(self, query):
        # init plans to agents
        plans = self.init_plan(query)

        # solve the query
        solutions = self.problem_solving(query, plans)

        # aggregate to get final solution
        solution = self.aggregation(query, solutions)
        # print("\n\n final solution: \n", solution)
        return solution

    def init_plan(self, query: str):
        # Fetch prompts from config.yaml (assumed to be loaded earlier)
        planner_prompt = PLANNER_PROMPT.replace("${query}", query).replace("${cnt_agents}", str(self.cnt_agents))
        planner_response = self.call_llm(planner_prompt, model_name=self.model_names[0])
        # print("\nplanner_prompt:\n", planner_prompt)
        # print("\nplanner_response:\n", planner_response)
        # Extract plans using regex
        try:
            plans = self.extract_plans(planner_response)
            # print("\nplans:\n", plans)
        except:
            print("Extract plans Error!")
        return plans

    def extract_plans(self, response: str):
        """
        Extracts the plans from the model's response using regex.
        Assumes the response is formatted like:
        1.You can solve this problem like this: first, ..., second, ..., finally, ....
        2.There is a idea for how to solve this problem: first, ..., second, ..., finally, ....
        ...
        """
        role_pattern = r"\d+\.\s*([^.]+)"  # extract the content between the number and the period
        
        plans = re.findall(role_pattern, response)
        
        if len(plans) == self.cnt_agents:
            # print("plans:")
            # print(plans)
            return plans
        else:
            raise ValueError(f"wrong cnt_agent, expect {self.cnt_agents} agents while we find {len(plans)} plans.")

    def problem_solving(self, query: str, plans: List[str]):
        consensus_reached = False
        solutions = []
        
        for i in range(self.cnt_agents):
            solver_prompt = SOLVER_PROMPT.replace("${query}", query).replace("${idea}", str(plans[i]))            
            solver_response = self.call_llm(solver_prompt, model_name=self.model_names[1])   
            # print("\nsolver_prompt:\n", solver_prompt)
            # print("\nsolver_response:\n", solver_response)
            if i < self.cnt_d_agents:
                solution = solver_response
            else:
                solution = solver_response
                for j in range(self.max_revise_rounds):
                    evaluator_prompt = EVALUATOR_PROMPT.replace("${query}", query).replace("${solution}", solution)
                    # print("\nevaluator_prompt:\n", evaluator_prompt)
                    evaluator_response = self.call_llm(evaluator_prompt, model_name=self.model_names[2])  
                    # print("\nevaluator_response:\n", evaluator_response)
                    if "The solution is correct" in evaluator_response or "the solution is correct" in evaluator_response:
                        consensus_reached = True
                    elif "The solution is incorrect" in evaluator_response or "the solution is incorrect" in evaluator_response:
                        consensus_reached = False
                    else:
                        print("\n\nThe evaluator didn't response in the prescribed format\n")
                        print("\nThe evaluator response:\n", evaluator_response)
                        consensus_reached = False
                        
                    if consensus_reached:
                        break
                    else:
                        revise_prompt = REVISE_PROMPT.replace("${query}", query).replace("${solution}", str(solver_response)).replace("${evaluation}", str(evaluator_response))
                        revise_response = self.call_llm(revise_prompt, model_name=self.model_names[3])  
                        solution = revise_response
                    
            solutions.append(solution)
            
        return solutions
            

    def aggregation(self, query: str, solutions: List[str]):
        # print("\n\nsolutions before aggregation:\n", solutions)
        solutions_str = ""
        for i, solution in enumerate(solutions):
            solutions_str += f"## [Solution {i+1}]:\n{solution}\n\n"
        aggregator_prompt = AGGREGATOR_PROMPT.replace("${query}", query).replace("${solutions}", solutions_str)
        aggregator_response = self.call_llm(aggregator_prompt, model_name=self.model_names[4])
        # print("\naggregator_prompt:\n", aggregator_prompt)
        # print("\naggregator_response:\n", aggregator_response)
        return aggregator_response