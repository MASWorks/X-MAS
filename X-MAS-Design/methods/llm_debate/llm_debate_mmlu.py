import os
import re
from ..mas_base import MAS
from ..utils import load_config

class LLMDebate_MMLU(MAS):
    def __init__(self, general_config, method_config_name="config_mmlu"):
        super().__init__(general_config)

        self.method_config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{method_config_name}.yaml"))
        self.agents_num = self.method_config["agents_num"]
        self.rounds_num = self.method_config["rounds_num"]
        self.model_names = self.method_config["model_names"]
    
    def inference(self, query):
        agent_contexts = [[{"role": "user", "content": f"""Can you answer the following question as accurately as possible? {query} Explain your answer, putting the answer in the form (X) at the end of your response."""}] for agent in range(self.agents_num)]

        for round in range(self.rounds_num):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = self.construct_message(agent_contexts_other, query, 2*round - 1)
                    agent_context.append(message)

                response = self.call_llm(messages=agent_context, model_name=self.model_names[0])
                agent_context.append({"role": "assistant", "content": response})
        
        pred_answers = []

        for agent_context in agent_contexts:
            text_answer = agent_context[-1]['content']
            pred_answer = self.parse_answer(text_answer)

            if pred_answer is None:
                pred_answer = self.solve_math_problems(text_answer)

            if pred_answer is None:
                continue
            pred_answers.append(pred_answer)

        final_answer = self.most_frequent(pred_answers)
        return final_answer
    
    def construct_message(self, agents, question, idx):
        if len(agents) == 0:
            return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

        prefix_string = "These are the solutions to the problem from other agents: "

        for agent in agents:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent solution: ```{}```".format(agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response. \n The original problem is {}.""".format(question)
        return {"role": "user", "content": prefix_string}

    def parse_answer(self, input_str):
        pattern = r'\((\w)\)'
        matches = re.findall(pattern, input_str)

        solution = None

        for match_str in matches[::-1]:
            solution = match_str.upper()
            if solution:
                break

        return solution

    def solve_math_problems(self, input_str):
        pattern = r"\d+\.?\d*"

        matches = re.findall(pattern, input_str)
        if matches:
            return matches[-1]

        return None

    def most_frequent(self, answers):
        counter = 0
        num = answers[0]

        for i in answers:
            current_frequency = answers.count(i)
            if current_frequency > counter:
                counter = current_frequency
                num = i

        return num