import os
from ..mas_base import MAS
from ..utils import load_config

class LLMDebate_MATH(MAS):
    def __init__(self, general_config, method_config_name="config_math"):
        super().__init__(general_config)

        self.method_config = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{method_config_name}.yaml"))
        self.agents_num = self.method_config["agents_num"]
        self.rounds_num = self.method_config["rounds_num"]
        self.model_names = self.method_config["model_names"]
    
    def inference(self, query):
        agent_contexts = [[{"role": "user", "content": f"""{query} Make sure to state your answer at the end of the response."""}] for agent in range(self.agents_num)]

        for round in range(self.rounds_num):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = self.construct_message(agent_contexts_other, query, 2*round - 1)
                    agent_context.append(message)

                response = self.call_llm(messages=agent_context, model_name=self.model_names[0])
                agent_context.append({"role": "assistant", "content": response})
        text_answers = []

        for agent_context in agent_contexts:
            text_answer = agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")
            text_answer = self.parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        final_answer = self.most_frequent(text_answers)
        return final_answer
    
    def construct_message(self, agents, question, idx):

        # Use introspection in the case in which there are no other agents.
        if len(agents) == 0:
            return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

        prefix_string = "These are the recent/updated opinions from other agents: "

        for agent in agents:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent response: ```{}```".format(agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response. \n The original math problem is {}.".format(question)
        return {"role": "user", "content": prefix_string}

    def parse_answer(self, sentence):
        parts = sentence.split(" ")

        for part in parts[::-1]:
            try:
                answer = float(part)
                return answer
            except:
                continue

    def most_frequent(self, answers):
        counter = 0
        num = answers[0]

        for i in answers:
            current_frequency = answers.count(i)
            if current_frequency > counter:
                counter = current_frequency
                num = i

        return num