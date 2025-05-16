from methods.mas_base import MAS

class CoT(MAS):
    def __init__(self, general_config):
        super().__init__(general_config)
    
    def inference(self, query):
        
        prompt = query + "\n\nLet's think step by step."
        
        response = self.call_llm(prompt=prompt)

        return response