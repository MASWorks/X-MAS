from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)
        
        # Pre-defined prompt template
        
        # Define the prompt template with few-shot examples for Intuitive Reasoning
        self.prompt = """Use symptom, signs, and laboratory disease associations to step by step deduce the correct response. Here are some examples:

Example Question 1:
Shortly after undergoing a bipolar prosthesis for a displaced femoral neck fracture of the left hip acquired after a fall the day before, an 80-year-old \
woman suddenly develops dyspnea. The surgery under general anesthesia with sevoflurane was uneventful, lasting 98 min, during which the patient \
maintained oxygen saturation readings of 100% on 8 l of oxygen. She has a history of hypertension, osteoporosis, and osteoarthritis of her right knee. \
Her medications include ramipril, naproxen, ranitidine, and a multivitamin. She appears cyanotic, drowsy, and is oriented only to person. Her \
temperature is 38.6 °C (101.5 °F), pulse is 135/min, respirations are 36/min, and blood pressure is 155/95 mm Hg. Pulse oximetry on room air shows an \
oxygen saturation of 81%. There are several scattered petechiae on the anterior chest wall. Laboratory studies show a hemoglobin concentration of \
10.5 g/dl, a leukocyte count of 9000/mm3, a platelet count of 145,000/mm3, and a creatine kinase of 190 U/l. An ECG shows sinus tachycardia. What is \
the most likely diagnosis?
Example Rationale 1:
This patient has findings of petechiae, altered mental status, shortness of breath, and recent surgery suggesting a \
diagnosis of fat emboli. The patient most likely has a fat embolism.

Example Question 2:
A 55-year-old man comes to the emergency department because of a dry cough and severe chest pain beginning that morning. Two months ago, he \
was diagnosed with inferior wall myocardial infarction and was treated with stent implantation of the right coronary artery. He has a history of \
hypertension and hypercholesterolemia. His medications include aspirin, clopidogrel, atorvastatin, and enalapril. His temperature is 38.5°C (101.3 °F), \
pulse is 92/min, respirations are 22/min, and blood pressure is 130/80 mm Hg. Cardiac examination shows a high-pitched scratching sound best \
heard while sitting upright and during expiration. The remainder of the examination shows no abnormalities. An ECG shows diffuse ST elevations. \
Serum studies show a troponin I of 0.005 ng/ml (N < 0.01). What is the most likely cause of this patient's symptoms?
Example Rationale 2:
This patient had a recent myocardial infarction with new development of diffuse ST elevations, chest pain, and a high \
pitched scratching murmur which are found in Dressler’s syndrome. This patient likely has Dressler’s Syndrome.

Question:
{}
Your Rationale:
"""

    def forward(self, taskInfo):
        """
        Process a medical diagnostic task using the Intuitive Reasoning Chain-of-Thought (CoT) prompt and few-shot examples.
        """
        # Generate a response by formatting the prompt with the input question.
        answer = self.llm.call_llm(self.prompt.format(taskInfo))
        
        # Return the generated response.
        return answer
