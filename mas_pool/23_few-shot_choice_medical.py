from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)
        
        # Pre-defined prompt template
        
        # Define a template for few-shot prompts tailored to answering medical multiple-choice questions
        self.medical_few_shot_prompt = """Instructions: The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, \
starting by summarizing the available information. Output a single option from the four options as the final answer.

Question: The energy for all forms of muscle contraction is provided by:
(A) ATP. (B) ADP. (C) phosphocreatine. (D) oxidative phosphorylation.
Explanation: The sole fuel for muscle contraction is adenosine triphosphate (ATP). During near maximal intense exercise \
the muscle store of ATP will be depleted in less than one second. Therefore, to maintain normal contractile function ATP \
must be continually resynthesized. These pathways include phosphocreatine and muscle glycogen breakdown, thus enabling \
substrate-level phosphorylation (‘anaerobic’) and oxidative phosphorylation by using reducing equivalents from carbohydrate \
and fat metabolism (‘aerobic’).
Answer: (A)

Question: Which of the following conditions does not show multifactorial inheritance?
(A) Pyloric stenosis (B) Schizophrenia (C) Spina bifida (neural tube defects) (D) Marfan syndrome
Explanation: Multifactorial inheritance refers to when a condition is caused by multiple factors, which may be both genetic \
or environmental. Marfan is an autosomal dominant trait. It is caused by mutations in the FBN1 gene, which encodes a \
protein called fibrillin-1. Hence, Marfan syndrome is not an example of multifactorial inheritance.
Answer: (D)

Question: What is the embryological origin of the hyoid bone?
(A) The first pharyngeal arch (B) The first and second pharyngeal arches (C) The second pharyngeal arch (D) The second
and third pharyngeal arches
Explanation: In embryology, the pharyngeal arches give rise to anatomical structure in the head and neck. The hyoid bone, \
a small bone in the midline of the neck anteriorly, is derived from the second and third pharyngeal arches.
Answer: (D)

Question: In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming \
the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry \
the b allele but are not expected to develop the cancer?
(A) 1/400 (B) 19/400 (C) 20/400 (D) 38/400
Explanation: The expected proportion of individuals who carry the b allele but are not expected to develop the cancer \
equals to the frequency of heterozygous allele in the given population. According to the Hardy-Weinberg equation p∧2 + \
2pq + q∧2 = 1, where p is the frequency of dominant allele frequency, q is the frequency of recessive allele frequency, p∧2 is \
the frequency of the homozygous dominant allele, q∧2 is the frequency of the recessive allele, and 2pq is the frequency of \
the heterozygous allele. Given that q∧2=1/400, hence, q=0.05 and p=1-q=0.95. The frequency of the heterozygous allele is \
2pq=2*0.05*0.95=38/400.
Answer: (D)

Question: {}
"""

    def forward(self, taskInfo):
        """
        Process a multiple-choice medical question and generate an answer using a language model.
        In-context learning provides the model with few-shot examples to establish context, guiding it to generate accurate and contextually appropriate answers for similar tasks.
        """
        # Format the question into the medical few-shot prompt
        prompt = self.medical_few_shot_prompt.format(taskInfo)

        # Use the language model to process the prompt and get the answer
        answer = self.llm.call_llm(prompt)
        
        # Return the final solution
        return answer