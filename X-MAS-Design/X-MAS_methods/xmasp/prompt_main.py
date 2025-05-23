PLANNER_PROMPT = """You are given a [Question]. Your task is to provide ${cnt_agents} different high-level ideas to the question in a list format like:
1.You can solve this problem from the perspective of ...
2.You might solve this problem by ...
...

The question is as follows:
${query}

Now, given the question, provide ${cnt_agents} different high-level ideas to the question directly.
Only respond with abstract ideas rather than detailed solutions. Do not include your reason."""


SOLVER_PROMPT = """You are given a [Question] and an idea for how to solve the question, your task is to solve the problem directly.

The question is as follows:
${query}

The idea for how to solve the question is as follows:
${idea}

Now, given the question and the idea, you should provide the correct solution to the problem step by step with your reasoning."""


EVALUATOR_PROMPT = """You are given a [Question] and a [Solution] to the question. Your task is to provide a final evaluation to the solution by considering the given solution. The question and the solution are as follows:

# [Question]:
${query}

## [Solution]:
${solution}

Now, given the question and solution, evaluate them carefully and directly provide a final evaluation 'The solution is correct' or 'The solution is incorrect' at the end. Be careful not to give evaluations other than 'The solution is correct' or 'The solution is incorrect' at the end."""


REVISE_PROMPT = """You are given a [Question], a [Solution] and an [Evaluation] to the question. Your task is to revise and provide a final complete solution to the question by reasoning over the given solution. The question, the solution and the evaluation are as follows:

# [Question]:
${query}

## [Solution]:
${solution}

### [Evaluation]:
${evaluation}

Now, given the question and solution, reason and revise over them carefully and provide a final complete solution to the question."""



AGGREGATOR_PROMPT = """You are given a [Question] and several [Solution] to the question. Your task is to provide a final complete solution to the question by reasoning over the given solutions. The question and the solutions are as follows:

# [Question]:
${query}

${solutions}

Now, given the question and all the above solutions, reason over them carefully and provide only one final complete solution to the question."""