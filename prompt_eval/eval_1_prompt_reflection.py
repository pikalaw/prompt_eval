from pydantic import BaseModel
from .dataset_loader import Sample
from .gemini import generate_content
from .grader import grade_answers


SOLVE_PROMPT = """Solve the given word problem.
First, find an initial answer with step-by-step reasoning.
Then, critique the answer. Do not try to solve the problem again. Simply check the correctness of each step of the initial answer. 
Lastly, heed your own critique and provide an improved answer.

Respond in the following format:

```
Initial answer: <Your step-by-step reasoning>
#### <Your initial answer in succint form>
Critique: <Check the correctness of each steps in the intial answer>
Revised answer: <Your revised step-by-step reasoning>
#### <Your final answer in succint form>
```

For example, suppose the word problem is this:
```
A train leaves New York for Boston, 200 miles away, at 3:00 PM. Another train leaves Boston for New York at the same time. The first train travels at 60 mph, and the second train travels at 80 mph. At what time do the two trains pass each other?
```

Your reply could be this:
```
Initial answer: The approach speed is 60 + 80 = 140 mph. So, the two trains will meet in 200 / 140 = 1.43 hours. Since the first train left at 3:00 PM, the two trains will meet at 3:00 PM + 1.43 hours = 4:26 PM.
#### 4:00 PM
Critique: The answer is correct, but the reasoning is not clear.
Revised answer: The approach speed is 60 + 80 = 140 mph. So, the two trains will meet in 200 / 140 = 1.43 hours. Since the first train left at 3:00 PM, the two trains will meet at 3:00 PM + 1.43 hours = 4:26 PM.
#### 4:00 PM
```
"""


class Experiment(BaseModel, frozen=True):
    question: str
    human_answer: str
    llm_answer: str
    grade: int


async def eval_1_prompt_reflection(
    model: str,
    sample: Sample,
) -> Experiment:
    model_answer = await generate_content(
        model=model, prompt=SOLVE_PROMPT, input=sample.question
    )
    grade = await grade_answers(
        model=model,
        question=sample.question,
        human_answer=sample.answer,
        model_answer=model_answer,
    )
    return Experiment(
        question=sample.question,
        human_answer=sample.answer,
        llm_answer=model_answer,
        grade=grade,
    )
