from pydantic import BaseModel
from .dataset_loader import Sample
from .gemini import generate_content
from .grader import grade_answers


SOLVE_PROMPT = """Solve the given word problem.
Find 3 different perspectives to solve the problem.
For each perspective, solve the problem independently.
Then, analyze all 3 solution candidates.
Reproduce a new solution from parts of the 3 solutions that are consistent among them.

Respond in the following format:

```
Solution candidate 1: <Your first independent attempt, step-by-step reasoning>
#### <Your first answer in succint form>
Solution candidate 2: <Your second independent attempt, step-by-step reasoning>
#### <Your second answer in succint form>
Solution candidate 3: <Your third independent attempt, step-by-step reasoning>
#### <Your third answer in succint form>
Final answer: <your step-by-step reasoning from the analysis of the 3 candidates above>
#### <Your final answer in succint form>
```

For example, suppose the word problem is this:
```
A train leaves New York for Boston, 200 miles away, at 3:00 PM. Another train leaves Boston for New York at the same time. The first train travels at 60 mph, and the second train travels at 80 mph. At what time do the two trains pass each other?
```

Your reply could be this:
```
Solution candidate 1: The approach speed is 60 + 80 = 140 mph. So, the two trains will meet in 200 / 140 = 1.43 hours. Since the first train left at 3:00 PM, the two trains will meet at 3:00 PM + 1.43 hours = 4:26 PM.
#### 4:00 PM
Solution candidate 2: The first train will travel 60 mph for 1 hour, and the second train will travel 80 mph for 1 hour. So, the two trains will meet at 4:00 PM.
#### 4:00 PM
Solution candidate 3: The first train will travel 60 mph for 2 hours, and the second train will travel 80 mph for 1.5 hours. So, the two trains will meet at 4:00 PM.
#### 4:00 PM
Final answer: The three solutions are consistent. The answer is 4:00 PM.
#### 4:00 PM
```
"""


class Experiment(BaseModel, frozen=True):
    question: str
    human_answer: str
    llm_answer: str
    grade: int


async def eval_1_prompt_consistency(
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
