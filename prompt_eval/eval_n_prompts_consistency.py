from pydantic import BaseModel
from .dataset_loader import Sample
from .gemini import generate_content
from .grader import grade_answers


SOLVE_PROMPT = """Solve the given word problem.

Respond in the following format:

```
Solution: <with your step-by-step reasoning>
#### <Your answer in succint form>
```

For example, suppose the word problem is this:
```
A train leaves New York for Boston, 200 miles away, at 3:00 PM. Another train leaves Boston for New York at the same time. The first train travels at 60 mph, and the second train travels at 80 mph. At what time do the two trains pass each other?
```

Your reply could be this:
```
Solution: The approach speed is 60 + 80 = 140 mph. So, the two trains will meet in 200 / 140 = 1.43 hours. Since the first train left at 3:00 PM, the two trains will meet at 3:00 PM + 1.43 hours = 4:26 PM.
#### 4:00 PM
```
"""

COMBINE_PROMPT = """You are given three solution candidates to a question.
Reproduce a new solution from parts of the 3 solutions that are consistent among them.
"""


class Experiment(BaseModel, frozen=True):
    question: str
    human_answer: str
    llm_answer: str
    grade: int


async def eval_n_prompts_consistency(
    model: str,
    sample: Sample,
) -> Experiment:
    candidate_1 = await generate_content(
        model=model, prompt=SOLVE_PROMPT, input=sample.question
    )
    candidate_2 = await generate_content(
        model=model, prompt=SOLVE_PROMPT, input=sample.question
    )
    candidate_3 = await generate_content(
        model=model, prompt=SOLVE_PROMPT, input=sample.question
    )
    model_answer = await generate_content(
        model=model,
        prompt=COMBINE_PROMPT,
        input=(
            f"{sample.question}\n\n"
            f"Candidate 1: {candidate_1}\n\n"
            f"Candidate 2: {candidate_2}\n\n"
            f"Candidate 3: {candidate_3}"
        ),
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
