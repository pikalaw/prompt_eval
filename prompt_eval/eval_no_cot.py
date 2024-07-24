from pydantic import BaseModel
from .dataset_loader import Sample
from .gemini import generate_content
from .grader import grade_answers


SOLVE_PROMPT = """Solve the given word problem. Respond in the following format:

```
<Your final answer in succint form>
```

For example, suppose the word problem is this:
```
A train leaves New York for Boston, 200 miles away, at 3:00 PM. Another train leaves Boston for New York at the same time. The first train travels at 60 mph, and the second train travels at 80 mph. At what time do the two trains pass each other?
```

Your reply could be this:
```
4:00 PM
```
"""


class Experiment(BaseModel, frozen=True):
    question: str
    human_answer: str
    llm_answer: str
    grade: int


async def eval_no_cot(model: str, sample: Sample) -> Experiment:
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
