from pydantic import BaseModel
from .dataset_loader import Sample
from .gemini import generate_content
from .shared import grade_answers


SOLVE_PROMPT = """Solve the given word problem. Respond in the following format:

```
<Your step-by-step reasoning>
#### <Your final answer in succint form>
```

For example, suppose the word problem is this:
```
A train leaves New York for Boston, 200 miles away, at 3:00 PM. Another train leaves Boston for New York at the same time. The first train travels at 60 mph, and the second train travels at 80 mph. At what time do the two trains pass each other?
```

Your reply could be this:
```
The approach speed is 60 + 80 = 140 mph. So, the two trains will meet in 200 / 140 = 1.43 hours. Since the first train left at 3:00 PM, the two trains will meet at 3:00 PM + 1.43 hours = 4:26 PM.
#### 4:00 PM
```
"""


class Experiment(BaseModel, frozen=True):
    question: str
    human_answer: str
    model_answer: str
    grade: int


async def eval_baseline(model: str, sample: Sample) -> Experiment:
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
        model_answer=model_answer,
        grade=grade,
    )
