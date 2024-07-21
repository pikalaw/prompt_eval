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

REFLECT_PROMPT = """Critique the answer for a question. Analyze its correctness, clarity, and completeness.

The format of the input looks like this:

```
Question: <The question>
Answer: <The reasoning for the answer>
#### <The final answer>
```
"""

REVISE_PROMPT = """Given a question, an answer, and a critique on the asnwer, revise the answer.
Heed the critique and provide an improved answer.

The format of the input looks like this:
```
Question: <The question>
Answer: <The reasoning for the answer>
#### <The final answer>
Critique: <The critique>
```

Your reply should be in the following format:
```
<Your revised reasoning>
#### <Your revised final answer>
```
"""


class Experiment(BaseModel, frozen=True):
    question: str
    human_answer: str
    initial_model_answer: str
    reflection: str
    final_model_answer: str
    grade: int


async def eval_N_prompts_reflection(
    model: str,
    sample: Sample,
) -> Experiment:
    initial_answer = await generate_content(
        model=model, prompt=SOLVE_PROMPT, input=sample.question
    )
    reflection = await generate_content(
        model=model,
        prompt=REFLECT_PROMPT,
        input=f"Question: {sample.question}\nAnswer: {initial_answer}",
    )
    final_answer = await generate_content(
        model=model,
        prompt=REVISE_PROMPT,
        input=f"Question: {sample.question}\nAnswer: {initial_answer}\nCritique: {reflection}",
    )
    grade = await grade_answers(
        model=model,
        question=sample.question,
        human_answer=sample.answer,
        model_answer=final_answer,
    )
    return Experiment(
        question=sample.question,
        human_answer=sample.answer,
        initial_model_answer=initial_answer,
        reflection=reflection,
        final_model_answer=final_answer,
        grade=grade,
    )
