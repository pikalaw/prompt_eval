import re
from .gemini import generate_content


MODEL = "gemini-1.5-flash"

_GRADE_PROMPT = """Given a word problem, two responses are provided.
The first one is the reference answer, which is correct.
The second one is the model's answer, which may or may not be correct.
Your job is to judge if the model's answer is the same as the reference answer.
If the model's answer is the same as the reference answer, respond with "1".
Otherwise, respond with "0".
Do not respond with any other text.

For example, suppose you are given this:

```
Question: A train leaves New York for Boston, 200 miles away, at 3:00 PM. Another train leaves Boston for New York at the same time. The first train travels at 60 mph, and the second train travels at 80 mph. At what time do the two trains pass each other?
Reference Answer: 4:00 PM
Model Answer: 4:00 PM
```

Your reply should be a single character:
1
"""


async def grade_answers(*, question: str, human_answer: str, model_answer: str) -> int:
    grade = await generate_content(
        model=MODEL,
        prompt=_GRADE_PROMPT,
        input=_format_grading_input(
            question=question,
            human_answer=human_answer,
            model_answer=model_answer,
        ),
    )
    return int(grade.strip())


def _extract_answer(solution: str) -> str:
    boundary = r"^####\s"
    result = re.split(boundary, solution, flags=re.MULTILINE)
    return result[-1].strip()


def _format_grading_input(
    *, question: str, human_answer: str, model_answer: str
) -> str:
    human_answer = _extract_answer(human_answer)
    model_answer = _extract_answer(model_answer)
    return f"""Question: {question}\nReference Answer: {human_answer}\nModel Answer: {model_answer}"""
