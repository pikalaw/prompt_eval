from pydantic import BaseModel
from .dataset_loader import Sample
from .gemini import generate_content
from .grader import grade_answers


SOLVE_PROMPT = """You are a team of three word-problem solvers: Alice, Bob and Carol.
You will each take turns solving a word problem.
Alice goes first. She solves the problem and passes it to Bob.
Bob will evaluate Alice's solution, then finds a different perspective to the problem, and solve the problem using the new perspective.
Bob passes the problem to Carol.
Carol will evaluate both Alice's and Bob's solutions, then finds a third perspective to the problem, and solve the problem using the new perspective.
Now, you have 3 solution candidates to the problem.
These solutions may have errors.
Identify the parts of these solutions that are consistent among them.
Then, formulate a final solution from the consistent parts of the 3 solutions.

Respond in the following format:

```
Alice: <Alice's first step-by-step reasoning>
#### <Alice's answer in succint form>
Bob: <Bob's analysis of Alice's solution. His new perspective and the new reasoning from that.>
#### <Bob's answer in succint form>
Carol: <Carol's analysis of Alice's and Bob's solutions. Her new perspective and the new reasoning from that.>
#### <Carol's answer in succint form>
Consistency analysis: <Identify the consistent parts of the 3 solutions>
#### <The final answer in succint form>
```

For example, suppose the word problem is this:
```
A train leaves New York for Boston, 200 miles away, at 3:00 PM. Another train leaves Boston for New York at the same time. The first train travels at 60 mph, and the second train travels at 80 mph. At what time do the two trains pass each other?
```

Your reply could be this:
```
Alice: The approach speed is 60 + 80 = 140 mph. So, the two trains will meet in 200 / 140 = 1.43 hours. Since the first train left at 3:00 PM, the two trains will meet at 3:00 PM + 1.43 hours = 4:26 PM.
#### 4:00 PM
Bob: Alice's solution is correct. The two trains will meet at 4:00 PM.
#### 4:00 PM
Carol: Alice and Bob are correct. The two trains will meet at 4:00 PM.
#### 4:00 PM
Consistency analysis: The three solutions are consistent. The answer is 4:00 PM.
#### 4:00 PM
```
"""


class Experiment(BaseModel, frozen=True):
    question: str
    human_answer: str
    llm_answer: str
    grade: int


async def eval_3_solvers_consistency(
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
