from datasets import load_dataset
import math
from pydantic import BaseModel
from typing import Iterator


class Sample(BaseModel, frozen=True):
    question: str
    answer: str


def load_samples(
    path: str,
    name: str,
    *,
    split: str,
    limit: int | None = None,
) -> Iterator[Sample]:
    ds = load_dataset(path, name, split=split)
    for row in [row for i, row in enumerate(ds) if i < (limit or math.inf)]:
        yield Sample(question=row["question"], answer=row["answer"])
