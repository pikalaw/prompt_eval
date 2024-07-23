import asyncio
from io import TextIOWrapper
import logging
import math
from typing import Iterable, Iterator
from .dataset_loader import Sample
from .eval_list import EvalFunc


async def _eval_and_log(
    model: str,
    eval_func: EvalFunc,
    sample: Sample,
    output: TextIOWrapper,
) -> None:
    experiment = await eval_func(model, sample)
    output.write(experiment.model_dump_json())
    output.write("\n")


def _batch_samples(
    samples: Iterable[Sample], batch_size: int
) -> Iterator[list[Sample]]:
    batch: list[Sample] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []


async def run_eval(
    *,
    model: str,
    eval_func: EvalFunc,
    samples: Iterable[Sample],
    output_filename: str,
    batch_size: int = 10,
    limit: int | None = None,
) -> None:
    actual_limit = limit or math.inf
    done = 0
    bad = 0
    with open(output_filename, "w") as output:
        for batch in _batch_samples(samples, batch_size=batch_size):
            results = await asyncio.gather(
                *[_eval_and_log(model, eval_func, sample, output) for sample in batch],
                return_exceptions=True,
            )

            done += len(results)
            bad += len(
                [exception for exception in results if isinstance(exception, Exception)]
            )
            logging.info(
                f"Eval {eval_func.__name__}: Done {done} samples, with {bad} errors."
            )

            if done >= actual_limit:
                break
