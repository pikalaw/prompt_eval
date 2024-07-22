import asyncio
import csv
from huggingface_hub import login
import logging
import math
import os
from typing import cast, Iterable, Iterator, TypeVar
from .dataset_loader import load_samples, Sample
from .eval_all import Experiment, eval_all_experiments


T = TypeVar("T")


async def eval_and_log(
    model: str,
    sample: Sample,
    writer: csv.DictWriter,
) -> None:
    experiment = await eval_all_experiments(model, sample)
    writer.writerow(experiment.model_dump())


def batch_samples(samples: Iterable[Sample], batch_size: int) -> Iterator[list[Sample]]:
    batch: list[Sample] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []


async def run_eval(
    *,
    model: str,
    samples: Iterable[Sample],
    output_filename: str,
    batch_size: int = 10,
    limit: int | None = None,
) -> None:
    actual_limit = limit or math.inf
    done = 0
    bad = 0
    with open(output_filename, "w") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=Experiment.model_fields.keys(),
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()

        for batch in batch_samples(samples, batch_size=batch_size):
            results = await asyncio.gather(
                *[eval_and_log(model, sample, writer) for sample in batch],
                return_exceptions=True,
            )

            done += len(results)
            bad += len(
                [exception for exception in results if isinstance(exception, Exception)]
            )
            logging.info(f"Done {done} samples, with {bad} errors.")

            if done >= actual_limit:
                break


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="error.log",
        filemode="w",
    )

    token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=token)

    samples = list(load_samples("openai/gsm8k", "main", split="train"))

    model = "gemini-1.5-flash"
    batch_size = 10
    limit = None

    await run_eval(
        model=model,
        samples=samples,
        output_filename="eval_result.csv",
        batch_size=batch_size,
        limit=limit,
    )


asyncio.run(main())
