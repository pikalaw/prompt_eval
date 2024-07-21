import asyncio
from devtools import debug
from huggingface_hub import login
from io import TextIOWrapper
import logging
import math
import os
from typing import Any, Awaitable, Callable, Iterable, Iterator
from .dataset_loader import load_samples, Sample
from .eval_baseline import eval_baseline
from .eval_1_prompt_reflection import eval_1_prompt_reflection
from .eval_N_prompts_reflection import eval_N_prompts_reflection


async def eval_and_log(
    model: str,
    eval_func: Callable[[str, Sample], Awaitable[Any]],
    sample: Sample,
    output: TextIOWrapper,
) -> None:
    try:
        experiment = await eval_func(model, sample)
        output.write(str(debug.format(experiment)))
        output.flush()
    except Exception as e:
        logging.exception(debug.format(sample))


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
    eval_func: Callable[[str, Sample], Awaitable[Any]],
    samples: Iterable[Sample],
    output_filename: str,
    batch_size: int = 10,
    limit: int | None = None,
) -> None:
    actual_limit = limit or math.inf
    done = 0
    bad = 0
    with open(output_filename, "w") as output:
        for batch in batch_samples(samples, batch_size=batch_size):
            results = await asyncio.gather(
                *[
                    eval_and_log(model, eval_func, sample, output)
                    for i, sample in enumerate(batch)
                ],
                return_exceptions=True,
            )

            done += len(batch)
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
        eval_func=eval_baseline,
        samples=samples,
        output_filename="baseline_output.log",
        batch_size=batch_size,
        limit=limit,
    )
    await run_eval(
        model=model,
        eval_func=eval_1_prompt_reflection,
        samples=samples,
        output_filename="1_prompts_output.log",
        batch_size=batch_size,
        limit=limit,
    )
    await run_eval(
        model=model,
        eval_func=eval_N_prompts_reflection,
        samples=samples,
        output_filename="N_prompts_output.log",
        batch_size=batch_size,
        limit=limit,
    )


asyncio.run(main())
