import asyncio
from devtools import debug
from huggingface_hub import login
from io import TextIOWrapper
import logging
import math
import os
from typing import Any, Callable, Iterable
from .dataset_loader import load_samples, Sample
from .eval_baseline import eval_baseline
from .eval_1_prompt_reflection import eval_1_prompt_reflection
from .eval_N_prompts_reflection import eval_N_prompts_reflection


done_count = 0


async def eval_and_log(
    eval_func: Callable[[Sample], Any],
    sample: Sample,
    output: TextIOWrapper,
) -> None:
    try:
        experiment = await eval_func(sample)
        output.write(str(debug.format(experiment)))
        output.flush()

        global done_count
        done_count += 1
        if done_count % 10 == 0:
            logging.info(debug.format(done_count))
    except Exception as e:
        logging.exception(debug.format(sample))


async def run_eval(
    *,
    eval_func: Callable[[Sample], Any],
    samples: Iterable[Sample],
    output_filename: str,
    limit: int | None = None
) -> None:
    actual_limit = limit or math.inf
    with open(output_filename, "w") as output:
        await asyncio.gather(
            *[
                eval_and_log(eval_func, sample, output)
                for i, sample in enumerate(samples)
                if i < actual_limit
            ],
            return_exceptions=True,
        )


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

    limit = None
    await run_eval(
        eval_func=eval_baseline,
        samples=samples,
        output_filename="baseline_output.log",
        limit=limit,
    )
    await run_eval(
        eval_func=eval_1_prompt_reflection,
        samples=samples,
        output_filename="1_prompts_output.log",
        limit=limit,
    )
    await run_eval(
        eval_func=eval_N_prompts_reflection,
        samples=samples,
        output_filename="N_prompts_output.log",
        limit=limit,
    )


asyncio.run(main())
