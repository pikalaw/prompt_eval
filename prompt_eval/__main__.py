import asyncio
from huggingface_hub import login
import logging
import os
import sys
from typing import Iterator
from .dataset_loader import load_samples
from .runner import EvalFunc, run_eval

from .eval_list import EVAL_FUNCTIONS


def selected_eval_funcs() -> Iterator[EvalFunc]:
    """Eval functions selected by the user."""
    for eval_func_name in sys.argv[1:]:
        if eval_func_name not in EVAL_FUNCTIONS:
            raise ValueError(f"Unknown eval function: {eval_func_name}")
        yield EVAL_FUNCTIONS[eval_func_name]


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

    for eval_func in selected_eval_funcs():
        await run_eval(
            model=model,
            eval_func=eval_func,
            samples=samples,
            output_filename=f"{eval_func.__name__}.json",
            batch_size=batch_size,
            limit=limit,
        )


asyncio.run(main())
