import asyncio
import csv
from devtools import debug
from huggingface_hub import login
import logging
import math
import os
from pydantic import BaseModel
from typing import cast, Iterable, Iterator, TypeVar
from .dataset_loader import load_samples, Sample
from .eval_baseline import eval_baseline
from .eval_1_prompt_reflection import eval_1_prompt_reflection
from .eval_N_prompts_reflection import eval_n_prompts_reflection


T = TypeVar("T")


class Experiment(BaseModel):
    # Data from the eval dataset.
    question: str
    human_answer: str

    # Output from the baseline model.
    llm_baseline_answer: str | None = None
    grade_baseline: int | None = None

    # Output from the 1-prompt reflection model.
    llm_1_prompt_answer: str | None = None
    grade_1_prompt: int | None = None

    # Output from the N-prompts reflection model.
    llm_n_prompts_initial_answer: str | None = None
    llm_n_prompts_reflection: str | None = None
    llm_n_prompts_final_answer: str | None = None
    grade_n_prompts: int | None = None


async def eval_all_experiments(
    model: str,
    sample: Sample,
) -> Experiment:
    experiment = Experiment(
        question=sample.question,
        human_answer=sample.answer,
    )

    try:
        baseline_experiment = await eval_baseline(model, sample)

        experiment.llm_baseline_answer = baseline_experiment.llm_answer
        experiment.grade_baseline = baseline_experiment.grade
    except Exception as e:
        logging.exception(debug.format(sample))

    try:
        one_prompt_experiment = await eval_1_prompt_reflection(model, sample)

        experiment.llm_1_prompt_answer = one_prompt_experiment.llm_answer
        experiment.grade_1_prompt = one_prompt_experiment.grade
    except Exception as e:
        logging.exception(debug.format(sample))

    try:
        n_prompts_experiment = await eval_n_prompts_reflection(model, sample)

        experiment.llm_n_prompts_initial_answer = (
            n_prompts_experiment.initial_model_answer
        )
        experiment.llm_n_prompts_reflection = n_prompts_experiment.reflection
        experiment.llm_n_prompts_final_answer = n_prompts_experiment.final_model_answer
        experiment.grade_n_prompts = n_prompts_experiment.grade
    except Exception as e:
        logging.exception(debug.format(sample))

    return experiment


def escape_string(s: T) -> T:
    """Escape all \n characters."""
    if s is None:
        return None
    assert isinstance(s, str)
    return cast(T, s.replace("\n", "\\n"))


def format_experiment(experiment: Experiment) -> Experiment:
    """For each string fields, escape all \n characters."""
    return Experiment(
        question=escape_string(experiment.question),
        human_answer=escape_string(experiment.human_answer),
        llm_baseline_answer=escape_string(experiment.llm_baseline_answer),
        grade_baseline=experiment.grade_baseline,
        llm_1_prompt_answer=escape_string(experiment.llm_1_prompt_answer),
        grade_1_prompt=experiment.grade_1_prompt,
        llm_n_prompts_initial_answer=escape_string(
            experiment.llm_n_prompts_initial_answer
        ),
        llm_n_prompts_reflection=escape_string(experiment.llm_n_prompts_reflection),
        llm_n_prompts_final_answer=escape_string(experiment.llm_n_prompts_final_answer),
        grade_n_prompts=experiment.grade_n_prompts,
    )


async def eval_and_log(
    model: str,
    sample: Sample,
    writer: csv.DictWriter,
) -> None:
    experiment = await eval_all_experiments(model, sample)
    formatted_experiment = format_experiment(experiment)
    writer.writerow(formatted_experiment.model_dump())


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
