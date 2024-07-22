from devtools import debug
import logging
from pydantic import BaseModel
from .dataset_loader import Sample
from .eval_baseline import eval_baseline
from .eval_1_prompt_reflection import eval_1_prompt_reflection
from .eval_N_prompts_reflection import eval_n_prompts_reflection


class Experiment(BaseModel):
    # Data from the eval dataset.
    question: str
    human_answer: str

    # Output from the baseline model.
    baseline_answer: str | None = None
    grade_baseline: int | None = None

    # Output from the 1-prompt reflection model.
    one_prompt_answer: str | None = None
    grade_1_prompt: int | None = None

    # Output from the N-prompts reflection model.
    n_prompts_initial_answer: str | None = None
    n_prompts_reflection: str | None = None
    n_prompts_final_answer: str | None = None
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

        experiment.baseline_answer = baseline_experiment.llm_answer
        experiment.grade_baseline = baseline_experiment.grade
    except Exception as e:
        logging.exception(debug.format(sample))

    try:
        one_prompt_experiment = await eval_1_prompt_reflection(model, sample)

        experiment.one_prompt_answer = one_prompt_experiment.llm_answer
        experiment.grade_1_prompt = one_prompt_experiment.grade
    except Exception as e:
        logging.exception(debug.format(sample))

    try:
        n_prompts_experiment = await eval_n_prompts_reflection(model, sample)

        experiment.n_prompts_initial_answer = n_prompts_experiment.initial_model_answer
        experiment.n_prompts_reflection = n_prompts_experiment.reflection
        experiment.n_prompts_final_answer = n_prompts_experiment.final_model_answer
        experiment.grade_n_prompts = n_prompts_experiment.grade
    except Exception as e:
        logging.exception(debug.format(sample))

    return experiment
