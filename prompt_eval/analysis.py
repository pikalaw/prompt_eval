import csv
from devtools import debug
from pydantic import field_validator
from typing import Any, Iterator
from .eval_all import Experiment as BaseExperiment


class Experiment(BaseExperiment):
    @field_validator("*", mode="before")
    @classmethod
    def empty_string_should_be_none(cls, value: Any) -> Any | None:
        stripped_value = str(value).strip()
        if stripped_value == "":
            return None
        return value


def read_experiment(path: str) -> Iterator[Experiment]:
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield Experiment.model_validate(row)


def main() -> None:
    experiments = list(read_experiment("eval_result.csv"))

    baseline_correct = len([e for e in experiments if e.grade_baseline == 1])
    baseline_incorrect = len([e for e in experiments if e.grade_baseline == 0])

    one_prompt_correct = len([e for e in experiments if e.grade_1_prompt == 1])
    one_prompt_incorrect = len([e for e in experiments if e.grade_1_prompt == 0])

    n_prompts_correct = len([e for e in experiments if e.grade_n_prompts == 1])
    n_prompts_incorrect = len([e for e in experiments if e.grade_n_prompts == 0])

    one_prompt_improvement = len(
        [e for e in experiments if e.grade_baseline == 0 and e.grade_1_prompt == 1]
    )
    one_prompt_deterioration = len(
        [e for e in experiments if e.grade_baseline == 1 and e.grade_1_prompt == 0]
    )
    n_prompt_improvement = len(
        [e for e in experiments if e.grade_baseline == 0 and e.grade_n_prompts == 1]
    )
    n_prompt_deterioration = len(
        [e for e in experiments if e.grade_baseline == 1 and e.grade_n_prompts == 0]
    )

    debug(
        baseline_correct=baseline_correct,
        baseline_incorrect=baseline_incorrect,
        baseline_accuracy=f"{baseline_correct / (baseline_correct + baseline_incorrect):.2%}",
    )
    debug(
        one_prompt_correct=one_prompt_correct,
        one_prompt_incorrect=one_prompt_incorrect,
        one_prompt_accuracy=f"{one_prompt_correct / (one_prompt_correct + one_prompt_incorrect):.2%}",
        one_prompt_improvement=one_prompt_improvement,
        one_prompt_deterioration=one_prompt_deterioration,
    )
    debug(
        n_prompts_correct=n_prompts_correct,
        n_prompts_incorrect=n_prompts_incorrect,
        n_prompts_accuracy=f"{n_prompts_correct / (n_prompts_correct + n_prompts_incorrect):.2%}",
        n_prompt_improvement=n_prompt_improvement,
        n_prompt_deterioration=n_prompt_deterioration,
    )

    debug(
        better_one_prompt=[
            e for e in experiments if e.grade_baseline == 0 and e.grade_1_prompt == 1
        ]
    )
    debug(
        worse_one_prompt=[
            e for e in experiments if e.grade_baseline == 1 and e.grade_1_prompt == 0
        ]
    )
    debug(
        better_n_prompts=[
            e for e in experiments if e.grade_baseline == 0 and e.grade_n_prompts == 1
        ]
    )
    debug(
        worse_n_prompts=[
            e for e in experiments if e.grade_baseline == 1 and e.grade_n_prompts == 0
        ]
    )


if __name__ == "__main__":
    main()
