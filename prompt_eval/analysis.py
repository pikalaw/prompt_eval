from devtools import debug
from pydantic import BaseModel
from typing import Iterable

from .eval_baseline import Experiment as BaselineExperiment
from .eval_1_prompt_reflection import Experiment as OnePromptExperiment
from .eval_n_prompts_reflection import Experiment as NPromptExperiment


class Experiment(BaseModel):
    question: str
    baseline_experiment: BaselineExperiment | None = None
    one_prompt_experiment: OnePromptExperiment | None = None
    n_prompts_experiment: NPromptExperiment | None = None

    @property
    def baseline_grade(self) -> int | None:
        return self.baseline_experiment.grade if self.baseline_experiment else None

    @property
    def one_prompt_grade(self) -> int | None:
        return self.one_prompt_experiment.grade if self.one_prompt_experiment else None

    @property
    def n_prompts_grade(self) -> int | None:
        return self.n_prompts_experiment.grade if self.n_prompts_experiment else None


def load_baseline_experiments() -> Iterable[BaselineExperiment]:
    try:
        with open("eval_baseline.json", "r") as f:
            return [BaselineExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_one_prompt_experiments() -> Iterable[OnePromptExperiment]:
    try:
        with open("eval_1_prompt_reflection.json", "r") as f:
            return [OnePromptExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_n_prompts_experiments() -> Iterable[NPromptExperiment]:
    try:
        with open("eval_n_prompts_reflection.json", "r") as f:
            return [NPromptExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_all_experiments() -> Iterable[Experiment]:
    baseline_experiments = {e.question: e for e in load_baseline_experiments()}
    one_prompt_experiments = {e.question: e for e in load_one_prompt_experiments()}
    n_prompts_experiments = {e.question: e for e in load_n_prompts_experiments()}

    all_experiments: dict[str, Experiment] = {}

    for question, baseline_experiment in baseline_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].baseline_experiment = baseline_experiment

    for question, one_prompt_experiment in one_prompt_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].one_prompt_experiment = one_prompt_experiment

    for question, n_prompts_experiment in n_prompts_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].n_prompts_experiment = n_prompts_experiment

    return all_experiments.values()


def main() -> None:
    experiments = list(load_all_experiments())

    baseline_correct = len([e for e in experiments if e.baseline_grade == 1])
    baseline_incorrect = len([e for e in experiments if e.baseline_grade == 0])

    one_prompt_correct = len([e for e in experiments if e.one_prompt_grade == 1])
    one_prompt_incorrect = len([e for e in experiments if e.one_prompt_grade == 0])

    n_prompts_correct = len([e for e in experiments if e.n_prompts_grade == 1])
    n_prompts_incorrect = len([e for e in experiments if e.n_prompts_grade == 0])

    better_one_prompt = [
        e for e in experiments if e.baseline_grade == 0 and e.one_prompt_grade == 1
    ]
    one_prompt_improvement = len(better_one_prompt)

    worse_one_prompt = [
        e for e in experiments if e.baseline_grade == 1 and e.one_prompt_grade == 0
    ]
    one_prompt_deterioration = len(worse_one_prompt)

    better_n_prompts = [
        e for e in experiments if e.baseline_grade == 0 and e.n_prompts_grade == 1
    ]
    n_prompt_improvement = len(better_n_prompts)

    worse_n_prompts = [
        e for e in experiments if e.baseline_grade == 1 and e.n_prompts_grade == 0
    ]
    n_prompt_deterioration = len(worse_n_prompts)

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

    debug(better_one_prompt)
    debug(worse_one_prompt)
    debug(better_n_prompts)
    debug(worse_n_prompts)


if __name__ == "__main__":
    main()
