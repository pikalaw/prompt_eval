"""Consider replacing this script with a SQL engine."""

from devtools import debug
from pydantic import BaseModel
from typing import Iterable

from ..eval_baseline import Experiment as BaselineExperiment
from ..eval_no_cot import Experiment as NoCotExperiment


class Experiment(BaseModel):
    question: str
    baseline_experiment: BaselineExperiment | None = None
    no_cot_experiment: NoCotExperiment | None = None

    @property
    def baseline_grade(self) -> int | None:
        return self.baseline_experiment.grade if self.baseline_experiment else None

    @property
    def no_cot_grade(self) -> int | None:
        return self.no_cot_experiment.grade if self.no_cot_experiment else None


def load_baseline_experiments() -> Iterable[BaselineExperiment]:
    try:
        with open("eval_baseline.json", "r") as f:
            return [BaselineExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_no_cot_experiments() -> Iterable[NoCotExperiment]:
    try:
        with open("eval_no_cot.json", "r") as f:
            return [NoCotExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_all_experiments() -> Iterable[Experiment]:
    baseline_experiments = {e.question: e for e in load_baseline_experiments()}
    no_cot_experiments = {e.question: e for e in load_no_cot_experiments()}

    all_experiments: dict[str, Experiment] = {}

    # Baseline
    for question, baseline_experiment in baseline_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].baseline_experiment = baseline_experiment

    # No COT
    for question, no_cot_experiment in no_cot_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].no_cot_experiment = no_cot_experiment

    return all_experiments.values()


def main() -> None:
    experiments = list(load_all_experiments())

    baseline_correct = len([e for e in experiments if e.baseline_grade == 1])
    baseline_incorrect = len([e for e in experiments if e.baseline_grade == 0])

    no_cot_correct = len([e for e in experiments if e.no_cot_grade == 1])
    no_cot_incorrect = len([e for e in experiments if e.no_cot_grade == 0])

    better_no_cot = [
        e for e in experiments if e.baseline_grade == 0 and e.no_cot_grade == 1
    ]
    no_cot_improvement = len(better_no_cot)

    worse_no_cot = [
        e for e in experiments if e.baseline_grade == 1 and e.no_cot_grade == 0
    ]
    no_cot_deterioration = len(worse_no_cot)

    debug(
        baseline_correct=baseline_correct,
        baseline_incorrect=baseline_incorrect,
        baseline_accuracy=f"{baseline_correct / (baseline_correct + baseline_incorrect):.2%}",
    )
    debug(
        no_cot_correct=no_cot_correct,
        no_cot_incorrect=no_cot_incorrect,
        no_cot_accuracy=f"{no_cot_correct / (no_cot_correct + no_cot_incorrect):.2%}",
        no_cot_improvement=no_cot_improvement,
        no_cot_deterioration=no_cot_deterioration,
    )

    debug(better_no_cot)
    debug(worse_no_cot)


if __name__ == "__main__":
    main()
