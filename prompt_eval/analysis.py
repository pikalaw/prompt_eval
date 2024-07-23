"""Consider replacing this script with a SQL engine."""

from devtools import debug
from pydantic import BaseModel
from typing import Iterable

from .eval_baseline import Experiment as BaselineExperiment
from .eval_1_prompt_reflection import Experiment as OnePromptReflectionExperiment
from .eval_n_prompts_reflection import Experiment as NPromptReflectionExperiment
from .eval_n_prompts_consistency import Experiment as NPromptConsistencyExperiment
from .eval_n_prompts_consistency import Experiment as NPromptConsistencyExperiment


class Experiment(BaseModel):
    question: str
    baseline_experiment: BaselineExperiment | None = None
    one_prompt_reflection_experiment: OnePromptReflectionExperiment | None = None
    n_prompts_reflection_experiment: NPromptReflectionExperiment | None = None
    one_prompt_consistency_experiment: NPromptConsistencyExperiment | None = None
    n_prompts_consistency_experiment: NPromptConsistencyExperiment | None = None

    @property
    def baseline_grade(self) -> int | None:
        return self.baseline_experiment.grade if self.baseline_experiment else None

    @property
    def one_prompt_reflection_grade(self) -> int | None:
        return (
            self.one_prompt_reflection_experiment.grade
            if self.one_prompt_reflection_experiment
            else None
        )

    @property
    def n_prompts_reflection_grade(self) -> int | None:
        return (
            self.n_prompts_reflection_experiment.grade
            if self.n_prompts_reflection_experiment
            else None
        )

    @property
    def one_prompt_consistency_grade(self) -> int | None:
        return (
            self.one_prompt_consistency_experiment.grade
            if self.one_prompt_consistency_experiment
            else None
        )

    @property
    def n_prompts_consistency_grade(self) -> int | None:
        return (
            self.n_prompts_consistency_experiment.grade
            if self.n_prompts_consistency_experiment
            else None
        )


def load_baseline_experiments() -> Iterable[BaselineExperiment]:
    try:
        with open("eval_baseline.json", "r") as f:
            return [BaselineExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_one_prompt_reflection_experiments() -> Iterable[OnePromptReflectionExperiment]:
    try:
        with open("eval_1_prompt_reflection.json", "r") as f:
            return [OnePromptReflectionExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_n_prompts_reflection_experiments() -> Iterable[NPromptReflectionExperiment]:
    try:
        with open("eval_n_prompts_reflection.json", "r") as f:
            return [NPromptReflectionExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_1_prompt_consistency_experiments() -> Iterable[NPromptConsistencyExperiment]:
    try:
        with open("eval_1_prompt_consistency.json", "r") as f:
            return [NPromptConsistencyExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_n_prompts_consistency_experiments() -> Iterable[NPromptConsistencyExperiment]:
    try:
        with open("eval_n_prompts_consistency.json", "r") as f:
            return [NPromptConsistencyExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_all_experiments() -> Iterable[Experiment]:
    baseline_experiments = {e.question: e for e in load_baseline_experiments()}
    one_prompt_reflection_experiments = {
        e.question: e for e in load_one_prompt_reflection_experiments()
    }
    n_prompts_reflection_experiments = {
        e.question: e for e in load_n_prompts_reflection_experiments()
    }
    one_prompt_consistency_experiments = {
        e.question: e for e in load_1_prompt_consistency_experiments()
    }
    n_prompts_consistency_experiments = {
        e.question: e for e in load_n_prompts_consistency_experiments()
    }

    all_experiments: dict[str, Experiment] = {}

    for question, baseline_experiment in baseline_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].baseline_experiment = baseline_experiment

    for (
        question,
        one_prompt_reflection_experiment,
    ) in one_prompt_reflection_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].one_prompt_reflection_experiment = (
            one_prompt_reflection_experiment
        )

    for (
        question,
        n_prompts_reflection_experiment,
    ) in n_prompts_reflection_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].n_prompts_reflection_experiment = (
            n_prompts_reflection_experiment
        )

    for (
        question,
        one_prompt_consistency_experiment,
    ) in one_prompt_consistency_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].one_prompt_consistency_experiment = (
            one_prompt_consistency_experiment
        )

    for (
        question,
        n_prompts_consistency_experiment,
    ) in n_prompts_consistency_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].n_prompts_consistency_experiment = (
            n_prompts_consistency_experiment
        )

    return all_experiments.values()


def main() -> None:
    experiments = list(load_all_experiments())

    baseline_correct = len([e for e in experiments if e.baseline_grade == 1])
    baseline_incorrect = len([e for e in experiments if e.baseline_grade == 0])

    one_prompt_reflection_correct = len(
        [e for e in experiments if e.one_prompt_reflection_grade == 1]
    )
    one_prompt_reflection_incorrect = len(
        [e for e in experiments if e.one_prompt_reflection_grade == 0]
    )

    n_prompts_reflection_correct = len(
        [e for e in experiments if e.n_prompts_reflection_grade == 1]
    )
    n_prompts_reflection_incorrect = len(
        [e for e in experiments if e.n_prompts_reflection_grade == 0]
    )

    better_one_prompt_reflection = [
        e
        for e in experiments
        if e.baseline_grade == 0 and e.one_prompt_reflection_grade == 1
    ]
    one_prompt_reflection_improvement = len(better_one_prompt_reflection)

    worse_one_prompt_reflection = [
        e
        for e in experiments
        if e.baseline_grade == 1 and e.one_prompt_reflection_grade == 0
    ]
    one_prompt_reflection_deterioration = len(worse_one_prompt_reflection)

    better_n_prompts_reflection = [
        e
        for e in experiments
        if e.baseline_grade == 0 and e.n_prompts_reflection_grade == 1
    ]
    n_prompt_reflection_improvement = len(better_n_prompts_reflection)

    worse_n_prompts_reflection = [
        e
        for e in experiments
        if e.baseline_grade == 1 and e.n_prompts_reflection_grade == 0
    ]
    n_prompt_reflection_deterioration = len(worse_n_prompts_reflection)

    one_prompt_consistency_correct = len(
        [e for e in experiments if e.one_prompt_consistency_grade == 1]
    )
    one_prompt_consistency_incorrect = len(
        [e for e in experiments if e.one_prompt_consistency_grade == 0]
    )

    n_prompts_consistency_correct = len(
        [e for e in experiments if e.n_prompts_consistency_grade == 1]
    )
    n_prompts_consistency_incorrect = len(
        [e for e in experiments if e.n_prompts_consistency_grade == 0]
    )

    better_one_prompt_consistency = [
        e
        for e in experiments
        if e.baseline_grade == 0 and e.one_prompt_consistency_grade == 1
    ]
    one_prompt_consistency_improvement = len(better_one_prompt_consistency)

    worse_one_prompt_consistency = [
        e
        for e in experiments
        if e.baseline_grade == 1 and e.one_prompt_consistency_grade == 0
    ]
    one_prompt_consistency_deterioration = len(worse_one_prompt_consistency)

    better_n_prompts_consistency = [
        e
        for e in experiments
        if e.baseline_grade == 0 and e.n_prompts_consistency_grade == 1
    ]
    n_prompt_consistency_improvement = len(better_n_prompts_consistency)

    worse_n_prompts_consistency = [
        e
        for e in experiments
        if e.baseline_grade == 1 and e.n_prompts_consistency_grade == 0
    ]
    n_prompt_consistency_deterioration = len(worse_n_prompts_consistency)

    debug(
        baseline_correct=baseline_correct,
        baseline_incorrect=baseline_incorrect,
        baseline_accuracy=f"{baseline_correct / (baseline_correct + baseline_incorrect):.2%}",
    )
    debug(
        one_prompt_reflection_correct=one_prompt_reflection_correct,
        one_prompt_reflection_incorrect=one_prompt_reflection_incorrect,
        one_prompt_reflection_accuracy=f"{one_prompt_reflection_correct / (one_prompt_reflection_correct + one_prompt_reflection_incorrect):.2%}",
        one_prompt_reflection_improvement=one_prompt_reflection_improvement,
        one_prompt_reflection_deterioration=one_prompt_reflection_deterioration,
    )
    debug(
        n_prompts_reflection_correct=n_prompts_reflection_correct,
        n_prompts_reflection_incorrect=n_prompts_reflection_incorrect,
        n_prompts_reflection_accuracy=f"{n_prompts_reflection_correct / (n_prompts_reflection_correct + n_prompts_reflection_incorrect):.2%}",
        n_prompt_reflection_improvement=n_prompt_reflection_improvement,
        n_prompt_reflection_deterioration=n_prompt_reflection_deterioration,
    )
    debug(
        one_prompt_consistency_correct=one_prompt_consistency_correct,
        one_prompt_consistency_incorrect=one_prompt_consistency_incorrect,
        one_prompt_consistency_accuracy=f"{one_prompt_consistency_correct / (one_prompt_consistency_correct + one_prompt_consistency_incorrect):.2%}",
        one_prompt_consistency_improvement=one_prompt_consistency_improvement,
        one_prompt_consistency_deterioration=one_prompt_consistency_deterioration,
    )
    debug(
        n_prompts_consistency_correct=n_prompts_consistency_correct,
        n_prompts_consistency_incorrect=n_prompts_consistency_incorrect,
        n_prompts_consistency_accuracy=f"{n_prompts_consistency_correct / (n_prompts_consistency_correct + n_prompts_consistency_incorrect):.2%}",
        n_prompt_consistency_improvement=n_prompt_consistency_improvement,
        n_prompt_consistency_deterioration=n_prompt_consistency_deterioration,
    )

    debug(better_one_prompt_reflection)
    debug(worse_one_prompt_reflection)

    debug(better_n_prompts_reflection)
    debug(worse_n_prompts_reflection)

    debug(better_one_prompt_consistency)
    debug(worse_one_prompt_consistency)

    debug(better_n_prompts_consistency)
    debug(worse_n_prompts_consistency)


if __name__ == "__main__":
    main()
