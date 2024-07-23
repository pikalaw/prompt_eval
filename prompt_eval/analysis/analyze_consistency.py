"""Consider replacing this script with a SQL engine."""

from devtools import debug
from pydantic import BaseModel
from typing import Iterable

from ..eval_baseline import Experiment as BaselineExperiment
from ..eval_1_prompt_consistency import Experiment as OnePromptConsistencyExperiment
from ..eval_n_prompts_consistency import Experiment as NPromptConsistencyExperiment
from ..eval_3_solvers_consistency import Experiment as ThreeSolversConsistencyExperiment


class Experiment(BaseModel):
    question: str
    baseline_experiment: BaselineExperiment | None = None
    one_prompt_consistency_experiment: OnePromptConsistencyExperiment | None = None
    n_prompts_consistency_experiment: NPromptConsistencyExperiment | None = None
    three_solvers_consistency_experiment: ThreeSolversConsistencyExperiment | None = (
        None
    )

    @property
    def baseline_grade(self) -> int | None:
        return self.baseline_experiment.grade if self.baseline_experiment else None

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

    @property
    def three_solvers_consistency_grade(self) -> int | None:
        return (
            self.three_solvers_consistency_experiment.grade
            if self.three_solvers_consistency_experiment
            else None
        )


def load_baseline_experiments() -> Iterable[BaselineExperiment]:
    try:
        with open("eval_baseline.json", "r") as f:
            return [BaselineExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_1_prompt_consistency_experiments() -> Iterable[OnePromptConsistencyExperiment]:
    try:
        with open("eval_1_prompt_consistency.json", "r") as f:
            return [
                OnePromptConsistencyExperiment.model_validate_json(row) for row in f
            ]
    except FileNotFoundError:
        return []


def load_n_prompts_consistency_experiments() -> Iterable[NPromptConsistencyExperiment]:
    try:
        with open("eval_n_prompts_consistency.json", "r") as f:
            return [NPromptConsistencyExperiment.model_validate_json(row) for row in f]
    except FileNotFoundError:
        return []


def load_3_solvers_consistency_experiments() -> (
    Iterable[ThreeSolversConsistencyExperiment]
):
    try:
        with open("eval_3_solvers_consistency.json", "r") as f:
            return [
                ThreeSolversConsistencyExperiment.model_validate_json(row) for row in f
            ]
    except FileNotFoundError:
        return []


def load_all_experiments() -> Iterable[Experiment]:
    baseline_experiments = {e.question: e for e in load_baseline_experiments()}
    one_prompt_consistency_experiments = {
        e.question: e for e in load_1_prompt_consistency_experiments()
    }
    n_prompts_consistency_experiments = {
        e.question: e for e in load_n_prompts_consistency_experiments()
    }
    three_solvers_consistency_experiments = {
        e.question: e for e in load_3_solvers_consistency_experiments()
    }

    all_experiments: dict[str, Experiment] = {}

    # Baseline
    for question, baseline_experiment in baseline_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].baseline_experiment = baseline_experiment

    # 1 prompt
    for (
        question,
        one_prompt_consistency_experiment,
    ) in one_prompt_consistency_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].one_prompt_consistency_experiment = (
            one_prompt_consistency_experiment
        )

    # N prompts
    for (
        question,
        n_prompts_consistency_experiment,
    ) in n_prompts_consistency_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].n_prompts_consistency_experiment = (
            n_prompts_consistency_experiment
        )

    # 3 solvers
    for (
        question,
        three_solvers_consistency_experiment,
    ) in three_solvers_consistency_experiments.items():
        if question not in all_experiments:
            all_experiments[question] = Experiment(question=question)
        all_experiments[question].three_solvers_consistency_experiment = (
            three_solvers_consistency_experiment
        )

    return all_experiments.values()


def main() -> None:
    experiments = list(load_all_experiments())

    baseline_correct = len([e for e in experiments if e.baseline_grade == 1])
    baseline_incorrect = len([e for e in experiments if e.baseline_grade == 0])

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

    three_solvers_consistency_correct = len(
        [e for e in experiments if e.three_solvers_consistency_grade == 1]
    )
    three_solvers_consistency_incorrect = len(
        [e for e in experiments if e.three_solvers_consistency_grade == 0]
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

    better_three_solvers_consistency = [
        e
        for e in experiments
        if e.baseline_grade == 0 and e.three_solvers_consistency_grade == 1
    ]
    three_solvers_consistency_improvement = len(better_three_solvers_consistency)

    worse_three_solvers_consistency = [
        e
        for e in experiments
        if e.baseline_grade == 1 and e.three_solvers_consistency_grade == 0
    ]
    three_solvers_consistency_deterioration = len(worse_three_solvers_consistency)

    debug(
        baseline_correct=baseline_correct,
        baseline_incorrect=baseline_incorrect,
        baseline_accuracy=f"{baseline_correct / (baseline_correct + baseline_incorrect):.2%}",
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
    debug(
        three_solvers_consistency_correct=three_solvers_consistency_correct,
        three_solvers_consistency_incorrect=three_solvers_consistency_incorrect,
        three_solvers_consistency_accuracy=f"{three_solvers_consistency_correct / (three_solvers_consistency_correct + three_solvers_consistency_incorrect):.2%}",
        three_solvers_consistency_improvement=three_solvers_consistency_improvement,
        three_solvers_consistency_deterioration=three_solvers_consistency_deterioration,
    )

    debug(better_one_prompt_consistency)
    debug(worse_one_prompt_consistency)

    debug(better_n_prompts_consistency)
    debug(worse_n_prompts_consistency)

    debug(better_three_solvers_consistency)
    debug(worse_three_solvers_consistency)


if __name__ == "__main__":
    main()
