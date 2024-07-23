from pydantic import BaseModel
from typing import Awaitable, Callable, Type, TypeAlias
from .dataset_loader import Sample

from .eval_baseline import eval_baseline
from .eval_1_prompt_reflection import eval_1_prompt_reflection
from .eval_n_prompts_reflection import eval_n_prompts_reflection
from .eval_1_prompt_consistency import eval_1_prompt_consistency
from .eval_n_prompts_consistency import eval_n_prompts_consistency


EvalFunc: TypeAlias = Callable[[str, Sample], Awaitable[BaseModel]]


EVAL_FUNCTIONS: dict[str, EvalFunc] = {
    eval_baseline.__name__: eval_baseline,
    eval_1_prompt_reflection.__name__: eval_1_prompt_reflection,
    eval_n_prompts_reflection.__name__: eval_n_prompts_reflection,
    eval_1_prompt_consistency.__name__: eval_1_prompt_consistency,
    eval_n_prompts_consistency.__name__: eval_n_prompts_consistency,
}
