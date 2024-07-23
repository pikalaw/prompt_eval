eval_all:
	python -m prompt_eval eval_baseline eval_1_prompt_reflection eval_n_prompts_reflection

analyze:
	python -m prompt_eval.analysis | tee analysis.log

lint:
	mypy .
