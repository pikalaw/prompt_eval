eval_all:
	python -m prompt_eval eval_baseline eval_1_prompt_reflection eval_n_prompts_reflection

analyze:
	python -m prompt_eval.analysis.analysis | tee analysis.log

analyze_consistency:
	python -m prompt_eval.analysis.analyze_consistency | tee analyze_consistency.log

lint:
	mypy .
