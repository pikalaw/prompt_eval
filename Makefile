eval:
	python -m prompt_eval

analyze:
	python -m prompt_eval.analysis | tee analysis.log

lint:
	mypy .
