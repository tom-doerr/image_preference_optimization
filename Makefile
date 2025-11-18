.PHONY: test test-fast mypy

test:
	pytest -q

test-fast:
	pytest -q tests/test_proposer_opts_build.py tests/test_modes_dispatch.py

mypy:
	mypy value_model.py value_scorer.py batch_ui.py queue_ui.py pair_ui.py proposer.py modes.py

