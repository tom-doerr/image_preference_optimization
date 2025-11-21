.PHONY: test test-fast mypy commit push

test:
	pytest -q

test-fast:
	pytest -q tests/test_proposer_opts_build.py tests/test_modes_dispatch.py

mypy:
	mypy value_model.py value_scorer.py batch_ui.py proposer.py

commit:
	git commit -am "wip"

push:
	git push
