SHELL := /bin/bash
.PHONY = requirements install run check test

requirements:
	@echo Freezing environment ...
	pip-chill --no-chill > requirements.txt

run:
	@python -m ds

.venv:
	python3.9 -m venv $@

.venv/updated: requirements.txt .venv
	.venv/bin/pip install -r $<
	touch $@

test: .venv/updated
	.venv/bin/python -m pytest tests

check: .venv/updated
	.venv/bin/pylint --disable=C0330,C0326 ds
	.venv/bin/black --check ds tests
	.venv/bin/isort --check ds tests
	.venv/bin/mypy ds tests