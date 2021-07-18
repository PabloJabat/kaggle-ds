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
	.venv/bin/pylint --disable=C0330,C0326 sdt
	.venv/bin/black --check sdt tests
	.venv/bin/isort --check sdt tests
	.venv/bin/mypy sdt tests