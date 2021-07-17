SHELL := /bin/bash
.PHONY = requirements install run check test

requirements:
	@echo Freezing environment ...
	pip-chill --no-chill > requirements.txt

install:
	.venv/bin/pip install requirements.txt

run:
	@python -m ds

check:
	.venv/bin/pylint --disable=C0330,C0326 sdt
	.venv/bin/black --check sdt tests
	.venv/bin/isort --check sdt tests
	.venv/bin/mypy ds tests

test:
	pytest tests