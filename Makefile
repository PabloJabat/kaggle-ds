SHELL := /bin/bash
.PHONY = requirements run test

requirements:
	@echo Freezing environment ...
	pip-chill --no-chill > requirements.txt

install:
	.venv/bin/pip install requirements.txt

run:
	@python -m ds

test:
	pytest tests