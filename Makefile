SHELL := /bin/bash
.PHONY = requirements run

requirements:
	@echo Freezing environment ...
	pip-chill --no-chill > requirements.txt

run:
	@python -m ds