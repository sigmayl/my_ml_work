.PHONY: help install lint format test triage fixtures clean

PY ?= python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

lint: ## ruff (lint + format check)
	ruff check .
	ruff format --check .

format: ## auto-fix and format
	ruff check --fix .
	ruff format .

test: ## pytest
	$(PY) -m pytest

triage: ## run the triage loop on the run_042 fixture (evidence only; ranking is TODO)
	$(PY) -m llm_triage.cli triage \
		--log logs/run_042_crash.log \
		--baseline configs/exp_baseline.yaml \
		--failed configs/exp_failed.yaml \
		--allow-partial

fixtures: ## regenerate the committed crash log from the metrics table
	$(PY) -m llm_triage.fixtures --write

clean: ## remove build/test artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
