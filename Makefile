VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

.PHONY: test lint format check clean

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

$(VENV)/.installed: $(VENV)/bin/activate requirements.txt
	$(PIP) install -r requirements.txt
	touch $@

$(VENV)/.installed-dev: $(VENV)/bin/activate requirements-dev.txt requirements.txt
	$(PIP) install -r requirements-dev.txt
	touch $@

test: $(VENV)/.installed-dev
	$(PYTEST) tests/ -v

lint: $(VENV)/.installed-dev
	$(RUFF) check .
	$(RUFF) format --check .

format: $(VENV)/.installed-dev
	$(RUFF) check --fix .
	$(RUFF) format .

check: lint test

clean:
	rm -rf $(VENV) .pytest_cache __pycache__ tests/__pycache__
