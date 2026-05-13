.PHONY: run test lint format docker-build docker-run health clean install dev-install

PY ?= python3
APP_DIR := $(shell pwd)

install:
	$(PY) -m pip install -e .

dev-install:
	$(PY) -m pip install -e ".[dev]" && pre-commit install

run:
	$(PY) -m streamlit run app.py

test:
	$(PY) -m pytest tests/

lint:
	$(PY) -m ruff check .
	$(PY) -m mypy components/ app.py

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

health:
	$(PY) health_server.py

docker-build:
	docker build -t zakupkicheck:2.0 .

docker-run:
	docker run -p 8501:8501 -p 8080:8080 --env-file ../../.env zakupkicheck:2.0

train-model:
	$(PY) models/train_lr.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
