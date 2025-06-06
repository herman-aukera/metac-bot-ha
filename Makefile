# Metaculus AI Forecasting Bot - Development Makefile
.PHONY: help install install-dev test test-unit test-integration test-e2e test-coverage lint format type-check clean run benchmark docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-e2e      Run end-to-end tests only"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run type checking with mypy"
	@echo "  clean         Clean up cache and temporary files"
	@echo "  run           Run the CLI forecast runner with sample data"
	@echo "  run-cli       Run the original forecasting bot CLI"
	@echo "  forecast      Run forecasts with --submit flag enabled"
	@echo "  benchmark     Run benchmark tests"
	@echo "  docs          Generate documentation"

# Installation
install:
	poetry install --only=main

install-dev:
	poetry install

# Testing
test:
	poetry run pytest

test-unit:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

test-e2e:
	poetry run pytest tests/e2e/ -v

test-coverage:
	poetry run pytest --cov=src --cov-report=html --cov-report=term-missing

test-coverage-xml:
	poetry run pytest --cov=src --cov-report=xml

# Code Quality
lint:
	poetry run flake8 src tests
	poetry run pylint src

format:
	poetry run black src tests
	poetry run isort src tests

type-check:
	poetry run mypy src

# Development
clean:
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf dist
	rm -rf build

# Application
run:
	poetry run python3 cli/run_forecast.py data/questions.json

run-cli:
	poetry run python3 src/main.py

forecast:
	poetry run python3 cli/run_forecast.py data/questions.json --submit

# Benchmarking
benchmark:
	poetry run python3 community_benchmark.py

# Documentation
docs:
	poetry run sphinx-build -b html docs docs/_build/html

# Development workflow shortcuts
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

pre-commit: format lint type-check test-unit
	@echo "Pre-commit checks passed!"

ci: install-dev lint type-check test-coverage-xml
	@echo "CI pipeline complete!"

# Quick development cycle
quick-test: test-unit
quick-check: format lint type-check quick-test
