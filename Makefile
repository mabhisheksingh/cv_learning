.PHONY: help install-dev format lint fix check all

PYTHON := python3
SRC := src

help:
	@echo "Available targets:"
	@echo "  install-dev  Install project with dev dependencies (uses uv)"
	@echo "  format       Run ruff format on source files"
	@echo "  lint         Run ruff lint checks on source files"
	@echo "  fix          Run ruff lint with auto-fix on source files"
	@echo "  check        Run both format --check and lint"
	@echo "  all          Run format and then lint"

install-dev:
	uv sync --group dev

format:
	uv run --group dev ruff format $(SRC)

lint:
	uv run --group dev ruff check $(SRC)

fix:
	uv run --group dev ruff check --fix $(SRC)

check:
	uv run --group dev ruff format --check $(SRC)
	uv run --group dev ruff check $(SRC)

all: format lint
