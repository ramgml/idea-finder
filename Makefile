.PHONY: sync lint typecheck test gates hooks

sync:
	uv sync

lint:
	uv run ruff check .

typecheck:
	uv run ty check .

test:
	uv run pytest

gates: lint typecheck test

hooks:
	git config core.hooksPath .githooks
