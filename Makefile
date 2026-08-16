.PHONY: install test lint run evaluate clean

install:
	pip install -r requirements.txt

test:
	pytest -v

lint:
	ruff check app tests

run:
	python -m app.server

evaluate:
	python -m app.evaluate

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
