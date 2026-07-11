.PHONY: install test lint clean

install:
	pip install -r requirements.txt

test:
	pytest -v

lint:
	ruff check app tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
