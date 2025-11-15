.PHONY: install index test clean

install:
	uv sync

index:
	uv run python run_indexer.py

test:
	uv run python -m pytest tests/

clean:
	rm -rf data/faiss_index

rebuild: clean index
