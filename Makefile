.PHONY: install install-vlm test lint synthetic train eval clean

install:
	pip install -e .

install-vlm:
	pip install -e ".[vlm]"

test:
	pytest -q

lint:
	ruff check src scripts tests

synthetic:
	python scripts/make_synthetic.py --output data/synthetic/train.jsonl --num-examples 1000 --seed 13
	python scripts/make_synthetic.py --output data/synthetic/test.jsonl --num-examples 300 --seed 42

train:
	python scripts/train.py --config configs/synthetic.yaml

eval:
	python scripts/evaluate.py --config configs/synthetic.yaml

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +
