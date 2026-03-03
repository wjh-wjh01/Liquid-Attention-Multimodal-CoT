.PHONY: prepare run collect test all

prepare:
	python3 scripts/prepare_data.py --config configs/datasets/local_v1.yaml

run:
	python3 scripts/run_experiments.py --config configs/base.yaml --matrix configs/experiments/experiment_matrix.yaml

collect:
	python3 scripts/collect_results.py --runs-dir outputs/runs --output-dir outputs/reports

test:
	python3 -m unittest discover -s tests -p 'test_*.py'

all: prepare run collect
