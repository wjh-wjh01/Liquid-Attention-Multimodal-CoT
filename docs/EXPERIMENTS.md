# Experiment Guide

## Controlled Same-Backbone Runs

Use the same visual encoder, language checkpoint, tokenizer revision, split files,
decoding budget, and seed list for all controlled rows.

```bash
python scripts/build_manifest.py \
  --benchmark ScienceQA \
  --source data/raw/scienceqa/problems.json \
  --image-root data/raw/scienceqa/images \
  --output data/processed/scienceqa/test.jsonl
```

Then run an experiment with layered configs:

```bash
python scripts/run_experiment.py \
  --config configs/base.yaml \
  --config configs/benchmarks/scienceqa.yaml \
  --set train.seed=13 \
  --set train.output_dir=outputs/scienceqa/seed13
```

## Ablations

Each ablation should differ from the full model by exactly one component.

```bash
python scripts/run_experiment.py \
  --config configs/base.yaml \
  --config configs/benchmarks/scienceqa.yaml \
  --config configs/ablations/no_visual_diffusion.yaml
```

## Significance

Prediction JSONL files must use identical example ordering.

```bash
python scripts/compare_predictions.py \
  --a outputs/ct_mcot/predictions.jsonl \
  --b outputs/mcout/predictions.jsonl
```

## Required Logs

- `resolved_config.yaml`
- `train_log.jsonl`
- `predictions.jsonl`
- `metrics.json`
- solver NFE and fallback logs for adaptive runs
- retrieval logs for knowledge-enabled runs
- failure-case JSONL files
