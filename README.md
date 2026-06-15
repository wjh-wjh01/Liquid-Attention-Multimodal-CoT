# CT-MCoT: Continuous-Time Multimodal Chain-of-Thought Reasoning

Reference implementation for **Continuous-Time Dynamics for Multimodal Chain-of-Thought Reasoning**.

CT-MCoT represents multimodal reasoning as a continuous latent trajectory. The core module jointly evolves latent thought states `H(t)` and multimodal attention-logit states `S(t)` with liquid time constants, fixed-step ODE solvers, and optional graph diffusion over visual tokens.

## Paper Scope

This repository contains:

- A runnable PyTorch implementation of the CT-MCoT liquid thought and attention block.
- A larger research-code layout with modules for memory construction, VLM adapters, retrieval, losses, diagnostics, statistical testing, callbacks, checkpointing, and experiment orchestration.
- Fixed-step Euler, midpoint, and RK4 solvers.
- A synthetic multimodal reachability benchmark for testing latent superposition behavior.
- Training, evaluation, prediction, and metric scripts.
- A 7B same-backbone experiment template matching the reproducibility protocol in the paper.

The public repository does **not** include proprietary model weights, benchmark data, or generated prediction logs. For ScienceQA, A-OKVQA, MMMU, MMStar, and MathVista, users must download the official datasets and fill in the paths/checkpoints in `configs/paper_7b_template.yaml`.

## Method Summary

Given projected text, image, and optional knowledge tokens:

```text
M = [E_text; E_visual; E_knowledge]
```

CT-MCoT integrates:

```text
dH/dt = F_H(H, S, M, t)
dS/dt = F_S(H, S, M, t)
```

where `H` is the latent thought state and `S` is the attention-logit state over multimodal memory. The final answer is decoded from the terminal thought state `H(T)` and pooled memory.

## Repository Layout

```text
ct-mcot/
  configs/
    base.yaml
    synthetic.yaml             # runnable toy experiment
    paper_7b_template.yaml     # paper-scale reproduction template
    benchmarks/                # ScienceQA, A-OKVQA, MMMU, MMStar, MathVista
    ablations/                 # component ablation configs
  docs/
    ARTIFACT_CHECKLIST.md
    DATA_FORMAT.md
  scripts/
    analyze_failures.py
    build_manifest.py
    compare_predictions.py
    export_features.py
    make_synthetic.py
    run_pipeline.py
    run_experiment.py
    slice_metrics.py
    train_baseline.py
    train.py
    evaluate.py
    summarize_predictions.py
  src/ct_mcot/
    analysis/                  # failure analysis and slice metrics
    baselines/                 # No-CoT, latent recurrent, ODE Transformer
    datasets/                  # benchmark adapters and answer normalization
    diagnostics/               # attention and Lyapunov diagnostics
    distributed/               # torch distributed helpers
    evaluation/                # paired bootstrap and McNemar tests
    features/                  # feature cache and dry-run feature export
    modules/                   # memory encoder, heads, losses, VLM adapters
    retrieval/                 # entity linking and hybrid KG retrieval
    training/                  # trainer, callbacks, curriculum, scheduler
    visualization/             # JSONL exports for plots and heatmaps
    utils/                     # config, logging, seed, checkpoint helpers
    model.py                   # CT-MCoT block
    solvers.py                 # Euler, midpoint, RK4
    data.py                    # JSONL dataset and synthetic generator
    train.py
    evaluate.py
    metrics.py
  tests/
    test_model_shapes.py
```

## Installation

```bash
git clone https://github.com/<USER>/ct-mcot.git
cd ct-mcot
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For full vision-language experiments:

```bash
pip install -r requirements-vlm.txt
```

## Quick Start: Synthetic Reachability

Generate train/test data:

```bash
python scripts/make_synthetic.py --output data/synthetic/train.jsonl --num-examples 1000 --seed 13
python scripts/make_synthetic.py --output data/synthetic/test.jsonl --num-examples 300 --seed 42
```

Train:

```bash
python scripts/train.py --config configs/synthetic.yaml
```

Evaluate:

```bash
python scripts/evaluate.py --config configs/synthetic.yaml
python scripts/summarize_predictions.py --predictions outputs/synthetic/predictions.jsonl
```

The larger experiment runner supports layered configs and command-line overrides:

```bash
python scripts/run_experiment.py \
  --config configs/base.yaml \
  --set train.output_dir=outputs/full_style_seed13 \
  --set train.seed=13
```

Baseline and analysis examples:

```bash
python scripts/train_baseline.py --config configs/synthetic.yaml --baseline no_cot
python scripts/analyze_failures.py --predictions outputs/synthetic/predictions.jsonl --output outputs/synthetic/failures.jsonl
python scripts/slice_metrics.py --predictions outputs/synthetic/predictions.jsonl
python scripts/run_pipeline.py --name synthetic --dry-run
```

Expected outputs:

```text
outputs/synthetic/ct_mcot.pt
outputs/synthetic/predictions.jsonl
outputs/synthetic/metrics.json
```

## Data Format

The lightweight training path expects precomputed multimodal memory tokens in JSONL:

```json
{
  "id": "example-0001",
  "features": [[0.1, 0.2], [0.3, 0.4]],
  "mask": [1, 1],
  "label": 0,
  "support_nodes": [0, 1]
}
```

For benchmark experiments, produce `features` by projecting text tokens, image patches/regions, and retrieved knowledge nodes into the same hidden dimension. See `docs/DATA_FORMAT.md`.

## Paper-Scale Reproduction Protocol

The controlled paper setting uses:

- Visual encoder: CLIP ViT-L/14.
- Language backbone: same 7B decoder-only checkpoint for all controlled methods.
- LoRA: rank 16, alpha 32, dropout 0.05.
- Latent slots: 8 for ScienceQA, A-OKVQA, and MMStar; 12 for MMMU, MathVista, and reachability.
- Solver: RK4, `T=1.0`, 12 fixed steps for training and fixed-step inference.
- Adaptive inference: Dormand-Prince 5(4), `rtol=1e-3`, `atol=1e-4`, max NFE 32.
- Seeds: 13, 21, 42.
- Metrics: accuracy, macro-F1, exact match where applicable, region recall/pointing-game accuracy when region labels exist, latency, NFE, and peak GPU memory.

Before reporting results, fill in:

```text
configs/paper_7b_template.yaml
```

Benchmark-specific config stubs are in:

```text
configs/benchmarks/
```

Ablation stubs for Table 4.5 style reporting are in:

```text
configs/ablations/
```

and archive the items in:

```text
docs/ARTIFACT_CHECKLIST.md
```

## Benchmark Reporting Format

For each benchmark and seed, report:

```text
method, benchmark, seed, split, checkpoint_revision, solver, steps, max_nfe,
accuracy, macro_f1, exact_match, latency_ms, peak_memory_gb, prediction_file
```

Prediction files should be JSONL and include at least:

```json
{
  "id": "example-id",
  "label": 1,
  "prediction": 1,
  "probabilities": [0.03, 0.97]
}
```

## Citation

```bibtex
@article{ctmcot2026,
  title = {Continuous-Time Dynamics for Multimodal Chain-of-Thought Reasoning},
  author = {Anonymous},
  journal = {Pattern Recognition Letters},
  year = {2026}
}
```

## License

MIT. Dataset and model checkpoint licenses remain governed by their original providers.
