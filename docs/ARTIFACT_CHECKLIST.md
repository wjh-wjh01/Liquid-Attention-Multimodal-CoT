# Artifact Checklist

This checklist follows the reproducibility items stated in the CT-MCoT paper.

- Exact visual encoder checkpoint and revision.
- Exact language backbone checkpoint, tokenizer version, and revision hash.
- Dataset versions, split files, and held-out example IDs.
- Preprocessing scripts for images, OCR, answer options, and retrieved knowledge.
- Config files for each benchmark and random seed.
- Prediction JSONL files for every method, benchmark, and seed.
- Evaluation scripts for accuracy, macro-F1, exact match, and answer normalization.
- Bootstrap confidence interval and paired significance scripts.
- Solver configuration: method, horizon, step count, tolerance, NFE cap, fallback policy.
- Hardware profile: GPU type/count, precision, batch size, wall-clock latency protocol.
- Failure logs: retrieval errors, solver fallback cases, unsupported visual grounding.
- License terms for datasets and model checkpoints.
