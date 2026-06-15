# CT-MCoT Model Card

## Intended Use

CT-MCoT is intended for controlled research on multimodal reasoning, latent
chain-of-thought, attention dynamics, and adaptive continuous-time computation.

## Out-of-Scope Use

The repository should not be treated as a frontier-model leaderboard entry.
The paper-scale configuration depends on externally downloaded datasets and
model checkpoints.

## Training Data

The repository does not redistribute ScienceQA, A-OKVQA, MMMU, MMStar, or
MathVista. Users must comply with each dataset license.

## Evaluation

Recommended reporting includes accuracy, macro-F1, exact match when applicable,
region-grounding metrics when labels exist, latency, NFE, memory, confidence
intervals, and paired tests.

## Limitations

- Retrieval quality can dominate knowledge-intensive failures.
- Symbolic arithmetic remains brittle without a verified solver.
- Adaptive solvers can introduce latency variance.
- Diagnostic scores are not full causal explanations.
