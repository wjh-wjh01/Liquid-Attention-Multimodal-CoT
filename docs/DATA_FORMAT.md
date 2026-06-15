# Data Format

The lightweight implementation consumes precomputed multimodal memory tokens as JSONL.

```json
{
  "id": "example-0001",
  "features": [[0.1, 0.2], [0.3, 0.4]],
  "mask": [1, 1],
  "label": 0,
  "support_nodes": [0, 1]
}
```

For full benchmark experiments, generate `features` by concatenating projected text,
image-region, and optional knowledge tokens:

```text
M = [E_text; E_visual; E_knowledge]
```

The current public code intentionally keeps large-model feature extraction outside the
core module because benchmark licenses and checkpoint access differ by user.
