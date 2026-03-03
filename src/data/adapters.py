from __future__ import annotations

import csv
import json
import pathlib
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.common.io_utils import write_json, write_jsonl
from src.data.schema import UnifiedSample

DATASET_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "scienceqa": {
        "question": ["question", "query", "problem", "stem"],
        "choices": ["choices", "options", "candidates", "answer_choices"],
        "answer": ["answer_idx", "answer", "label", "target"],
        "image": ["image", "image_path", "image_file", "img", "figure"],
        "difficulty": ["difficulty", "level", "grade"],
        "split": ["split", "set"],
    },
    "mmlu_pro": {
        "question": ["question", "query", "problem", "instruction"],
        "choices": ["choices", "options", "candidates"],
        "answer": ["answer_idx", "answer", "label", "gold"],
        "image": ["image", "image_path", "img"],
        "difficulty": ["difficulty", "subject_level", "level"],
        "split": ["split", "set"],
    },
    "cmmcot": {
        "question": ["question", "query", "problem", "instruction"],
        "choices": ["choices", "options", "candidates"],
        "answer": ["answer_idx", "answer", "label", "gold"],
        "image": ["image", "image_path", "image_file", "img", "figure"],
        "difficulty": ["difficulty", "complexity", "level"],
        "split": ["split", "set"],
    },
}

GENERIC_ALIASES = {
    "question": ["question", "query", "prompt", "problem"],
    "choices": ["choices", "options", "candidates", "answers"],
    "answer": ["answer", "answer_idx", "label", "target", "gold"],
    "image": ["image", "image_path", "img", "figure"],
    "difficulty": ["difficulty", "level"],
    "split": ["split", "set"],
}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "eval": "test",
}


def _get_first(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in record:
            return record[k]
    return None


def _normalize_split(value: Any) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    return SPLIT_ALIASES.get(v)


def _split_from_path(path: pathlib.Path) -> Optional[str]:
    lower = str(path).lower()
    for key, out in SPLIT_ALIASES.items():
        if re.search(rf"(^|[\\/_\-.]){re.escape(key)}($|[\\/_\-.])", lower):
            return out
    return None


def _parse_json(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        rows: List[Dict[str, Any]] = []
        has_split_obj = all(k in obj for k in ("train", "val", "test"))
        if has_split_obj:
            for split in ("train", "val", "test"):
                for item in obj.get(split, []):
                    if isinstance(item, dict):
                        item = dict(item)
                        item.setdefault("split", split)
                        rows.append(item)
            return rows
        for key in ("data", "examples", "items", "records"):
            if key in obj and isinstance(obj[key], list):
                return [x for x in obj[key] if isinstance(x, dict)]
        if all(not isinstance(v, (dict, list)) for v in obj.values()):
            return [obj]
    return []


def _parse_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _parse_csv(path: pathlib.Path, delimiter: str = ",") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(dict(row))
    return rows


def _extract_choices(record: Dict[str, Any], aliases: Dict[str, List[str]]) -> List[str]:
    raw_choices = _get_first(record, aliases["choices"] + GENERIC_ALIASES["choices"])
    choices: List[str] = []
    if isinstance(raw_choices, list):
        choices = [str(x).strip() for x in raw_choices if str(x).strip()]
    elif isinstance(raw_choices, dict):
        for key in sorted(raw_choices.keys()):
            val = str(raw_choices[key]).strip()
            if val:
                choices.append(val)
    elif isinstance(raw_choices, str):
        if "||" in raw_choices:
            choices = [x.strip() for x in raw_choices.split("||") if x.strip()]
        elif "\n" in raw_choices:
            choices = [x.strip() for x in raw_choices.splitlines() if x.strip()]
        else:
            choices = [x.strip() for x in raw_choices.split(";") if x.strip()]

    if not choices:
        abc = []
        for letter in ["A", "B", "C", "D", "E", "F"]:
            for cand in (letter, letter.lower(), f"option_{letter.lower()}", f"choice_{letter.lower()}"):
                if cand in record and str(record[cand]).strip():
                    abc.append(str(record[cand]).strip())
                    break
        choices = abc

    if len(choices) < 2:
        choices = choices + ["Unknown"] * (2 - len(choices))

    return choices


def _extract_answer_idx(record: Dict[str, Any], aliases: Dict[str, List[str]], choices: List[str]) -> int:
    raw = _get_first(record, aliases["answer"] + GENERIC_ALIASES["answer"])
    if raw is None:
        return -1

    if isinstance(raw, int):
        return raw if 0 <= raw < len(choices) else -1

    s = str(raw).strip()
    if not s:
        return -1

    if s.isdigit():
        idx = int(s)
        if 0 <= idx < len(choices):
            return idx
        if 1 <= idx <= len(choices):
            return idx - 1
        return -1

    if len(s) == 1 and s.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        idx = ord(s.upper()) - ord("A")
        return idx if 0 <= idx < len(choices) else -1

    sl = s.lower()
    for i, c in enumerate(choices):
        if sl == c.lower():
            return i

    return -1


def _resolve_image_path(raw_value: Any, raw_dir: pathlib.Path) -> Optional[str]:
    if raw_value is None:
        return None
    s = str(raw_value).strip()
    if not s:
        return None
    p = pathlib.Path(s)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = raw_dir / s
    if candidate.exists():
        return str(candidate.resolve())
    return None


def _load_raw_records(raw_dir: pathlib.Path) -> List[Tuple[pathlib.Path, Dict[str, Any]]]:
    if not raw_dir.exists():
        return []

    pairs: List[Tuple[pathlib.Path, Dict[str, Any]]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        rows: List[Dict[str, Any]] = []
        try:
            if ext == ".jsonl":
                rows = _parse_jsonl(path)
            elif ext == ".json":
                rows = _parse_json(path)
            elif ext == ".csv":
                rows = _parse_csv(path, delimiter=",")
            elif ext == ".tsv":
                rows = _parse_csv(path, delimiter="\t")
        except Exception:
            rows = []

        for row in rows:
            pairs.append((path, row))

    return pairs


def _partition_if_needed(splits: Dict[str, List[UnifiedSample]], seed: int) -> Dict[str, List[UnifiedSample]]:
    if any(len(splits[s]) > 0 for s in ("train", "val", "test")):
        return splits

    all_rows = splits.get("all", [])
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    n = len(all_rows)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    splits["train"] = all_rows[:n_train]
    splits["val"] = all_rows[n_train : n_train + n_val]
    splits["test"] = all_rows[n_train + n_val :]
    return splits


def _subset_rows(rows: List[UnifiedSample], target: int, seed: int) -> List[UnifiedSample]:
    if target <= 0 or len(rows) <= target:
        return rows
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:target])
    return [rows[i] for i in idxs]


def _renumber(rows: List[UnifiedSample], dataset: str, split: str) -> List[UnifiedSample]:
    out = []
    for i, r in enumerate(rows):
        r = UnifiedSample(
            id=f"{dataset}-{split}-{i:06d}",
            dataset=r.dataset,
            split=split,
            question=r.question,
            choices=r.choices,
            answer_idx=r.answer_idx,
            image_path=r.image_path,
            difficulty=r.difficulty,
            metadata=r.metadata,
        )
        out.append(r)
    return out


def prepare_dataset(
    dataset_name: str,
    raw_root: pathlib.Path,
    processed_root: pathlib.Path,
    subset_sizes: Dict[str, int],
    seed: int = 3407,
) -> Dict[str, Any]:
    aliases = DATASET_ALIASES.get(dataset_name, GENERIC_ALIASES)
    raw_dir = raw_root / dataset_name
    out_dir = processed_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "dataset": dataset_name,
        "raw_dir": str(raw_dir),
        "processed_dir": str(out_dir),
        "status": "ok",
        "counts": {},
        "warnings": [],
    }

    if not raw_dir.exists():
        result["status"] = "missing_raw_dir"
        result["warnings"].append(f"Raw directory not found: {raw_dir}")
        return result

    raw_pairs = _load_raw_records(raw_dir)
    if not raw_pairs:
        result["status"] = "missing_raw_files"
        result["warnings"].append(f"No parseable files under: {raw_dir}")
        return result

    splits: Dict[str, List[UnifiedSample]] = {"train": [], "val": [], "test": [], "all": []}
    dropped = 0

    for path, rec in raw_pairs:
        split = _normalize_split(_get_first(rec, aliases["split"] + GENERIC_ALIASES["split"]))
        if split is None:
            split = _split_from_path(path)
        if split is None:
            split = "all"

        question = _get_first(rec, aliases["question"] + GENERIC_ALIASES["question"])
        question = "" if question is None else str(question).strip()
        choices = _extract_choices(rec, aliases)
        answer_idx = _extract_answer_idx(rec, aliases, choices)
        difficulty = _get_first(rec, aliases["difficulty"] + GENERIC_ALIASES["difficulty"])
        if difficulty is not None:
            difficulty = str(difficulty)
        image_raw = _get_first(rec, aliases["image"] + GENERIC_ALIASES["image"])
        image_path = _resolve_image_path(image_raw, raw_dir)

        if not question:
            dropped += 1
            continue

        sample = UnifiedSample(
            id="",
            dataset=dataset_name,
            split=split,
            question=question,
            choices=choices,
            answer_idx=answer_idx,
            image_path=image_path,
            difficulty=difficulty,
            metadata={
                "source_file": str(path),
            },
        )
        splits.setdefault(split, []).append(sample)

    splits = _partition_if_needed(splits, seed=seed)

    for split in ("train", "val", "test"):
        target = int(subset_sizes.get(split, 0))
        rows = splits.get(split, [])
        if target > 0 and len(rows) < target:
            result["warnings"].append(
                f"{dataset_name}:{split} has only {len(rows)} samples < target {target}, using all available"
            )
        sampled = _subset_rows(rows, target, seed=seed + len(split))
        sampled = _renumber(sampled, dataset_name, split)
        write_jsonl(out_dir / f"{split}.jsonl", [r.to_dict() for r in sampled])
        result["counts"][split] = len(sampled)

    result["dropped_no_question"] = dropped
    write_json(out_dir / "prepare_summary.json", result)
    return result


def prepare_all_datasets(
    raw_root: pathlib.Path,
    processed_root: pathlib.Path,
    subsets_cfg: Dict[str, Dict[str, int]],
    seed: int = 3407,
) -> List[Dict[str, Any]]:
    reports = []
    for dataset_name, subset_sizes in subsets_cfg.items():
        report = prepare_dataset(
            dataset_name=dataset_name,
            raw_root=raw_root,
            processed_root=processed_root,
            subset_sizes=subset_sizes,
            seed=seed,
        )
        reports.append(report)
    return reports
