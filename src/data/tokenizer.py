from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from typing import Iterable, List


class SimpleTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self) -> None:
        self.token_to_id = {self.PAD: 0, self.UNK: 1}
        self.id_to_token = {0: self.PAD, 1: self.UNK}

    @staticmethod
    def _basic_tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text.lower())

    def fit(self, texts: Iterable[str], max_vocab_size: int = 20000, min_freq: int = 1) -> None:
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(self._basic_tokenize(str(t)))

        kept = [tok for tok, c in counter.most_common(max_vocab_size) if c >= min_freq]
        self.token_to_id = {self.PAD: 0, self.UNK: 1}
        for tok in kept:
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text: str, max_len: int) -> List[int]:
        toks = self._basic_tokenize(str(text))[:max_len]
        ids = [self.token_to_id.get(tok, self.token_to_id[self.UNK]) for tok in toks]
        if len(ids) < max_len:
            ids.extend([self.token_to_id[self.PAD]] * (max_len - len(ids)))
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = [self.id_to_token.get(i, self.UNK) for i in ids if i != self.token_to_id[self.PAD]]
        return " ".join(toks)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str | pathlib.Path) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "SimpleTokenizer":
        p = pathlib.Path(path)
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        tok = cls()
        tok.token_to_id = {k: int(v) for k, v in obj["token_to_id"].items()}
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        return tok
