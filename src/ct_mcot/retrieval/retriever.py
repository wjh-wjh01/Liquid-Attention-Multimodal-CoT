from __future__ import annotations

import math
from dataclasses import dataclass

from .entity_linking import EntityLinker
from .knowledge_store import KnowledgeNode, KnowledgeStore


@dataclass
class RetrievalResult:
    node: KnowledgeNode
    score: float
    reason: str


class HybridRetriever:
    def __init__(self, store: KnowledgeStore, top_k: int = 8, expansion_hops: int = 1):
        self.store = store
        self.top_k = top_k
        self.expansion_hops = expansion_hops
        self.linker = EntityLinker()

    def retrieve(
        self,
        question: str,
        choices: list[str] | None = None,
        visual_tags: list[str] | None = None,
    ) -> list[RetrievalResult]:
        query = " ".join([question, *(choices or []), *(visual_tags or [])])
        mentions = self.linker.extract(query)
        lexical = self.store.lexical_candidates(query)
        expanded = self.store.expand([node.id for node in lexical], self.expansion_hops)
        candidates = {node.id: node for node in [*lexical, *expanded]}
        results = []
        for node in candidates.values():
            overlap = _token_overlap(query, " ".join([node.text, *node.aliases]))
            mention_bonus = 0.2 if any(m.text.lower() in node.text.lower() for m in mentions) else 0.0
            relation_penalty = 0.05 * _min_relation_distance(node, lexical)
            score = overlap + mention_bonus - relation_penalty
            results.append(RetrievalResult(node=node, score=score, reason="hybrid_lexical_relation"))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: self.top_k]


def _token_overlap(a: str, b: str) -> float:
    aa = {x.lower() for x in a.split() if len(x) > 2}
    bb = {x.lower() for x in b.split() if len(x) > 2}
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / math.sqrt(len(aa) * len(bb))


def _min_relation_distance(node: KnowledgeNode, roots: list[KnowledgeNode]) -> int:
    if node.id in {root.id for root in roots}:
        return 0
    return 1
