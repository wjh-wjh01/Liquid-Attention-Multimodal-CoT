from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KnowledgeNode:
    id: str
    text: str
    aliases: list[str] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class KnowledgeStore:
    def __init__(self, nodes: list[KnowledgeNode]):
        self.nodes = nodes
        self.by_id = {node.id: node for node in nodes}
        self.alias_index: dict[str, set[str]] = {}
        for node in nodes:
            for alias in [node.text, *node.aliases]:
                self.alias_index.setdefault(alias.lower(), set()).add(node.id)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "KnowledgeStore":
        nodes = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    nodes.append(
                        KnowledgeNode(
                            id=str(row["id"]),
                            text=row["text"],
                            aliases=row.get("aliases", []),
                            relations=[tuple(x) for x in row.get("relations", [])],
                            metadata=row.get("metadata", {}),
                        )
                    )
        return cls(nodes)

    def lexical_candidates(self, query: str) -> list[KnowledgeNode]:
        query_lower = query.lower()
        ids: set[str] = set()
        for alias, node_ids in self.alias_index.items():
            if alias in query_lower or any(part in query_lower for part in alias.split()):
                ids.update(node_ids)
        return [self.by_id[item] for item in ids]

    def expand(self, node_ids: list[str], hops: int = 1) -> list[KnowledgeNode]:
        seen = set(node_ids)
        frontier = set(node_ids)
        for _ in range(hops):
            nxt: set[str] = set()
            for node_id in frontier:
                node = self.by_id.get(node_id)
                if node is None:
                    continue
                for _, _, dst in node.relations:
                    if dst not in seen and dst in self.by_id:
                        nxt.add(dst)
            seen.update(nxt)
            frontier = nxt
        return [self.by_id[item] for item in seen if item in self.by_id]
