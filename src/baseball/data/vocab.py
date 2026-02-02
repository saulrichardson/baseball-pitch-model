from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Vocab:
    """
    Integer vocabulary mapping.

    Conventions:
    - 0 is reserved for OOV / unknown.
    - IDs are contiguous and start at 1.
    """

    token_to_id: dict[str, int]

    @property
    def size(self) -> int:
        return 1 + len(self.token_to_id)

    def encode(self, token: str | None) -> int:
        if token is None:
            return 0
        return self.token_to_id.get(str(token), 0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid vocab file: {path}")
        return cls(token_to_id={str(k): int(v) for k, v in obj.items()})


def build_vocab_from_counts(counts: pl.DataFrame, min_count: int) -> Vocab:
    """
    `counts` must have columns: ["token", "count"].
    """

    if "token" not in counts.columns or "count" not in counts.columns:
        raise ValueError("counts must have ['token', 'count'] columns")

    filtered = counts.filter(pl.col("count") >= min_count).sort("count", descending=True)
    token_to_id: dict[str, int] = {}
    next_id = 1
    for token in filtered.get_column("token").to_list():
        token_to_id[str(token)] = next_id
        next_id += 1
    return Vocab(token_to_id=token_to_id)

