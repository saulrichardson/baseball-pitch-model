from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreparedArtifacts:
    prepared_dir: Path
    meta: dict[str, Any]

    def _data_path(self, key: str, default: str) -> Path:
        rel = self.meta.get(key)
        if isinstance(rel, str) and rel:
            return self.prepared_dir / rel
        return self.prepared_dir / default

    @property
    def train_path(self) -> Path:
        # New (scale) format uses sharded directories: meta['train_data'] == "train"
        return self._data_path("train_data", "train.parquet")

    @property
    def valid_path(self) -> Path:
        return self._data_path("valid_data", "valid.parquet")

    @property
    def vocabs_dir(self) -> Path:
        return self.prepared_dir / "vocabs"


def load_prepared(prepared_dir: Path) -> PreparedArtifacts:
    meta_path = prepared_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Prepared meta not found: {meta_path}. Run `python -m baseball prepare`.")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid meta.json: {meta_path}")
    return PreparedArtifacts(prepared_dir=prepared_dir, meta=meta)
