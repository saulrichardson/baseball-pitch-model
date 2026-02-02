from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_prepared(self) -> Path:
        return self.root / "data" / "prepared"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def models_exported(self) -> Path:
        return self.root / "models" / "exported"


def get_paths(artifact_root: str | None) -> Paths:
    """
    Resolve the artifact root.

    This project intentionally does not silently fall back to local directories.
    If you are not on Greene (no $VAST), pass --artifact-root explicitly.
    """

    root = artifact_root or os.environ.get("BASEBALL_ARTIFACT_ROOT") or os.environ.get("VAST")
    if not root:
        raise ConfigError(
            "Missing artifact root. Set BASEBALL_ARTIFACT_ROOT or VAST, or pass --artifact-root."
        )
    return Paths(root=Path(root).expanduser().resolve())


def ensure_dirs(paths: Paths) -> None:
    for p in [paths.data_raw, paths.data_prepared, paths.runs, paths.models_exported]:
        p.mkdir(parents=True, exist_ok=True)

