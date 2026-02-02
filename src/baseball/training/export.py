from __future__ import annotations

import shutil
from pathlib import Path

from baseball.config import Paths
from baseball.training.runs import resolve_run_dir


def export_latest(paths: Paths, *, run_id: str | None = None) -> Path:
    """
    Export the latest completed run into a self-contained bundle directory under
    `paths.models_exported/<run_id>/`.
    """

    run_dir = resolve_run_dir(paths.runs, run_id=run_id)
    out_dir = paths.models_exported / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=False)

    # Model checkpoint
    src_ckpt = run_dir / "checkpoints" / "best.pt"
    shutil.copy2(src_ckpt, out_dir / "best.pt")

    # Run config + metrics
    for name in ["config.json", "metrics.jsonl"]:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # Prepared dataset metadata + vocabs (required for encoding + normalization).
    prepared_dir = paths.data_prepared
    for name in ["meta.json"]:
        src = prepared_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    vocab_src = prepared_dir / "vocabs"
    if vocab_src.exists():
        shutil.copytree(vocab_src, out_dir / "vocabs")

    return out_dir
