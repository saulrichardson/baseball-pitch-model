from __future__ import annotations

from pathlib import Path


def latest_completed_run(runs_root: Path) -> Path:
    if not runs_root.exists():
        raise FileNotFoundError(f"No runs found under: {runs_root}")

    candidates: list[Path] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "checkpoints" / "best.pt").exists():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No completed runs (with best.pt) found under: {runs_root}")

    return sorted(candidates)[-1]


def get_run_dir(runs_root: Path, run_id: str) -> Path:
    run_id = run_id.strip()
    if not run_id:
        raise ValueError("--run-id must be non-empty")
    run_dir = runs_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    if not (run_dir / "checkpoints" / "best.pt").exists():
        raise FileNotFoundError(f"Run is not completed (missing best.pt): {run_dir}")
    return run_dir


def resolve_run_dir(runs_root: Path, run_id: str | None) -> Path:
    return get_run_dir(runs_root, run_id) if run_id is not None else latest_completed_run(runs_root)
