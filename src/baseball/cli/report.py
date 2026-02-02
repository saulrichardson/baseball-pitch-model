from __future__ import annotations

import argparse
from pathlib import Path

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.report import generate_report


def add_report_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("report", help="Generate a model report (metrics + baselines + calibration).")
    add_artifact_root_arg(p)
    p.add_argument("--run-id", type=str, required=True, help="Run directory name under runs/ (must have best.pt).")
    p.add_argument("--split", choices=["train", "valid"], default="valid", help="Evaluation split for the report.")
    p.add_argument(
        "--baseline-split",
        choices=["train", "valid"],
        default="train",
        help="Split used to fit empirical baselines (default: train).",
    )
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: runs/<run_id>/report/).")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda (used for model evaluation).")
    p.add_argument("--batch-size", type=int, default=4096, help="Batch size for model evaluation.")
    p.add_argument("--baseline-batch-size", type=int, default=8192, help="Batch size for baseline fit/eval.")
    p.add_argument("--baseline-alpha", type=float, default=1.0, help="Additive smoothing alpha for baselines.")
    p.add_argument("--calibration-bins", type=int, default=15, help="Number of calibration bins for ECE.")
    p.set_defaults(func=cmd_report)


def cmd_report(args: argparse.Namespace) -> None:
    paths = get_paths(args.artifact_root)
    ensure_dirs(paths)
    out_dir = Path(args.out_dir) if args.out_dir else None
    payload = generate_report(
        paths,
        run_id=args.run_id,
        split=args.split,
        baseline_split=args.baseline_split,
        out_dir=out_dir,
        device=args.device,
        batch_size=args.batch_size,
        baseline_batch_size=args.baseline_batch_size,
        baseline_alpha=args.baseline_alpha,
        calibration_bins=args.calibration_bins,
    )
    # Keep stdout minimal; report artifacts are written to disk.
    print(f"Wrote report for run_id={payload['run_id']} to: {out_dir or (paths.runs / args.run_id / 'report')}")

