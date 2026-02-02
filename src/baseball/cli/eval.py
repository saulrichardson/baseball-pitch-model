from __future__ import annotations

import argparse

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.training.eval import evaluate_latest


def add_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("eval", help="Evaluate a trained model.")
    add_artifact_root_arg(p)
    p.add_argument("--split", choices=["train", "valid"], default="valid")
    p.add_argument("--run-id", type=str, default=None, help="Evaluate a specific run (directory name under runs/).")
    p.set_defaults(func=cmd_eval)


def cmd_eval(args: argparse.Namespace) -> None:
    paths = get_paths(args.artifact_root)
    ensure_dirs(paths)
    evaluate_latest(paths=paths, split=args.split, run_id=args.run_id)
