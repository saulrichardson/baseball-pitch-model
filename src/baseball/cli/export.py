from __future__ import annotations

import argparse

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.training.export import export_latest


def add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("export", help="Export latest trained model bundle.")
    add_artifact_root_arg(p)
    p.add_argument("--run-id", type=str, default=None, help="Export a specific run (directory name under runs/).")
    p.set_defaults(func=cmd_export)


def cmd_export(args: argparse.Namespace) -> None:
    paths = get_paths(args.artifact_root)
    ensure_dirs(paths)
    out = export_latest(paths, run_id=args.run_id)
    print(str(out))
