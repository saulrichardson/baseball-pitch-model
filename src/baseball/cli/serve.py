from __future__ import annotations

import argparse

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.serve.api import run_server


def add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("serve", help="Run inference API.")
    add_artifact_root_arg(p)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)


def cmd_serve(args: argparse.Namespace) -> None:
    paths = get_paths(args.artifact_root)
    ensure_dirs(paths)
    run_server(paths=paths, host=args.host, port=args.port)

