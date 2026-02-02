from __future__ import annotations

import argparse

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.data.download import download_statcast_range


def add_download_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("download", help="Download Statcast pitch-level data.")
    add_artifact_root_arg(p)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--chunk-days", type=int, default=7, help="Download granularity (days per chunk).")
    p.set_defaults(func=cmd_download)


def cmd_download(args: argparse.Namespace) -> None:
    paths = get_paths(args.artifact_root)
    ensure_dirs(paths)
    download_statcast_range(
        start=args.start,
        end=args.end,
        out_dir=paths.data_raw,
        chunk_days=args.chunk_days,
    )

