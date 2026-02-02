from __future__ import annotations

import argparse

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.data.prepare import prepare_dataset


def add_prepare_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("prepare", help="Prepare modeling dataset from raw Statcast downloads.")
    add_artifact_root_arg(p)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--history-len", type=int, default=8, help="Number of previous pitches within PA.")
    p.add_argument("--valid-frac", type=float, default=0.1, help="Validation fraction (time-based split).")
    p.add_argument("--min-pitch-type-count", type=int, default=50, help="Min count to include a pitch_type token in vocab.")
    p.add_argument("--min-description-count", type=int, default=25, help="Min count to include a description token in vocab.")
    p.add_argument("--min-pitcher-count", type=int, default=50, help="Min count to include a pitcher ID in vocab.")
    p.add_argument("--min-batter-count", type=int, default=50, help="Min count to include a batter ID in vocab.")
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing prepared dataset directory.",
    )
    p.set_defaults(func=cmd_prepare)


def cmd_prepare(args: argparse.Namespace) -> None:
    paths = get_paths(args.artifact_root)
    ensure_dirs(paths)
    prepare_dataset(
        raw_dir=paths.data_raw,
        prepared_dir=paths.data_prepared,
        start=args.start,
        end=args.end,
        history_len=args.history_len,
        valid_frac=args.valid_frac,
        min_pitch_type_count=args.min_pitch_type_count,
        min_description_count=args.min_description_count,
        min_pitcher_count=args.min_pitcher_count,
        min_batter_count=args.min_batter_count,
        overwrite=args.overwrite,
    )
