from __future__ import annotations

import argparse
from pathlib import Path

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.profile import ProfileBy, build_pitcher_profiles, default_profile_path, write_profile


def add_profile_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "profile",
        help="Build per-pitcher conditional profiles (pitch mix + location) from a trained model.",
    )
    add_artifact_root_arg(p)
    p.add_argument("--run-id", type=str, required=True, help="Run directory name under runs/ (must have best.pt).")
    p.add_argument("--split", choices=["train", "valid"], default="valid")
    p.add_argument(
        "--by",
        choices=[
            "pitcher",
            "pitcher_count",
            "pitcher_count_prev",
            "pitcher_count_prev_outcome",
            "pitcher_situation",
            "pitcher_situation_prev",
            "pitcher_situation_prev_outcome",
            "pitcher_situation_batter_cluster",
            "pitcher_situation_batter_cluster_prev",
            "pitcher_situation_batter_cluster_prev_outcome",
        ],
        default="pitcher_count",
        help=(
            "Aggregation key: pitcher only, pitcher+count, pitcher+count+previous pitch type, "
            "pitcher+count+previous pitch type+previous pitch outcome (Statcast description), "
            "or pitcher+count+game situation (optionally with previous pitch and/or batter clusters)."
        ),
    )
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--min-n", type=int, default=25, help="Minimum pitches per group to keep in the output.")
    p.add_argument(
        "--batter-clusters",
        type=int,
        default=32,
        help=(
            "Only used when --by includes batter_cluster: cluster batters using the model's learned batter embedding "
            "(OOV is reserved as cluster 0)."
        ),
    )
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output parquet path. Default: runs/<run_id>/pitcher_profiles_<by>_<split>.parquet",
    )
    p.set_defaults(func=cmd_profile)


def cmd_profile(args: argparse.Namespace) -> None:
    paths = get_paths(args.artifact_root)
    ensure_dirs(paths)

    by: ProfileBy = args.by
    df = build_pitcher_profiles(
        paths,
        run_id=str(args.run_id),
        split=str(args.split),
        by=by,
        batch_size=int(args.batch_size),
        min_n=int(args.min_n),
        batter_clusters=int(args.batter_clusters),
        device=str(args.device),
    )

    out_path = Path(args.out) if args.out else default_profile_path(paths, run_id=str(args.run_id), split=str(args.split), by=by)
    write_profile(df, out_path)
    print(str(out_path))
