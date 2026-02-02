from __future__ import annotations

import argparse
from pathlib import Path

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.profile import ProfileBy, default_profile_path
from baseball.viz import (
    VizError,
    build_count_policy_grid,
    list_pitch_types,
    load_profile,
    render_count_policy_svg,
    render_zone_heatmaps_svg,
)


def add_viz_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "viz",
        help="Generate SVG visuals (pitcher profiles, zone heatmaps) from profile parquet outputs.",
    )
    add_artifact_root_arg(p)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--profile-parquet",
        type=str,
        default="",
        help="Path to a profile parquet file produced by `python -m baseball profile`.",
    )
    src.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id under runs/ (uses the default profile path).",
    )

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
        default="pitcher_situation_prev_outcome",
        help="Only used with --run-id to locate the profile parquet.",
    )
    p.add_argument("--split", choices=["train", "valid"], default="train", help="Only used with --run-id.")

    p.add_argument(
        "--pitcher",
        type=str,
        required=True,
        help="Pitcher identifier from the profile parquet. Usually MLBAM id string (column `pitcher`).",
    )
    p.add_argument(
        "--pitch-types",
        type=str,
        default="",
        help="Comma-separated pitch type tokens to render zone heatmaps for (e.g. FF,SL). Default: top 2 by predicted usage.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="docs/assets",
        help="Output directory for generated SVGs (default: docs/assets).",
    )
    p.set_defaults(func=cmd_viz)


def cmd_viz(args: argparse.Namespace) -> None:
    if args.profile_parquet:
        profile_path = Path(str(args.profile_parquet))
    else:
        paths = get_paths(args.artifact_root)
        ensure_dirs(paths)
        by: ProfileBy = args.by
        profile_path = default_profile_path(paths, run_id=str(args.run_id), split=str(args.split), by=by)

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_profile(profile_path)
    pitch_types = list_pitch_types(df)

    # Filter to the requested pitcher.
    try:
        df_pitcher = df.filter((df["pitcher"] == str(args.pitcher)) | (df["pitcher_id"] == int(str(args.pitcher))))  # type: ignore[arg-type]
    except Exception:
        df_pitcher = df.filter(df["pitcher"] == str(args.pitcher))
    if df_pitcher.is_empty():
        raise VizError(f"Pitcher not found in profile parquet: {args.pitcher!r}")

    # Pitch types to show in zone heatmaps.
    tokens: list[str]
    if str(args.pitch_types).strip():
        tokens = [t.strip() for t in str(args.pitch_types).split(",") if t.strip()]
    else:
        # Default: top 2 by predicted usage, weighted by n.
        n = df_pitcher["n"].to_numpy()
        scores: list[tuple[str, float]] = []
        for pt in pitch_types:
            if pt.token in {"OOV"}:
                continue
            w = float((df_pitcher[pt.pred_col].to_numpy() * n).sum() / max(1.0, float(n.sum())))
            scores.append((pt.token, w))
        scores.sort(key=lambda x: x[1], reverse=True)
        tokens = [t for t, _ in scores[:2]]

    pitcher_label = str(args.pitcher)
    title = f"Pitcher profile — {pitcher_label}"

    # Count policy grid (pred vs empirical).
    count_policy = build_count_policy_grid(df_pitcher, pitch_types)
    svg_count = render_count_policy_svg(title=title, count_policy=count_policy)
    out_count = out_dir / f"profile_{pitcher_label}_count_policy.svg"
    out_count.write_text(svg_count, encoding="utf-8")

    # Zone heatmaps for selected pitch types.
    svg_zone = render_zone_heatmaps_svg(
        title=f"Zone heatmaps — {pitcher_label}",
        df_pitcher=df_pitcher,
        pitch_types=pitch_types,
        pitch_tokens=tokens,
    )
    out_zone = out_dir / f"profile_{pitcher_label}_zone_heatmaps.svg"
    out_zone.write_text(svg_zone, encoding="utf-8")

    print(str(out_count))
    print(str(out_zone))
