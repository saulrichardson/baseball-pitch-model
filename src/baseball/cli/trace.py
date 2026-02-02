from __future__ import annotations

import argparse
from pathlib import Path

from baseball.trace import render_trace_file


def add_trace_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("trace", help="Render pitch-by-pitch JSONL traces to SVG/HTML.")
    p.add_argument("--events", type=str, required=True, help="Input trace JSONL produced by `simulate --events-out`.")
    p.add_argument("--out", type=str, required=True, help="Output path (.svg or .html).")
    p.add_argument("--format", choices=["svg", "html"], default="html", help="Output format.")
    p.add_argument("--game-pk", type=int, default=None, help="If trace contains multiple games, select one.")
    p.add_argument(
        "--at-bat",
        type=int,
        default=None,
        help="For --format svg: which at_bat_number to render (default: first in trace).",
    )
    p.add_argument("--max-at-bats", type=int, default=10, help="For --format html: render up to N at-bats.")
    p.add_argument("--max-pitches", type=int, default=12, help="Max pitches per at-bat in rendered output.")
    p.set_defaults(func=cmd_trace)


def cmd_trace(args: argparse.Namespace) -> None:
    render_trace_file(
        trace_path=Path(str(args.events)),
        out_path=Path(str(args.out)),
        fmt=str(args.format),
        game_pk=(int(args.game_pk) if args.game_pk is not None else None),
        at_bat_number=(int(args.at_bat) if args.at_bat is not None else None),
        max_at_bats=int(args.max_at_bats),
        max_pitches_per_ab=int(args.max_pitches),
    )

