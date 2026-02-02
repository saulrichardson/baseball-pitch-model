from __future__ import annotations

import argparse

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.simulate import simulate, write_json


def add_simulate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("simulate", help="Replay or roll forward through held-out games.")
    add_artifact_root_arg(p)
    p.add_argument("--run-id", type=str, required=True, help="Run directory name under runs/ (must have best.pt).")
    p.add_argument("--split", choices=["train", "valid"], default="valid")
    p.add_argument("--mode", choices=["replay", "rollout"], default="replay")
    p.add_argument(
        "--count-mode",
        choices=["heads", "clamp", "rules", "constrained"],
        default="heads",
        help=(
            "How to update balls/strikes/PA-end in rollout mode. "
            "`heads` uses the model's state heads. "
            "`clamp` decodes balls/strikes under simple within-PA constraints (no decreases, +1 max, only one increments). "
            "`rules` deterministically updates from predicted `description`. "
            "`constrained` chooses a consistent transition using both heads + rules."
        ),
    )
    p.add_argument("--game-pk", type=int, action="append", default=None, help="Restrict to a specific game_pk (can repeat).")
    p.add_argument("--max-games", type=int, default=5, help="If --game-pk not provided, simulate up to N games.")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--out", type=str, default="-", help="Output JSON path (default: stdout).")
    p.set_defaults(func=cmd_simulate)


def cmd_simulate(args: argparse.Namespace) -> None:
    paths = get_paths(args.artifact_root)
    ensure_dirs(paths)
    payload = simulate(
        paths,
        run_id=args.run_id,
        split=args.split,
        mode=args.mode,
        count_mode=args.count_mode,
        game_pks=args.game_pk,
        max_games=args.max_games,
        device=args.device,
    )
    write_json(args.out, payload)
