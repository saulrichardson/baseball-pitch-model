from __future__ import annotations

import argparse


def add_artifact_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=None,
        help="Root directory for data/models/runs (defaults to $BASEBALL_ARTIFACT_ROOT or $VAST).",
    )

