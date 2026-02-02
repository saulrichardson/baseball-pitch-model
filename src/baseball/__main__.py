from __future__ import annotations

import argparse

from baseball.cli.download import add_download_parser
from baseball.cli.eval import add_eval_parser
from baseball.cli.export import add_export_parser
from baseball.cli.prepare import add_prepare_parser
from baseball.cli.profile import add_profile_parser
from baseball.cli.report import add_report_parser
from baseball.cli.simulate import add_simulate_parser
from baseball.cli.serve import add_serve_parser
from baseball.cli.train import add_train_parser
from baseball.cli.viz import add_viz_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m baseball",
        description="Pitch type + location modeling pipeline (Statcast).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_download_parser(subparsers)
    add_prepare_parser(subparsers)
    add_train_parser(subparsers)
    add_eval_parser(subparsers)
    add_report_parser(subparsers)
    add_export_parser(subparsers)
    add_profile_parser(subparsers)
    add_simulate_parser(subparsers)
    add_serve_parser(subparsers)
    add_viz_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
