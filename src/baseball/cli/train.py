from __future__ import annotations

import argparse

from baseball.cli.common import add_artifact_root_arg
from baseball.config import ensure_dirs, get_paths
from baseball.training.train import train


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("train", help="Train a model.")
    add_artifact_root_arg(p)
    p.add_argument(
        "--model",
        choices=[
            "baseline_mlp",
            "transformer_mdn",
            "transformer_mdn_mt",
            "transformer_mdn_v2",
            "transformer_mdn_state",
            "transformer_mdn_state_mt",
        ],
        default="transformer_mdn",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run directory name (useful for Slurm job arrays).",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4096, help="Microbatch size (per optimizer step uses grad-accum).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch-size*grad-accum).")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision on CUDA.")
    p.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16", help="AMP dtype (older GPUs should use fp16).")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (keep low for RAM efficiency).")
    p.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch batches per worker (only if num-workers>0).")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Stream parquet instead of loading full dataset in RAM.")
    p.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction, default=False, help="Use torch.compile (optional).")
    p.add_argument(
        "--grad-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing for transformer layers (saves GPU memory, costs compute).",
    )
    p.add_argument(
        "--state-corrupt-p",
        type=float,
        default=0.0,
        help=(
            "Training-time robustness: with probability p, jitter balls/strikes in the pre-pitch state features. "
            "Useful for improving open-loop rollout stability."
        ),
    )
    p.add_argument(
        "--state-corrupt-max-delta",
        type=int,
        default=1,
        help="Max absolute integer delta for state corruption (applied to balls/strikes).",
    )
    p.add_argument("--eval-max-batches", type=int, default=200, help="Limit eval batches per epoch (0 = full eval).")
    p.add_argument("--loc-weight", type=float, default=0.3, help="Weight for location NLL loss.")
    p.add_argument("--state-weight", type=float, default=0.2, help="Weight for next-state loss (state models only).")
    p.add_argument("--desc-weight", type=float, default=0.1, help="Weight for description CE loss (multi-task models only).")

    # Transformer scaling knobs (used by transformer_mdn* / transformer_mdn_v2 / transformer_mdn_state*)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--layers", dest="num_layers", type=int, default=None)
    p.add_argument("--mdn-components", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.set_defaults(func=cmd_train)


def cmd_train(args: argparse.Namespace) -> None:
    paths = get_paths(args.artifact_root)
    ensure_dirs(paths)
    eval_max_batches = int(args.eval_max_batches)
    if eval_max_batches <= 0:
        eval_max_batches = None
    train(
        paths=paths,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        run_id=args.run_id,
        grad_accum=args.grad_accum,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        streaming=args.streaming,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        mdn_components=args.mdn_components,
        dropout=args.dropout,
        compile_model=args.compile_model,
        grad_checkpointing=args.grad_checkpointing,
        state_corrupt_p=args.state_corrupt_p,
        state_corrupt_max_delta=args.state_corrupt_max_delta,
        eval_max_batches=eval_max_batches,
        loc_weight=args.loc_weight,
        state_weight=args.state_weight,
        desc_weight=args.desc_weight,
    )
