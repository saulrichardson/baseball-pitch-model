from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseball.config import Paths
from baseball.data.vocab import Vocab
from baseball.logging import RunLogger
from baseball.training.dataset import PitchDataset
from baseball.training.io import load_prepared
from baseball.training.mdn import mdn_mean, mdn_nll
from baseball.training.models import (
    BaselineMLP,
    ModelConfig,
    TransformerMDN,
    TransformerMDNMT,
    TransformerMDNState,
    TransformerMDNStateMT,
    TransformerMDNV2,
)
from baseball.training.streaming import PitchParquetBatchIterable


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _new_run_dir(runs_root: Path) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    # Avoid collisions for rapid iterations.
    suffix = os.urandom(2).hex()
    run_dir = runs_root / f"{run_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _validate_run_id(run_id: str) -> str:
    run_id = run_id.strip()
    if not run_id:
        raise ValueError("--run-id must be a non-empty string")
    if "/" in run_id or "\\" in run_id:
        raise ValueError("--run-id must not contain path separators")
    if run_id in {".", ".."}:
        raise ValueError("--run-id is not allowed")
    return run_id


def _new_run_dir_with_id(runs_root: Path, run_id: str) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = _validate_run_id(run_id)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_vocabs(prepared_dir: Path, meta: dict[str, Any]) -> dict[str, Vocab]:
    vocab_paths = meta.get("vocab_paths")
    if not isinstance(vocab_paths, dict):
        raise ValueError("meta.json missing vocab_paths")

    vocabs: dict[str, Vocab] = {}
    for name in ["pitch_type", "pitcher", "batter"]:
        rel = vocab_paths.get(name)
        if not isinstance(rel, str):
            raise ValueError(f"meta.json missing vocab_paths['{name}']")
        path = prepared_dir / rel
        vocabs[name] = Vocab.load(path)
    # Optional outcome vocabulary (schema_version >= 4).
    if "description" in vocab_paths:
        rel = vocab_paths.get("description")
        if not isinstance(rel, str):
            raise ValueError("meta.json has vocab_paths['description'] but it is not a string")
        vocabs["description"] = Vocab.load(prepared_dir / rel)
    return vocabs


def _gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
    """
    Diagonal Gaussian NLL for baseline: y,mu,log_s are [B,2]
    """

    s = torch.exp(log_s).clamp(min=1e-4)
    z = (y - mu) / s
    # log(2*pi) term; average over 2 dims
    nll = 0.5 * (z**2 + 2.0 * torch.log(s) + np.log(2.0 * np.pi))
    return nll.sum(dim=-1).mean()


def _masked_cross_entropy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked cross entropy for per-row discrete targets.

    logits: [B,C]
    y: [B]
    mask: [B] float or bool (1 = include)
    """

    if mask.dtype != torch.float32 and mask.dtype != torch.float64:
        mask = mask.float()
    per = F.cross_entropy(logits, y, reduction="none")  # [B]
    denom = mask.sum().clamp(min=1.0)
    return (per * mask).sum() / denom


def _state_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Multi-task next-state loss (masked by y_has_next).

    Expects:
      batch['y_has_next'] in {0,1}
      batch targets present for each head
      out contains corresponding logits keys.
    """

    mask = batch["y_has_next"].float()
    losses = [
        _masked_cross_entropy(out["pa_end_logits"], batch["y_pa_end"], mask),
        _masked_cross_entropy(out["next_balls_logits"], batch["y_next_balls"], mask),
        _masked_cross_entropy(out["next_strikes_logits"], batch["y_next_strikes"], mask),
        _masked_cross_entropy(out["next_outs_when_up_logits"], batch["y_next_outs_when_up"], mask),
        _masked_cross_entropy(out["next_on_1b_logits"], batch["y_next_on_1b_occ"], mask),
        _masked_cross_entropy(out["next_on_2b_logits"], batch["y_next_on_2b_occ"], mask),
        _masked_cross_entropy(out["next_on_3b_logits"], batch["y_next_on_3b_occ"], mask),
        _masked_cross_entropy(out["next_inning_topbot_logits"], batch["y_next_inning_topbot_id"], mask),
        _masked_cross_entropy(out["inning_delta_logits"], batch["y_inning_delta"], mask),
        _masked_cross_entropy(out["score_diff_delta_logits"], batch["y_score_diff_delta_id"], mask),
    ]
    return torch.stack(losses).mean()


def _assert_finite_metrics(metrics: dict[str, float], *, split: str) -> None:
    bad = {k: v for k, v in metrics.items() if isinstance(v, float) and not np.isfinite(v)}
    if bad:
        raise RuntimeError(f"Non-finite metrics on split='{split}': {bad}")


def _assert_finite_params(model: torch.nn.Module) -> None:
    for name, p in model.named_parameters():
        if p is None:
            continue
        if not torch.isfinite(p).all():
            raise RuntimeError(f"Non-finite parameter detected: {name}")


@torch.no_grad()
def _evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    plate_x_mean: float,
    plate_x_std: float,
    plate_z_mean: float,
    plate_z_std: float,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()

    total = 0
    correct = 0
    top3 = 0

    ce_sum = 0.0
    nll_loc_sum = 0.0
    mse_loc_sum = 0.0
    mse_loc_ft_sum = 0.0

    desc_total = 0
    desc_correct = 0
    desc_top3 = 0
    desc_ce_sum = 0.0

    state_total = 0.0
    state_loss_sum = 0.0
    state_correct: dict[str, int] = {}

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        y_type = batch["y_type"]
        y_loc = batch["y_loc"]

        out = model(batch)
        type_logits = out["type_logits"]
        ce = F.cross_entropy(type_logits, y_type)
        pred = type_logits.argmax(dim=-1)
        correct += int((pred == y_type).sum().item())

        topk = torch.topk(type_logits, k=min(3, type_logits.size(-1)), dim=-1).indices
        top3 += int((topk == y_type.unsqueeze(-1)).any(dim=-1).sum().item())

        desc_logits = out.get("desc_logits")
        if desc_logits is not None:
            y_desc = batch.get("y_desc")
            if y_desc is None:
                raise RuntimeError(
                    "Model returned 'desc_logits' but batch is missing 'y_desc'. "
                    "Re-run `python -m baseball prepare` with schema_version >= 4 (description_id present)."
                )
            dce = F.cross_entropy(desc_logits, y_desc)
            dpred = desc_logits.argmax(dim=-1)
            desc_correct += int((dpred == y_desc).sum().item())
            dtopk = torch.topk(desc_logits, k=min(3, desc_logits.size(-1)), dim=-1).indices
            desc_top3 += int((dtopk == y_desc.unsqueeze(-1)).any(dim=-1).sum().item())
            desc_ce_sum += float(dce.item()) * int(y_desc.size(0))
            desc_total += int(y_desc.size(0))

        if model_name in {"transformer_mdn", "transformer_mdn_v2", "transformer_mdn_state", "transformer_mdn_mt", "transformer_mdn_state_mt"}:
            nll_loc = mdn_nll(
                y=y_loc,
                logit_pi=out["mdn_logit_pi"],
                mu=out["mdn_mu"],
                log_sx=out["mdn_log_sx"],
                log_sz=out["mdn_log_sz"],
                rho=out["mdn_rho"],
            )
            loc_mean = mdn_mean(out["mdn_logit_pi"], out["mdn_mu"])
        else:
            nll_loc = _gaussian_nll(y_loc, out["loc_mu"], out["loc_log_s"])
            loc_mean = out["loc_mu"]

        if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"}:
            # Mask out rows without a next pitch in the same game.
            mask = batch["y_has_next"].float()
            st_loss = _state_loss(out, batch)
            # Convert averaged state loss into a sum-weighted value for epoch averaging.
            m = float(mask.sum().item())
            state_loss_sum += float(st_loss.item()) * m
            state_total += m

            def _acc(name: str, logits_key: str, y_key: str) -> None:
                pred = out[logits_key].argmax(dim=-1)
                yv = batch[y_key]
                ok = ((pred == yv) & batch["y_has_next"].bool()).sum().item()
                state_correct[name] = state_correct.get(name, 0) + int(ok)

            _acc("pa_end", "pa_end_logits", "y_pa_end")
            _acc("next_balls", "next_balls_logits", "y_next_balls")
            _acc("next_strikes", "next_strikes_logits", "y_next_strikes")
            _acc("next_outs_when_up", "next_outs_when_up_logits", "y_next_outs_when_up")
            _acc("next_on_1b", "next_on_1b_logits", "y_next_on_1b_occ")
            _acc("next_on_2b", "next_on_2b_logits", "y_next_on_2b_occ")
            _acc("next_on_3b", "next_on_3b_logits", "y_next_on_3b_occ")
            _acc("next_inning_topbot", "next_inning_topbot_logits", "y_next_inning_topbot_id")
            _acc("inning_delta", "inning_delta_logits", "y_inning_delta")
            _acc("score_diff_delta", "score_diff_delta_logits", "y_score_diff_delta_id")

        bs = y_type.size(0)
        ce_sum += float(ce.item()) * bs
        nll_loc_sum += float(nll_loc.item()) * bs
        mse_loc_sum += float(((loc_mean - y_loc) ** 2).mean(dim=-1).sum().item())

        loc_mean_ft = torch.stack(
            [
                loc_mean[:, 0] * plate_x_std + plate_x_mean,
                loc_mean[:, 1] * plate_z_std + plate_z_mean,
            ],
            dim=1,
        )
        y_loc_ft = torch.stack(
            [
                y_loc[:, 0] * plate_x_std + plate_x_mean,
                y_loc[:, 1] * plate_z_std + plate_z_mean,
            ],
            dim=1,
        )
        mse_loc_ft_sum += float(((loc_mean_ft - y_loc_ft) ** 2).mean(dim=-1).sum().item())

        total += bs

    acc = correct / max(1, total)
    acc3 = top3 / max(1, total)
    out_metrics: dict[str, float] = {
        "n": float(total),
        "ce": float(ce_sum / max(1, total)),
        "acc": float(acc),
        "acc_top3": float(acc3),
        "loc_nll": float(nll_loc_sum / max(1, total)),
        "loc_rmse": float(np.sqrt(mse_loc_sum / max(1, total))),
        "loc_rmse_ft": float(np.sqrt(mse_loc_ft_sum / max(1, total))),
    }
    if desc_total > 0:
        out_metrics["desc_n"] = float(desc_total)
        out_metrics["desc_ce"] = float(desc_ce_sum / max(1, desc_total))
        out_metrics["desc_acc"] = float(desc_correct / max(1, desc_total))
        out_metrics["desc_acc_top3"] = float(desc_top3 / max(1, desc_total))
    if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"}:
        out_metrics["state_n"] = float(state_total)
        out_metrics["state_loss"] = float(state_loss_sum / max(1.0, state_total))
        for k, v in sorted(state_correct.items()):
            out_metrics[f"state_acc_{k}"] = float(v / max(1.0, state_total))
    return out_metrics


def train(
    paths: Paths,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    *,
    run_id: str | None,
    grad_accum: int,
    amp: bool,
    amp_dtype: str,
    num_workers: int,
    prefetch_factor: int,
    streaming: bool,
    d_model: int | None,
    nhead: int | None,
    num_layers: int | None,
    mdn_components: int | None,
    dropout: float | None,
    compile_model: bool,
    grad_checkpointing: bool,
    state_corrupt_p: float,
    state_corrupt_max_delta: int,
    eval_max_batches: int | None,
    loc_weight: float,
    state_weight: float,
    desc_weight: float,
) -> None:
    _seed_all(seed)

    prepared = load_prepared(paths.data_prepared)
    vocabs = _load_vocabs(prepared.prepared_dir, prepared.meta)

    cfg = ModelConfig(
        history_len=int(prepared.meta["history_len"]),
        n_pitch_types=vocabs["pitch_type"].size,
        n_pitchers=vocabs["pitcher"].size,
        n_batters=vocabs["batter"].size,
        n_descriptions=vocabs["description"].size if "description" in vocabs else 0,
        cont_dim=len(prepared.meta["cont_features"]),
    )
    if d_model is not None:
        cfg = ModelConfig(**{**asdict(cfg), "d_model": int(d_model)})
    if nhead is not None:
        cfg = ModelConfig(**{**asdict(cfg), "nhead": int(nhead)})
    if num_layers is not None:
        cfg = ModelConfig(**{**asdict(cfg), "num_layers": int(num_layers)})
    if mdn_components is not None:
        cfg = ModelConfig(**{**asdict(cfg), "mdn_components": int(mdn_components)})
    if dropout is not None:
        cfg = ModelConfig(**{**asdict(cfg), "dropout": float(dropout)})
    if grad_checkpointing:
        cfg = ModelConfig(**{**asdict(cfg), "gradient_checkpointing": True})

    if model_name == "baseline_mlp":
        model = BaselineMLP(cfg)
    elif model_name == "transformer_mdn":
        model = TransformerMDN(cfg)
    elif model_name == "transformer_mdn_mt":
        model = TransformerMDNMT(cfg)
    elif model_name == "transformer_mdn_v2":
        model = TransformerMDNV2(cfg)
    elif model_name == "transformer_mdn_state":
        if "state_targets" not in prepared.meta:
            raise RuntimeError(
                "Prepared dataset is missing state_targets in meta.json. "
                "Re-run `python -m baseball prepare` with an updated pipeline (schema_version >= 3)."
            )
        model = TransformerMDNState(cfg)
    elif model_name == "transformer_mdn_state_mt":
        if "state_targets" not in prepared.meta:
            raise RuntimeError(
                "Prepared dataset is missing state_targets in meta.json. "
                "Re-run `python -m baseball prepare` with schema_version >= 3."
            )
        if "description" not in vocabs:
            raise RuntimeError(
                "transformer_mdn_state_mt requires a description vocab (schema_version >= 4). "
                "Re-run `python -m baseball prepare` with outcome-aware schema."
            )
        model = TransformerMDNStateMT(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not streaming and (prepared.train_path.is_dir() or prepared.valid_path.is_dir()):
        raise RuntimeError(
            "Prepared dataset is sharded (directory of parquet files). "
            "Use --streaming to train without loading everything into RAM."
        )

    if streaming:
        train_ds = PitchParquetBatchIterable(
            prepared.train_path,
            prepared.meta,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        valid_ds = PitchParquetBatchIterable(
            prepared.valid_path,
            prepared.meta,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            shuffle_rows_within_rowgroup=False,
        )
        dl_kwargs: dict[str, Any] = {
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": False,
        }
        if num_workers > 0:
            dl_kwargs["num_workers"] = num_workers
            dl_kwargs["prefetch_factor"] = prefetch_factor
        else:
            dl_kwargs["num_workers"] = 0

        train_loader = DataLoader(train_ds, batch_size=None, shuffle=False, **dl_kwargs)

        valid_nw = max(0, min(2, num_workers))
        valid_kwargs = dict(dl_kwargs)
        valid_kwargs["num_workers"] = valid_nw
        if valid_nw == 0:
            valid_kwargs.pop("prefetch_factor", None)
        valid_loader = DataLoader(valid_ds, batch_size=None, shuffle=False, **valid_kwargs)
    else:
        train_ds = PitchDataset(prepared.train_path, prepared.meta)
        valid_ds = PitchDataset(prepared.valid_path, prepared.meta)

        dl_kwargs2: dict[str, Any] = {
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": False,
        }
        if num_workers > 0:
            dl_kwargs2["num_workers"] = num_workers
            dl_kwargs2["prefetch_factor"] = prefetch_factor
        else:
            dl_kwargs2["num_workers"] = 0

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs2)

        valid_nw = max(0, min(2, num_workers))
        valid_kwargs2 = dict(dl_kwargs2)
        valid_kwargs2["num_workers"] = valid_nw
        if valid_nw == 0:
            valid_kwargs2.pop("prefetch_factor", None)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **valid_kwargs2)

    run_dir = _new_run_dir_with_id(paths.runs, run_id) if run_id is not None else _new_run_dir(paths.runs)
    logger = RunLogger(run_dir=run_dir, metrics_path=run_dir / "metrics.jsonl")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Loss weights (tunable; used only by models that emit corresponding outputs).
    loc_weight = float(loc_weight)
    state_weight = float(state_weight)
    desc_weight = float(desc_weight)
    if loc_weight < 0.0:
        raise ValueError("--loc-weight must be >= 0")
    if state_weight < 0.0:
        raise ValueError("--state-weight must be >= 0")
    if desc_weight < 0.0:
        raise ValueError("--desc-weight must be >= 0")

    state_corrupt_p = float(state_corrupt_p)
    state_corrupt_max_delta = int(state_corrupt_max_delta)
    if not (0.0 <= state_corrupt_p <= 1.0):
        raise ValueError("--state-corrupt-p must be between 0 and 1")
    if state_corrupt_max_delta < 0:
        raise ValueError("--state-corrupt-max-delta must be >= 0")

    corrupt_idx_balls = None
    corrupt_idx_strikes = None
    balls_mean = balls_std = strikes_mean = strikes_std = None
    if state_corrupt_p > 0.0:
        cont_features = prepared.meta.get("cont_features")
        if not isinstance(cont_features, list):
            raise RuntimeError("Prepared meta.json missing cont_features")
        cont_features = [str(x) for x in cont_features]
        try:
            corrupt_idx_balls = int(cont_features.index("balls"))
            corrupt_idx_strikes = int(cont_features.index("strikes"))
        except ValueError as e:
            raise RuntimeError("state corruption requires cont_features to include balls and strikes") from e

        norms = prepared.meta.get("norms")
        if not isinstance(norms, dict):
            raise RuntimeError("Prepared meta.json missing norms (required for state corruption).")
        balls_mean = float(norms["balls"]["mean"])
        balls_std = float(norms["balls"]["std"])
        strikes_mean = float(norms["strikes"]["mean"])
        strikes_std = float(norms["strikes"]["std"])
        if balls_std <= 0 or strikes_std <= 0:
            raise RuntimeError("Invalid norms std for balls/strikes (must be > 0).")

    config = {
        "run_id": run_dir.name,
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "seed": seed,
        "device": str(device),
        "amp": amp and device.type == "cuda",
        "amp_dtype": amp_dtype,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "streaming": streaming,
        "compile_model": compile_model,
        "grad_checkpointing": bool(grad_checkpointing),
        "augmentation": {
            "state_corrupt_p": float(state_corrupt_p),
            "state_corrupt_max_delta": int(state_corrupt_max_delta),
        },
        "eval_max_batches": eval_max_batches,
        "loss_weights": {"loc_weight": loc_weight, "state_weight": state_weight, "desc_weight": desc_weight},
        "model_config": asdict(cfg),
        "prepared_meta_path": str((prepared.prepared_dir / "meta.json").resolve()),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    grad_accum = int(grad_accum)
    if grad_accum <= 0:
        raise ValueError("--grad-accum must be >= 1")

    amp_enabled = bool(amp and device.type == "cuda")
    if amp_dtype not in {"fp16", "bf16"}:
        raise ValueError("--amp-dtype must be one of: fp16, bf16")
    if amp_dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested --amp-dtype bf16 but CUDA device does not support bf16.")

    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    if compile_model:
        # torch.compile is optional; on some cluster setups it can increase compile time.
        model = torch.compile(model)  # type: ignore[attr-defined]

    best_valid = float("inf")
    best_path = None

    for epoch in range(1, epochs + 1):
        model.train()

        total_batches = getattr(train_ds, "num_batches", None) if streaming else None
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs} train", total=total_batches)
        total_loss = 0.0
        total_n = 0
        optimizer.zero_grad(set_to_none=True)

        pending_steps = 0
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if state_corrupt_p > 0.0:
                x_cont = batch.get("x_cont")
                if x_cont is None:
                    raise RuntimeError("Batch is missing x_cont (required for state corruption).")
                if corrupt_idx_balls is None or corrupt_idx_strikes is None:
                    raise RuntimeError("Internal error: state corruption indices were not initialized.")
                if balls_mean is None or balls_std is None or strikes_mean is None or strikes_std is None:
                    raise RuntimeError("Internal error: state corruption norms were not initialized.")

                def _jitter(idx: int, *, mean: float, std: float, lo: float, hi: float) -> None:
                    raw = x_cont[:, idx] * std + mean
                    mask = (torch.rand(raw.shape, device=raw.device) < state_corrupt_p).to(raw.dtype)
                    delta = torch.randint(
                        -state_corrupt_max_delta,
                        state_corrupt_max_delta + 1,
                        raw.shape,
                        device=raw.device,
                    ).to(raw.dtype)
                    raw = (raw + delta * mask).clamp(lo, hi)
                    x_cont[:, idx] = (raw - mean) / std

                _jitter(corrupt_idx_balls, mean=balls_mean, std=balls_std, lo=0.0, hi=3.0)
                _jitter(corrupt_idx_strikes, mean=strikes_mean, std=strikes_std, lo=0.0, hi=2.0)

            y_type = batch["y_type"]
            y_loc = batch["y_loc"]

            if amp_enabled:
                dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = model(batch)
                    ce = F.cross_entropy(out["type_logits"], y_type)
                    if model_name in {
                        "transformer_mdn",
                        "transformer_mdn_v2",
                        "transformer_mdn_state",
                        "transformer_mdn_mt",
                        "transformer_mdn_state_mt",
                    }:
                        loc = mdn_nll(
                            y=y_loc,
                            logit_pi=out["mdn_logit_pi"],
                            mu=out["mdn_mu"],
                            log_sx=out["mdn_log_sx"],
                            log_sz=out["mdn_log_sz"],
                            rho=out["mdn_rho"],
                        )
                    else:
                        loc = _gaussian_nll(y_loc, out["loc_mu"], out["loc_log_s"])
                    desc = torch.tensor(0.0, device=device)
                    if "desc_logits" in out:
                        if "y_desc" not in batch:
                            raise RuntimeError(
                                "Model returned 'desc_logits' but batch is missing 'y_desc'. "
                                "Re-run `python -m baseball prepare` with schema_version >= 4."
                            )
                        desc = F.cross_entropy(out["desc_logits"], batch["y_desc"])
                    st = torch.tensor(0.0, device=device)
                    if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"}:
                        st = _state_loss(out, batch)
                    loss = (ce + desc_weight * desc + loc_weight * loc + state_weight * st) / grad_accum
                scaler.scale(loss).backward()
            else:
                out = model(batch)
                ce = F.cross_entropy(out["type_logits"], y_type)
                if model_name in {"transformer_mdn", "transformer_mdn_v2", "transformer_mdn_state", "transformer_mdn_mt", "transformer_mdn_state_mt"}:
                    loc = mdn_nll(
                        y=y_loc,
                        logit_pi=out["mdn_logit_pi"],
                        mu=out["mdn_mu"],
                        log_sx=out["mdn_log_sx"],
                        log_sz=out["mdn_log_sz"],
                        rho=out["mdn_rho"],
                    )
                else:
                    loc = _gaussian_nll(y_loc, out["loc_mu"], out["loc_log_s"])
                desc = torch.tensor(0.0, device=device)
                if "desc_logits" in out:
                    if "y_desc" not in batch:
                        raise RuntimeError(
                            "Model returned 'desc_logits' but batch is missing 'y_desc'. "
                            "Re-run `python -m baseball prepare` with schema_version >= 4."
                        )
                    desc = F.cross_entropy(out["desc_logits"], batch["y_desc"])
                st = torch.tensor(0.0, device=device)
                if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"}:
                    st = _state_loss(out, batch)
                loss = (ce + desc_weight * desc + loc_weight * loc + state_weight * st) / grad_accum
                loss.backward()

            pending_steps += 1
            if pending_steps >= grad_accum:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_steps = 0
                _assert_finite_params(model)

            bs = y_type.size(0)
            # Undo grad accumulation scaling for logging.
            total_loss += float(loss.item()) * bs * grad_accum
            total_n += bs
            postfix = {"loss": total_loss / max(1, total_n), "ce": float(ce.item()), "loc": float(loc.item())}
            if float(desc.item()) > 0:
                postfix["desc"] = float(desc.item())
            if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"}:
                postfix["state"] = float(st.item())
            pbar.set_postfix(postfix)

        # Flush final partial accumulation.
        if pending_steps > 0:
            if amp_enabled:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _assert_finite_params(model)

        norms = prepared.meta["norms"]
        px_mean = float(norms["plate_x"]["mean"])
        px_std = float(norms["plate_x"]["std"])
        pz_mean = float(norms["plate_z"]["mean"])
        pz_std = float(norms["plate_z"]["std"])

        train_metrics = _evaluate_epoch(
            model,
            train_loader,
            device=device,
            model_name=model_name,
            plate_x_mean=px_mean,
            plate_x_std=px_std,
            plate_z_mean=pz_mean,
            plate_z_std=pz_std,
            max_batches=eval_max_batches,
        )
        valid_metrics = _evaluate_epoch(
            model,
            valid_loader,
            device=device,
            model_name=model_name,
            plate_x_mean=px_mean,
            plate_x_std=px_std,
            plate_z_mean=pz_mean,
            plate_z_std=pz_std,
            max_batches=eval_max_batches,
        )

        logger.log_event("epoch_end", {"epoch": epoch, "split": "train", **train_metrics})
        logger.log_event("epoch_end", {"epoch": epoch, "split": "valid", **valid_metrics})

        _assert_finite_metrics(train_metrics, split="train")
        _assert_finite_metrics(valid_metrics, split="valid")

        valid_loss = valid_metrics["ce"] + loc_weight * valid_metrics["loc_nll"]
        if "desc_ce" in valid_metrics:
            valid_loss = float(valid_loss + desc_weight * valid_metrics["desc_ce"])
        if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"}:
            valid_loss = float(valid_loss + state_weight * valid_metrics.get("state_loss", float("inf")))
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_path = run_dir / "checkpoints" / "best.pt"
            torch.save(
                {
                    "model_name": model_name,
                    "model_config": asdict(cfg),
                    "state_dict": model.state_dict(),
                    "prepared_meta": prepared.meta,
                },
                best_path,
            )

    if best_path is None:
        raise RuntimeError("Training completed without producing a checkpoint.")
    logger.log_event("run_end", {"best_valid": best_valid, "best_checkpoint": str(best_path)})
