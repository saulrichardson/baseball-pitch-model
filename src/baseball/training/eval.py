from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseball.config import Paths
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
from baseball.training.runs import resolve_run_dir
from baseball.training.streaming import PitchParquetBatchIterable


def _load_best_checkpoint(run_dir: Path) -> dict[str, Any]:
    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt}")
    return torch.load(ckpt, map_location="cpu")


def _masked_cross_entropy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.float32 and mask.dtype != torch.float64:
        mask = mask.float()
    per = F.cross_entropy(logits, y, reduction="none")
    denom = mask.sum().clamp(min=1.0)
    return (per * mask).sum() / denom


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    px_mean: float,
    px_std: float,
    pz_mean: float,
    pz_std: float,
) -> dict[str, float]:
    model.eval()

    n = 0
    correct = 0
    top3 = 0
    ce_sum = 0.0
    desc_total = 0
    desc_correct = 0
    desc_top3 = 0
    desc_ce_sum = 0.0
    loc_nll_sum = 0.0
    loc_nll_count = 0
    mse_ft_sum = 0.0

    state_total = 0.0
    state_loss_sum = 0.0
    state_correct: dict[str, int] = {}

    for batch in tqdm(loader, desc="eval"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        y_type = batch["y_type"]
        y_loc = batch["y_loc"]

        out = model(batch)
        logits = out["type_logits"]
        ce = F.cross_entropy(logits, y_type)

        pred = logits.argmax(dim=-1)
        correct += int((pred == y_type).sum().item())
        topk = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1).indices
        top3 += int((topk == y_type.unsqueeze(-1)).any(dim=-1).sum().item())

        desc_logits = out.get("desc_logits")
        if desc_logits is not None:
            y_desc = batch.get("y_desc")
            if y_desc is None:
                raise RuntimeError(
                    "Model returned 'desc_logits' but batch is missing 'y_desc'. "
                    "Re-run `python -m baseball prepare` with schema_version >= 4."
                )
            dce = F.cross_entropy(desc_logits, y_desc)
            dpred = desc_logits.argmax(dim=-1)
            desc_correct += int((dpred == y_desc).sum().item())
            dtopk = torch.topk(desc_logits, k=min(3, desc_logits.size(-1)), dim=-1).indices
            desc_top3 += int((dtopk == y_desc.unsqueeze(-1)).any(dim=-1).sum().item())
            desc_ce_sum += float(dce.item()) * int(y_desc.size(0))
            desc_total += int(y_desc.size(0))

        if model_name in {
            "transformer_mdn",
            "transformer_mdn_v2",
            "transformer_mdn_state",
            "transformer_mdn_mt",
            "transformer_mdn_state_mt",
        }:
            loc_nll = mdn_nll(
                y=y_loc,
                logit_pi=out["mdn_logit_pi"],
                mu=out["mdn_mu"],
                log_sx=out["mdn_log_sx"],
                log_sz=out["mdn_log_sz"],
                rho=out["mdn_rho"],
            )
            loc_mean = mdn_mean(out["mdn_logit_pi"], out["mdn_mu"])
        else:
            loc_mean = out["loc_mu"]
            # Baseline location NLL is not directly comparable to MDN NLL; report as NaN.
            loc_nll = torch.tensor(float("nan"), device=device)

        if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"}:
            mask = batch["y_has_next"].float()
            st = torch.stack(
                [
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
            ).mean()
            m = float(mask.sum().item())
            state_loss_sum += float(st.item()) * m
            state_total += m

            def _acc(name: str, logits_key: str, y_key: str) -> None:
                pred = out[logits_key].argmax(dim=-1)
                ok = ((pred == batch[y_key]) & batch["y_has_next"].bool()).sum().item()
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

        # Unnormalize to feet units.
        loc_mean_ft = torch.stack(
            [loc_mean[:, 0] * px_std + px_mean, loc_mean[:, 1] * pz_std + pz_mean], dim=1
        )
        y_loc_ft = torch.stack([y_loc[:, 0] * px_std + px_mean, y_loc[:, 1] * pz_std + pz_mean], dim=1)
        mse_ft_sum += float(((loc_mean_ft - y_loc_ft) ** 2).mean(dim=-1).sum().item())

        bs = y_type.size(0)
        n += bs
        ce_sum += float(ce.item()) * bs
        if torch.isfinite(loc_nll):
            loc_nll_sum += float(loc_nll.item()) * bs
            loc_nll_count += bs

    return {
        "n": float(n),
        "acc": float(correct / max(1, n)),
        "acc_top3": float(top3 / max(1, n)),
        "ce": float(ce_sum / max(1, n)),
        **(
            {
                "desc_n": float(desc_total),
                "desc_acc": float(desc_correct / max(1, desc_total)),
                "desc_acc_top3": float(desc_top3 / max(1, desc_total)),
                "desc_ce": float(desc_ce_sum / max(1, desc_total)),
            }
            if desc_total > 0
            else {}
        ),
        "loc_nll": float(loc_nll_sum / loc_nll_count) if loc_nll_count > 0 else float("nan"),
        "loc_rmse_ft": float((mse_ft_sum / max(1, n)) ** 0.5),
        "state_n": float(state_total) if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"} else float("nan"),
        "state_loss": float(state_loss_sum / max(1.0, state_total))
        if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"}
        else float("nan"),
        **(
            {f"state_acc_{k}": float(v / max(1.0, state_total)) for k, v in sorted(state_correct.items())}
            if model_name in {"transformer_mdn_state", "transformer_mdn_state_mt"}
            else {}
        ),
    }


def evaluate_latest(paths: Paths, split: str = "valid", *, run_id: str | None = None) -> None:
    run_dir = resolve_run_dir(paths.runs, run_id=run_id)
    ckpt = _load_best_checkpoint(run_dir)

    model_name = str(ckpt["model_name"])
    cfg = ModelConfig(**ckpt["model_config"])
    if model_name == "baseline_mlp":
        model = BaselineMLP(cfg)
    elif model_name == "transformer_mdn":
        model = TransformerMDN(cfg)
    elif model_name == "transformer_mdn_mt":
        model = TransformerMDNMT(cfg)
    elif model_name == "transformer_mdn_v2":
        model = TransformerMDNV2(cfg)
    elif model_name == "transformer_mdn_state":
        model = TransformerMDNState(cfg)
    elif model_name == "transformer_mdn_state_mt":
        model = TransformerMDNStateMT(cfg)
    else:
        raise ValueError(f"Unknown model in checkpoint: {model_name}")
    model.load_state_dict(ckpt["state_dict"])

    prepared = load_prepared(paths.data_prepared)
    meta = prepared.meta
    norms = meta["norms"]
    px_mean = float(norms["plate_x"]["mean"])
    px_std = float(norms["plate_x"]["std"])
    pz_mean = float(norms["plate_z"]["mean"])
    pz_std = float(norms["plate_z"]["std"])

    ds_path = prepared.valid_path if split == "valid" else prepared.train_path
    ds = PitchParquetBatchIterable(
        ds_path,
        meta,
        batch_size=4096,
        shuffle=False,
        seed=0,
        shuffle_rows_within_rowgroup=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dl_kwargs: dict[str, Any] = {
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "num_workers": 2,
        "prefetch_factor": 2,
    }
    loader = DataLoader(ds, batch_size=None, shuffle=False, **dl_kwargs)
    metrics = _evaluate(
        model,
        loader,
        device=device,
        model_name=model_name,
        px_mean=px_mean,
        px_std=px_std,
        pz_mean=pz_mean,
        pz_std=pz_std,
    )

    out = {"run_dir": str(run_dir), "model": model_name, "split": split, **metrics}
    print(json.dumps(out, indent=2, sort_keys=True))
