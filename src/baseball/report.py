from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseball.config import Paths
from baseball.data.vocab import Vocab
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


class ReportError(RuntimeError):
    pass


BaselineName = Literal["global", "pitcher", "pitcher_count", "pitcher_count_prev", "pitcher_count_prev_outcome"]


def _last_nonzero(x: np.ndarray) -> np.ndarray:
    """
    x: [B,L] int64 with 0 as padding.
    Return: [B] last non-zero element or 0.
    """

    if x.ndim != 2:
        raise ReportError("Expected hist array to be rank-2 [B,L].")
    if x.shape[1] == 0:
        return np.zeros((x.shape[0],), dtype=np.int64)

    mask = x != 0
    has = mask.any(axis=1)
    rev = mask[:, ::-1]
    idx_from_end = np.argmax(rev, axis=1).astype(np.int64)
    last_idx = (x.shape[1] - 1 - idx_from_end).astype(np.int64)
    out = x[np.arange(x.shape[0]), last_idx].astype(np.int64, copy=False)
    out[~has] = 0
    return out


def _pack_pitcher_count_key(pitcher_id: np.ndarray, balls: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    pitcher_id = pitcher_id.astype(np.int64, copy=False)
    balls = balls.astype(np.int64, copy=False)
    strikes = strikes.astype(np.int64, copy=False)
    if int(balls.min()) < 0 or int(balls.max()) > 3:
        raise ReportError(f"balls out of expected range [0,3]: min={balls.min()} max={balls.max()}")
    if int(strikes.min()) < 0 or int(strikes.max()) > 2:
        raise ReportError(f"strikes out of expected range [0,2]: min={strikes.min()} max={strikes.max()}")
    return pitcher_id * 100 + balls * 10 + strikes


def _pack_pitcher_count_prev_key(
    pitcher_id: np.ndarray, balls: np.ndarray, strikes: np.ndarray, prev_type: np.ndarray
) -> np.ndarray:
    pitcher_id = pitcher_id.astype(np.int64, copy=False)
    balls = balls.astype(np.int64, copy=False)
    strikes = strikes.astype(np.int64, copy=False)
    prev_type = prev_type.astype(np.int64, copy=False)
    if int(prev_type.min()) < 0 or int(prev_type.max()) >= 10_000:
        raise ReportError(f"prev_type out of expected range [0,9999]: min={prev_type.min()} max={prev_type.max()}")
    return pitcher_id * 100_000 + balls * 10_000 + strikes * 1_000 + prev_type


def _pack_pitcher_count_prev_outcome_key(
    pitcher_id: np.ndarray,
    balls: np.ndarray,
    strikes: np.ndarray,
    prev_type: np.ndarray,
    prev_desc: np.ndarray,
) -> np.ndarray:
    pitcher_id = pitcher_id.astype(np.int64, copy=False)
    balls = balls.astype(np.int64, copy=False)
    strikes = strikes.astype(np.int64, copy=False)
    prev_type = prev_type.astype(np.int64, copy=False)
    prev_desc = prev_desc.astype(np.int64, copy=False)
    if int(prev_type.min()) < 0 or int(prev_type.max()) >= 100_000:
        raise ReportError(
            f"prev_type out of expected range [0,99999]: min={prev_type.min()} max={prev_type.max()}"
        )
    if int(prev_desc.min()) < 0 or int(prev_desc.max()) >= 100_000:
        raise ReportError(
            f"prev_desc out of expected range [0,99999]: min={prev_desc.min()} max={prev_desc.max()}"
        )
    return (
        pitcher_id * 1_000_000_000
        + balls * 100_000_000
        + strikes * 10_000_000
        + prev_type * 100_000
        + prev_desc
    )


def _bucket_inning(inning: np.ndarray) -> np.ndarray:
    inning = inning.astype(np.int64, copy=False)
    out = np.zeros_like(inning, dtype=np.int64)
    out[(inning >= 4) & (inning <= 6)] = 1
    out[(inning >= 7) & (inning <= 8)] = 2
    out[inning >= 9] = 3
    return out


def _bucket_score_diff(score_diff: np.ndarray) -> np.ndarray:
    score_diff = score_diff.astype(np.int64, copy=False)
    out = np.empty_like(score_diff, dtype=np.int64)
    out[score_diff <= -3] = 0
    out[score_diff == -2] = 1
    out[score_diff == -1] = 2
    out[score_diff == 0] = 3
    out[score_diff == 1] = 4
    out[score_diff == 2] = 5
    out[score_diff >= 3] = 6
    return out


def _bucket_pitch_number(pitch_number: np.ndarray) -> np.ndarray:
    """
    Bucket pitch_number into:
      0: pitch 1
      1: pitch 2
      2: pitch 3
      3: pitch 4+
    """

    pitch_number = pitch_number.astype(np.int64, copy=False)
    out = np.zeros_like(pitch_number, dtype=np.int64)
    out[pitch_number == 2] = 1
    out[pitch_number == 3] = 2
    out[pitch_number >= 4] = 3
    return out


def _runners_state(on_1b: np.ndarray, on_2b: np.ndarray, on_3b: np.ndarray) -> np.ndarray:
    on_1b = on_1b.astype(np.int64, copy=False)
    on_2b = on_2b.astype(np.int64, copy=False)
    on_3b = on_3b.astype(np.int64, copy=False)
    return (on_1b > 0).astype(np.int64) + 2 * (on_2b > 0).astype(np.int64) + 4 * (on_3b > 0).astype(np.int64)


@dataclass
class _CountsAgg:
    counts: np.ndarray  # [T]
    loc_sum: np.ndarray  # [2] normalized
    n: int


def _update_agg_map(
    agg: dict[int, _CountsAgg],
    *,
    keys: np.ndarray,
    y_type: np.ndarray,
    y_loc: np.ndarray,
    n_types: int,
) -> None:
    if keys.ndim != 1:
        raise ReportError("keys must be rank-1")
    if y_type.ndim != 1:
        raise ReportError("y_type must be rank-1")
    if y_loc.ndim != 2 or y_loc.shape[1] != 2:
        raise ReportError("y_loc must be [B,2]")
    if keys.shape[0] != y_type.shape[0] or keys.shape[0] != y_loc.shape[0]:
        raise ReportError("keys/y_type/y_loc must have same batch size")

    uniq, inv = np.unique(keys, return_inverse=True)
    for i, key in enumerate(uniq.tolist()):
        mask = inv == i
        ys = y_type[mask]
        loc = y_loc[mask]
        if ys.size == 0:
            continue
        add = np.bincount(ys, minlength=n_types).astype(np.int64, copy=False)
        loc_sum = loc.sum(axis=0).astype(np.float64, copy=False)
        n = int(ys.size)
        cur = agg.get(int(key))
        if cur is None:
            agg[int(key)] = _CountsAgg(counts=add.astype(np.int64, copy=True), loc_sum=loc_sum.copy(), n=n)
        else:
            cur.counts += add
            cur.loc_sum += loc_sum
            cur.n += n


@dataclass(frozen=True)
class BaselineFit:
    n_types: int
    norms: dict[str, dict[str, float]]
    global_: _CountsAgg
    by_pitcher: dict[int, _CountsAgg]
    by_pitcher_count: dict[int, _CountsAgg]
    by_pitcher_count_prev: dict[int, _CountsAgg]
    by_pitcher_count_prev_outcome: dict[int, _CountsAgg] | None


def fit_baselines(
    paths: Paths,
    *,
    split: str = "train",
    batch_size: int = 8192,
) -> BaselineFit:
    if split not in {"train", "valid"}:
        raise ValueError("--split must be one of: train, valid")
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    prepared = load_prepared(paths.data_prepared)
    meta = prepared.meta
    norms = meta.get("norms")
    if not isinstance(norms, dict):
        raise ReportError("Prepared meta.json missing norms.")

    pitch_type_vocab = Vocab.load(prepared.vocabs_dir / "pitch_type.json")
    n_types = int(pitch_type_vocab.size)

    ds_path = prepared.valid_path if split == "valid" else prepared.train_path
    ds = PitchParquetBatchIterable(
        ds_path,
        meta,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        shuffle_rows_within_rowgroup=False,
        return_raw_cont=True,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )

    cont_features = meta.get("cont_features")
    if not isinstance(cont_features, list):
        raise ReportError("Prepared meta.json missing cont_features.")
    cont_features = [str(x) for x in cont_features]
    try:
        idx_balls = cont_features.index("balls")
        idx_strikes = cont_features.index("strikes")
    except ValueError as e:
        raise ReportError("cont_features must include balls and strikes.") from e

    global_counts = np.zeros((n_types,), dtype=np.int64)
    global_loc_sum = np.zeros((2,), dtype=np.float64)
    global_n = 0

    by_pitcher: dict[int, _CountsAgg] = {}
    by_pitcher_count: dict[int, _CountsAgg] = {}
    by_pitcher_count_prev: dict[int, _CountsAgg] = {}

    has_desc_hist = bool(meta.get("seq_features", {}).get("hist_description_id_cols"))
    by_pitcher_count_prev_outcome: dict[int, _CountsAgg] | None = {} if has_desc_hist else None

    for batch in tqdm(loader, desc=f"fit_baselines[{split}]"):
        x_cat = batch["x_cat"].numpy().astype(np.int64, copy=False)
        x_cont_raw_t = batch.get("x_cont_raw")
        if x_cont_raw_t is None:
            raise ReportError("Internal error: expected x_cont_raw from streaming dataset.")
        x_cont_raw = x_cont_raw_t.numpy().astype(np.int64, copy=False)

        pitcher_id = x_cat[:, 0]
        balls = x_cont_raw[:, idx_balls]
        strikes = x_cont_raw[:, idx_strikes]

        y_type = batch["y_type"].numpy().astype(np.int64, copy=False)
        y_loc = batch["y_loc"].numpy().astype(np.float32, copy=False)

        global_counts += np.bincount(y_type, minlength=n_types).astype(np.int64, copy=False)
        global_loc_sum += y_loc.sum(axis=0).astype(np.float64, copy=False)
        global_n += int(y_type.size)

        # pitcher baseline
        _update_agg_map(by_pitcher, keys=pitcher_id, y_type=y_type, y_loc=y_loc, n_types=n_types)

        # pitcher+count baseline
        k_pc = _pack_pitcher_count_key(pitcher_id, balls, strikes)
        _update_agg_map(by_pitcher_count, keys=k_pc, y_type=y_type, y_loc=y_loc, n_types=n_types)

        # pitcher+count+prev pitch type
        hist_type = batch["hist_type"].numpy().astype(np.int64, copy=False)
        prev_type = _last_nonzero(hist_type)
        k_pcp = _pack_pitcher_count_prev_key(pitcher_id, balls, strikes, prev_type)
        _update_agg_map(by_pitcher_count_prev, keys=k_pcp, y_type=y_type, y_loc=y_loc, n_types=n_types)

        if by_pitcher_count_prev_outcome is not None:
            hist_desc_t = batch.get("hist_desc")
            if hist_desc_t is None:
                raise ReportError("Prepared data includes description history, but batch is missing hist_desc.")
            hist_desc = hist_desc_t.numpy().astype(np.int64, copy=False)
            prev_desc = _last_nonzero(hist_desc)
            k_pcpo = _pack_pitcher_count_prev_outcome_key(pitcher_id, balls, strikes, prev_type, prev_desc)
            _update_agg_map(by_pitcher_count_prev_outcome, keys=k_pcpo, y_type=y_type, y_loc=y_loc, n_types=n_types)

    if global_n <= 0:
        raise ReportError("No rows found while fitting baselines.")

    return BaselineFit(
        n_types=n_types,
        norms={str(k): dict(v) for k, v in norms.items()},
        global_=_CountsAgg(counts=global_counts, loc_sum=global_loc_sum, n=global_n),
        by_pitcher=by_pitcher,
        by_pitcher_count=by_pitcher_count,
        by_pitcher_count_prev=by_pitcher_count_prev,
        by_pitcher_count_prev_outcome=by_pitcher_count_prev_outcome,
    )


def _counts_to_log_probs(counts: np.ndarray, *, alpha: float) -> np.ndarray:
    if counts.ndim != 1:
        raise ReportError("counts must be rank-1")
    if alpha <= 0:
        raise ValueError("--alpha must be > 0")
    c = counts.astype(np.float64, copy=False)
    denom = float(c.sum() + alpha * c.size)
    if denom <= 0:
        raise ReportError("Invalid denom in counts_to_log_probs.")
    p = (c + alpha) / denom
    return np.log(p)


def _counts_to_loc_mean(loc_sum: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        raise ReportError("n must be > 0 for location mean.")
    return (loc_sum / float(n)).astype(np.float32, copy=False)


def evaluate_baselines(
    baselines: BaselineFit,
    paths: Paths,
    *,
    split: str,
    alpha: float = 1.0,
    batch_size: int = 8192,
) -> dict[str, Any]:
    if split not in {"train", "valid"}:
        raise ValueError("--split must be one of: train, valid")
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    prepared = load_prepared(paths.data_prepared)
    meta = prepared.meta
    cont_features = meta.get("cont_features")
    if not isinstance(cont_features, list):
        raise ReportError("Prepared meta.json missing cont_features.")
    cont_features = [str(x) for x in cont_features]
    try:
        idx_balls = cont_features.index("balls")
        idx_strikes = cont_features.index("strikes")
    except ValueError as e:
        raise ReportError("cont_features must include balls and strikes.") from e

    px_mean = float(baselines.norms["plate_x"]["mean"])
    px_std = float(baselines.norms["plate_x"]["std"])
    pz_mean = float(baselines.norms["plate_z"]["mean"])
    pz_std = float(baselines.norms["plate_z"]["std"])

    ds_path = prepared.valid_path if split == "valid" else prepared.train_path
    ds = PitchParquetBatchIterable(
        ds_path,
        meta,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        shuffle_rows_within_rowgroup=False,
        return_raw_cont=True,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )

    # Pre-compute global params.
    g_counts = baselines.global_.counts.astype(np.int64, copy=False)
    g_total = float(g_counts.sum() + float(alpha) * float(baselines.n_types))
    if g_total <= 0:
        raise ReportError("Invalid global total while evaluating baselines.")
    g_pred = int(np.argmax(g_counts))
    g_top3 = np.argsort(-g_counts)[:3].astype(np.int64, copy=False)
    g_loc = _counts_to_loc_mean(baselines.global_.loc_sum, baselines.global_.n)

    # Metrics accumulators (single pass over eval split).
    names: list[BaselineName] = ["global", "pitcher", "pitcher_count", "pitcher_count_prev"]
    if baselines.by_pitcher_count_prev_outcome is not None:
        names.append("pitcher_count_prev_outcome")

    acc_top1: dict[str, int] = {n: 0 for n in names}
    acc_top3: dict[str, int] = {n: 0 for n in names}
    ce_sum: dict[str, float] = {n: 0.0 for n in names}
    loc_mse_sum: dict[str, float] = {n: 0.0 for n in names}
    fallback_n: dict[str, int] = {n: 0 for n in names if n != "global"}
    total_n = 0

    def _eval_grouped(
        *,
        name: BaselineName,
        keys: np.ndarray,
        counts_map: dict[int, _CountsAgg] | None,
        y_type: np.ndarray,
        y_loc_ft: np.ndarray,
    ) -> None:
        if keys.ndim != 1:
            raise ReportError("keys must be rank-1")
        if y_type.ndim != 1:
            raise ReportError("y_type must be rank-1")
        if y_loc_ft.ndim != 2 or y_loc_ft.shape[1] != 2:
            raise ReportError("y_loc_ft must be [B,2]")

        uniq, inv = np.unique(keys, return_inverse=True)
        for i, key in enumerate(uniq.tolist()):
            mask = inv == i
            y = y_type[mask]
            loc = y_loc_ft[mask]
            if y.size == 0:
                continue

            if name == "global" or counts_map is None:
                c = g_counts
                total = g_total
                pred = g_pred
                top3_idx = g_top3
                mu_norm = g_loc
            else:
                agg = counts_map.get(int(key))
                if agg is None:
                    fallback_n[name] += int(y.size)
                    c = g_counts
                    total = g_total
                    pred = g_pred
                    top3_idx = g_top3
                    mu_norm = g_loc
                else:
                    c = agg.counts.astype(np.int64, copy=False)
                    total = float(c.sum() + float(alpha) * float(baselines.n_types))
                    if total <= 0:
                        raise ReportError(f"Invalid total for baseline={name} key={key}")
                    pred = int(np.argmax(c))
                    top3_idx = np.argsort(-c)[:3].astype(np.int64, copy=False)
                    mu_norm = _counts_to_loc_mean(agg.loc_sum, agg.n)

            # Top-1 / Top-3
            acc_top1[name] += int((y == pred).sum())
            acc_top3[name] += int(np.isin(y, top3_idx).sum())

            # Cross-entropy: -log p(y)
            p_num = c[y].astype(np.float64, copy=False) + float(alpha)
            ce_sum[name] += float((-np.log(p_num / total)).sum())

            # Location RMSE in feet (use per-group mean in normalized coords).
            mu_ft = np.array([mu_norm[0] * px_std + px_mean, mu_norm[1] * pz_std + pz_mean], dtype=np.float64)
            loc_mse_sum[name] += float((((loc - mu_ft) ** 2).mean(axis=1)).sum())

    for batch in tqdm(loader, desc=f"eval_baselines[{split}]"):
        x_cat = batch["x_cat"].numpy().astype(np.int64, copy=False)
        x_cont_raw = batch["x_cont_raw"].numpy().astype(np.int64, copy=False)
        y_type = batch["y_type"].numpy().astype(np.int64, copy=False)
        y_loc = batch["y_loc"].numpy().astype(np.float32, copy=False)

        pitcher_id = x_cat[:, 0]
        balls = x_cont_raw[:, idx_balls]
        strikes = x_cont_raw[:, idx_strikes]

        y_loc_ft = np.stack(
            [y_loc[:, 0] * px_std + px_mean, y_loc[:, 1] * pz_std + pz_mean],
            axis=1,
        ).astype(np.float64, copy=False)

        total_n += int(y_type.size)

        # Global (keyless).
        _eval_grouped(name="global", keys=np.zeros_like(pitcher_id), counts_map=None, y_type=y_type, y_loc_ft=y_loc_ft)

        # Pitcher
        _eval_grouped(name="pitcher", keys=pitcher_id, counts_map=baselines.by_pitcher, y_type=y_type, y_loc_ft=y_loc_ft)

        # Pitcher + count
        k_pc = _pack_pitcher_count_key(pitcher_id, balls, strikes)
        _eval_grouped(
            name="pitcher_count",
            keys=k_pc,
            counts_map=baselines.by_pitcher_count,
            y_type=y_type,
            y_loc_ft=y_loc_ft,
        )

        # Pitcher + count + prev pitch
        hist_type = batch["hist_type"].numpy().astype(np.int64, copy=False)
        prev_type = _last_nonzero(hist_type)
        k_pcp = _pack_pitcher_count_prev_key(pitcher_id, balls, strikes, prev_type)
        _eval_grouped(
            name="pitcher_count_prev",
            keys=k_pcp,
            counts_map=baselines.by_pitcher_count_prev,
            y_type=y_type,
            y_loc_ft=y_loc_ft,
        )

        if baselines.by_pitcher_count_prev_outcome is not None:
            hist_desc_t = batch.get("hist_desc")
            if hist_desc_t is None:
                raise ReportError("Baseline pitcher_count_prev_outcome enabled but batch has no hist_desc.")
            hist_desc = hist_desc_t.numpy().astype(np.int64, copy=False)
            prev_desc = _last_nonzero(hist_desc)
            k_pcpo = _pack_pitcher_count_prev_outcome_key(pitcher_id, balls, strikes, prev_type, prev_desc)
            _eval_grouped(
                name="pitcher_count_prev_outcome",
                keys=k_pcpo,
                counts_map=baselines.by_pitcher_count_prev_outcome,
                y_type=y_type,
                y_loc_ft=y_loc_ft,
            )

    out: dict[str, Any] = {
        "split": str(split),
        "alpha": float(alpha),
        "n_types": int(baselines.n_types),
        "baselines": [],
    }

    for name in names:
        entry = {
            "name": str(name),
            "split": str(split),
            "n": int(total_n),
            "acc": float(acc_top1[name] / max(1, total_n)),
            "acc_top3": float(acc_top3[name] / max(1, total_n)),
            "ce": float(ce_sum[name] / max(1, total_n)),
            "loc_rmse_ft": float((loc_mse_sum[name] / max(1, total_n)) ** 0.5),
            "alpha": float(alpha),
        }
        if name != "global":
            entry["fallback_n"] = int(fallback_n[name])
            entry["fallback_frac"] = float(fallback_n[name] / max(1, total_n))
        out["baselines"].append(entry)

    return out


@dataclass(frozen=True)
class CalibrationBin:
    count: int
    acc: float
    conf: float


def _calibration_bins(conf: torch.Tensor, correct: torch.Tensor, *, n_bins: int) -> tuple[list[CalibrationBin], float]:
    if conf.ndim != 1 or correct.ndim != 1:
        raise ReportError("conf/correct must be rank-1")
    if conf.numel() != correct.numel():
        raise ReportError("conf/correct must have same length")
    if n_bins <= 1:
        raise ValueError("--calibration-bins must be > 1")

    # Bin by confidence in [0,1]. Use right-open bins except the last.
    bins: list[CalibrationBin] = []
    ece = 0.0
    n = float(conf.numel())
    for b in range(n_bins):
        lo = float(b) / float(n_bins)
        hi = float(b + 1) / float(n_bins)
        if b == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        cnt = int(m.sum().item())
        if cnt == 0:
            bins.append(CalibrationBin(count=0, acc=float("nan"), conf=float("nan")))
            continue
        acc = float(correct[m].float().mean().item())
        c = float(conf[m].mean().item())
        bins.append(CalibrationBin(count=cnt, acc=acc, conf=c))
        ece += abs(acc - c) * (float(cnt) / max(1.0, n))
    return bins, float(ece)


@torch.no_grad()
def evaluate_model_with_calibration(
    paths: Paths,
    *,
    run_id: str,
    split: str,
    batch_size: int = 4096,
    calibration_bins: int = 15,
    device: str | None = None,
) -> dict[str, Any]:
    if split not in {"train", "valid"}:
        raise ValueError("--split must be one of: train, valid")
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    prepared = load_prepared(paths.data_prepared)
    meta = prepared.meta
    norms = meta.get("norms")
    if not isinstance(norms, dict):
        raise ReportError("Prepared meta.json missing norms.")
    px_mean = float(norms["plate_x"]["mean"])
    px_std = float(norms["plate_x"]["std"])
    pz_mean = float(norms["plate_z"]["mean"])
    pz_std = float(norms["plate_z"]["std"])

    run_dir = resolve_run_dir(paths.runs, run_id=run_id)
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_name = str(ckpt.get("model_name"))
    cfg = ModelConfig(**ckpt["model_config"])
    if model_name == "baseline_mlp":
        model: torch.nn.Module = BaselineMLP(cfg)
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
        raise ReportError(f"Unknown model in checkpoint: {model_name}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if device is None or device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    model.to(dev)

    ds_path = prepared.valid_path if split == "valid" else prepared.train_path
    ds = PitchParquetBatchIterable(
        ds_path,
        meta,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        shuffle_rows_within_rowgroup=False,
        return_raw_cont=True,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )

    n = 0
    correct = 0
    top3 = 0
    ce_sum = 0.0
    brier_sum = 0.0
    loc_nll_sum = 0.0
    loc_nll_count = 0
    mse_ft_sum = 0.0

    desc_total = 0
    desc_correct = 0
    desc_top3 = 0
    desc_ce_sum = 0.0

    # Calibration accumulators (online; avoids storing per-example arrays)
    bin_count = np.zeros((calibration_bins,), dtype=np.int64)
    bin_conf_sum = np.zeros((calibration_bins,), dtype=np.float64)
    bin_acc_sum = np.zeros((calibration_bins,), dtype=np.float64)

    # Slice aggregations. Keep these high-signal and low-dimensional so they stay readable.
    cont_features = meta.get("cont_features")
    if not isinstance(cont_features, list):
        raise ReportError("Prepared meta.json missing cont_features.")
    cont_features = [str(x) for x in cont_features]
    try:
        idx_inning = cont_features.index("inning")
        idx_score_diff = cont_features.index("score_diff")
        idx_on_1b = cont_features.index("on_1b_occ")
        idx_on_2b = cont_features.index("on_2b_occ")
        idx_on_3b = cont_features.index("on_3b_occ")
        idx_pitch_number = cont_features.index("pitch_number")
        idx_balls = cont_features.index("balls")
        idx_strikes = cont_features.index("strikes")
    except ValueError as e:
        raise ReportError("cont_features missing required columns for slice metrics.") from e

    @dataclass
    class _SliceAgg:
        n: int = 0
        correct: int = 0
        top3: int = 0
        ce_sum: float = 0.0
        loc_mse_sum: float = 0.0

    slice_aggs: dict[str, dict[int, _SliceAgg]] = {
        "count": {},
        "pitch_number_bucket": {},
        "inning_bucket": {},
        "runners_state": {},
        "score_bucket": {},
        "stand_id": {},
        "p_throws_id": {},
    }

    def _update_slice(
        name: str,
        keys: np.ndarray,
        *,
        correct_b: np.ndarray,
        top3_b: np.ndarray,
        ce: np.ndarray,
        loc_mse: np.ndarray,
    ) -> None:
        if keys.ndim != 1:
            raise ReportError("slice keys must be rank-1")
        uniq, inv = np.unique(keys.astype(np.int64, copy=False), return_inverse=True)
        m = slice_aggs[name]
        for i, key in enumerate(uniq.tolist()):
            mask = inv == i
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            cur = m.get(int(key))
            if cur is None:
                cur = _SliceAgg()
                m[int(key)] = cur
            cur.n += cnt
            cur.correct += int(correct_b[mask].sum())
            cur.top3 += int(top3_b[mask].sum())
            cur.ce_sum += float(ce[mask].sum())
            cur.loc_mse_sum += float(loc_mse[mask].sum())

    for batch_cpu in tqdm(loader, desc=f"eval_report[{split}]"):
        # Avoid copying x_cont_raw to GPU; it's only for slice grouping.
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch_cpu.items() if k != "x_cont_raw"}
        y_type = batch["y_type"]
        y_loc = batch["y_loc"]

        out = model(batch)
        logits = out["type_logits"]
        ce = F.cross_entropy(logits, y_type)
        ce_per = F.cross_entropy(logits, y_type, reduction="none").detach().cpu().numpy().astype(np.float64, copy=False)

        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        correct_b = pred.eq(y_type)
        correct_b_cpu = correct_b.detach().cpu().numpy().astype(np.int64, copy=False)

        correct += int(correct_b.sum().item())
        topk = torch.topk(probs, k=min(3, probs.size(-1)), dim=-1).indices
        top3_b = (topk == y_type.unsqueeze(-1)).any(dim=-1)
        top3 += int(top3_b.sum().item())
        top3_b_cpu = top3_b.detach().cpu().numpy().astype(np.int64, copy=False)

        conf = probs.max(dim=-1).values.detach().float().cpu().numpy()
        corr = correct_b.detach().cpu().numpy().astype(np.int64, copy=False)
        # Bin index: floor(conf * n_bins), clamped to n_bins-1.
        bin_idx = np.minimum((conf * float(calibration_bins)).astype(np.int64), calibration_bins - 1)
        bin_count += np.bincount(bin_idx, minlength=calibration_bins).astype(np.int64, copy=False)
        bin_conf_sum += np.bincount(bin_idx, weights=conf, minlength=calibration_bins).astype(np.float64, copy=False)
        bin_acc_sum += np.bincount(bin_idx, weights=corr, minlength=calibration_bins).astype(np.float64, copy=False)

        # Brier score (multiclass): sum_i (p_i - y_i)^2
        p_true = probs.gather(-1, y_type.unsqueeze(-1)).squeeze(-1)
        brier = (probs.pow(2).sum(dim=-1) - 2.0 * p_true + 1.0).mean()

        desc_logits = out.get("desc_logits")
        if desc_logits is not None:
            y_desc = batch.get("y_desc")
            if y_desc is None:
                raise ReportError(
                    "Model returned desc_logits but batch is missing y_desc. "
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
            loc_nll = torch.tensor(float("nan"), device=dev)

        loc_mean_ft = torch.stack(
            [loc_mean[:, 0] * px_std + px_mean, loc_mean[:, 1] * pz_std + pz_mean], dim=1
        )
        y_loc_ft = torch.stack([y_loc[:, 0] * px_std + px_mean, y_loc[:, 1] * pz_std + pz_mean], dim=1)
        loc_mse_per = ((loc_mean_ft - y_loc_ft) ** 2).mean(dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)
        mse_ft_sum += float(loc_mse_per.sum())

        bs = int(y_type.size(0))
        n += bs
        ce_sum += float(ce.item()) * bs
        brier_sum += float(brier.item()) * bs
        if torch.isfinite(loc_nll):
            loc_nll_sum += float(loc_nll.item()) * bs
            loc_nll_count += bs

        # Slice aggregation (CPU-only keys derived from raw context/cats).
        x_cont_raw_t = batch_cpu.get("x_cont_raw")
        if x_cont_raw_t is None:
            raise ReportError("Internal error: x_cont_raw missing despite return_raw_cont=True.")
        x_cont_raw = x_cont_raw_t.numpy().astype(np.int64, copy=False)
        x_cat = batch_cpu["x_cat"].numpy().astype(np.int64, copy=False)

        balls = x_cont_raw[:, idx_balls]
        strikes = x_cont_raw[:, idx_strikes]
        inning = x_cont_raw[:, idx_inning]
        score_diff = x_cont_raw[:, idx_score_diff]
        on_1b = x_cont_raw[:, idx_on_1b]
        on_2b = x_cont_raw[:, idx_on_2b]
        on_3b = x_cont_raw[:, idx_on_3b]
        pitch_number = x_cont_raw[:, idx_pitch_number]
        stand_id = x_cat[:, 2]
        p_throws_id = x_cat[:, 3]

        _update_slice(
            "count",
            balls * 10 + strikes,
            correct_b=correct_b_cpu,
            top3_b=top3_b_cpu,
            ce=ce_per,
            loc_mse=loc_mse_per,
        )
        _update_slice(
            "pitch_number_bucket",
            _bucket_pitch_number(pitch_number),
            correct_b=correct_b_cpu,
            top3_b=top3_b_cpu,
            ce=ce_per,
            loc_mse=loc_mse_per,
        )
        _update_slice(
            "inning_bucket",
            _bucket_inning(inning),
            correct_b=correct_b_cpu,
            top3_b=top3_b_cpu,
            ce=ce_per,
            loc_mse=loc_mse_per,
        )
        _update_slice(
            "runners_state",
            _runners_state(on_1b, on_2b, on_3b),
            correct_b=correct_b_cpu,
            top3_b=top3_b_cpu,
            ce=ce_per,
            loc_mse=loc_mse_per,
        )
        _update_slice(
            "score_bucket",
            _bucket_score_diff(score_diff),
            correct_b=correct_b_cpu,
            top3_b=top3_b_cpu,
            ce=ce_per,
            loc_mse=loc_mse_per,
        )
        _update_slice(
            "stand_id",
            stand_id,
            correct_b=correct_b_cpu,
            top3_b=top3_b_cpu,
            ce=ce_per,
            loc_mse=loc_mse_per,
        )
        _update_slice(
            "p_throws_id",
            p_throws_id,
            correct_b=correct_b_cpu,
            top3_b=top3_b_cpu,
            ce=ce_per,
            loc_mse=loc_mse_per,
        )

    bins: list[CalibrationBin] = []
    ece = 0.0
    for b in range(calibration_bins):
        cnt = int(bin_count[b])
        if cnt == 0:
            bins.append(CalibrationBin(count=0, acc=float("nan"), conf=float("nan")))
            continue
        acc_b = float(bin_acc_sum[b] / float(cnt))
        conf_b = float(bin_conf_sum[b] / float(cnt))
        bins.append(CalibrationBin(count=cnt, acc=acc_b, conf=conf_b))
        ece += abs(acc_b - conf_b) * (float(cnt) / max(1.0, float(n)))

    def _slice_rows(name: str) -> list[dict[str, Any]]:
        m = slice_aggs[name]
        rows: list[dict[str, Any]] = []
        for k, agg in m.items():
            rows.append(
                {
                    "key": int(k),
                    "n": int(agg.n),
                    "acc": float(agg.correct / max(1, agg.n)),
                    "acc_top3": float(agg.top3 / max(1, agg.n)),
                    "ce": float(agg.ce_sum / max(1, agg.n)),
                    "loc_rmse_ft": float((agg.loc_mse_sum / max(1, agg.n)) ** 0.5),
                }
            )
        rows.sort(key=lambda r: (-r["n"], r["key"]))
        return rows

    slices_out = {name: _slice_rows(name) for name in sorted(slice_aggs.keys())}

    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "model": model_name,
        "split": str(split),
        "n": float(n),
        "acc": float(correct / max(1, n)),
        "acc_top3": float(top3 / max(1, n)),
        "ce": float(ce_sum / max(1, n)),
        "brier": float(brier_sum / max(1, n)),
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
        "calibration": {
            "n_bins": int(calibration_bins),
            "ece": float(ece),
            "bins": [b.__dict__ for b in bins],
        },
        "slices": slices_out,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_report(
    paths: Paths,
    *,
    run_id: str,
    split: str = "valid",
    baseline_split: str = "train",
    out_dir: Path | None = None,
    device: str | None = "auto",
    batch_size: int = 4096,
    baseline_batch_size: int = 8192,
    baseline_alpha: float = 1.0,
    calibration_bins: int = 15,
) -> dict[str, Any]:
    if split not in {"train", "valid"}:
        raise ValueError("--split must be one of: train, valid")
    if baseline_split not in {"train", "valid"}:
        raise ValueError("--baseline-split must be one of: train, valid")

    run_dir = resolve_run_dir(paths.runs, run_id=run_id)
    if out_dir is None:
        out_dir = run_dir / "report"

    model_metrics = evaluate_model_with_calibration(
        paths,
        run_id=run_id,
        split=split,
        batch_size=batch_size,
        calibration_bins=calibration_bins,
        device=device,
    )

    fitted = fit_baselines(paths, split=baseline_split, batch_size=baseline_batch_size)
    baseline_metrics = evaluate_baselines(
        fitted,
        paths,
        split=split,
        alpha=baseline_alpha,
        batch_size=baseline_batch_size,
    )

    payload: dict[str, Any] = {
        "run_id": str(run_dir.name),
        "split": str(split),
        "baseline_split": str(baseline_split),
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
    }

    write_json(out_dir / "model_metrics.json", model_metrics)
    if "slices" in model_metrics:
        write_json(
            out_dir / "slices.json",
            {
                "run_id": str(run_dir.name),
                "split": str(split),
                "slices": model_metrics["slices"],
            },
        )
    write_json(out_dir / "baselines.json", baseline_metrics)
    write_json(out_dir / "report.json", payload)
    return payload
