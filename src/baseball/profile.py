from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseball.config import Paths
from baseball.data.vocab import Vocab
from baseball.training.io import load_prepared
from baseball.training.mdn import mdn_mean
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


class ProfileError(RuntimeError):
    pass


ProfileBy = Literal[
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
]


@dataclass
class _Agg:
    n: int
    prob_sum: np.ndarray  # [T]
    true_counts: np.ndarray  # [T]
    pred_loc_sum: np.ndarray  # [2] feet
    true_loc_sum: np.ndarray  # [2] feet


def _safe_col(s: str) -> str:
    s = str(s)
    if s == "<OOV>":
        s = "OOV"
    return "".join(c if c.isalnum() else "_" for c in s)


def _id_to_token(vocab: Vocab) -> list[str]:
    # 0 is OOV.
    size = vocab.size
    id_to_tok = ["<OOV>"] * size
    for tok, idx in vocab.token_to_id.items():
        if 0 <= idx < size:
            id_to_tok[idx] = tok
    return id_to_tok


def _last_nonzero(hist_type: np.ndarray) -> np.ndarray:
    """
    hist_type: [B,L] int64, padded with 0.
    Returns: [B] int64, last non-zero id or 0 if no history.
    """

    if hist_type.ndim != 2:
        raise ProfileError("hist_type must be rank-2 [B,L]")
    if hist_type.shape[1] == 0:
        return np.zeros((hist_type.shape[0],), dtype=np.int64)

    mask = hist_type != 0
    has = mask.any(axis=1)
    rev = mask[:, ::-1]
    # argmax returns 0 even when all False; we correct using `has`.
    idx_from_end = np.argmax(rev, axis=1).astype(np.int64)
    last_idx = (hist_type.shape[1] - 1 - idx_from_end).astype(np.int64)
    out = hist_type[np.arange(hist_type.shape[0]), last_idx].astype(np.int64, copy=False)
    out[~has] = 0
    return out


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


def _encode_key(
    *,
    by: ProfileBy,
    pitcher_id: np.ndarray,
    balls: np.ndarray | None = None,
    strikes: np.ndarray | None = None,
    stand_id: np.ndarray | None = None,
    runners_state: np.ndarray | None = None,
    inning_bucket: np.ndarray | None = None,
    score_bucket: np.ndarray | None = None,
    batter_cluster: np.ndarray | None = None,
    prev_type: np.ndarray | None = None,
    prev_desc: np.ndarray | None = None,
) -> np.ndarray:
    pitcher_id = pitcher_id.astype(np.int64, copy=False)
    if by == "pitcher":
        return pitcher_id

    if balls is None or strikes is None:
        raise ProfileError(f"ProfileBy={by} requires balls+strikes")
    balls = balls.astype(np.int64, copy=False)
    strikes = strikes.astype(np.int64, copy=False)

    if by == "pitcher_count":
        return pitcher_id * 100 + balls * 10 + strikes

    if by == "pitcher_count_prev":
        if prev_type is None:
            raise ProfileError("ProfileBy=pitcher_count_prev requires prev_type")
        prev_type = prev_type.astype(np.int64, copy=False)
        return pitcher_id * 100_000 + balls * 10_000 + strikes * 1_000 + prev_type

    if by == "pitcher_count_prev_outcome":
        if prev_type is None:
            raise ProfileError("ProfileBy=pitcher_count_prev_outcome requires prev_type")
        if prev_desc is None:
            raise ProfileError("ProfileBy=pitcher_count_prev_outcome requires prev_desc")
        prev_type = prev_type.astype(np.int64, copy=False)
        prev_desc = prev_desc.astype(np.int64, copy=False)
        # Encode into a single int64 key. Constants are chosen to avoid collisions
        # for realistic vocab sizes while staying within int64 range.
        return (
            pitcher_id * 1_000_000_000
            + balls * 100_000_000
            + strikes * 10_000_000
            + prev_type * 10_000
            + prev_desc
        )

    if by in {
        "pitcher_situation",
        "pitcher_situation_prev",
        "pitcher_situation_prev_outcome",
        "pitcher_situation_batter_cluster",
        "pitcher_situation_batter_cluster_prev",
        "pitcher_situation_batter_cluster_prev_outcome",
    }:
        if stand_id is None:
            raise ProfileError(f"ProfileBy={by} requires stand_id")
        if runners_state is None:
            raise ProfileError(f"ProfileBy={by} requires runners_state")
        if inning_bucket is None:
            raise ProfileError(f"ProfileBy={by} requires inning_bucket")
        if score_bucket is None:
            raise ProfileError(f"ProfileBy={by} requires score_bucket")

        include_prev_type = by in {
            "pitcher_situation_prev",
            "pitcher_situation_prev_outcome",
            "pitcher_situation_batter_cluster_prev",
            "pitcher_situation_batter_cluster_prev_outcome",
        }
        include_prev_desc = by in {"pitcher_situation_prev_outcome", "pitcher_situation_batter_cluster_prev_outcome"}
        include_batter_cluster = by in {
            "pitcher_situation_batter_cluster",
            "pitcher_situation_batter_cluster_prev",
            "pitcher_situation_batter_cluster_prev_outcome",
        }

        if include_prev_type and prev_type is None:
            raise ProfileError(f"ProfileBy={by} requires prev_type")
        if include_prev_desc and prev_desc is None:
            raise ProfileError(f"ProfileBy={by} requires prev_desc")
        if include_batter_cluster and batter_cluster is None:
            raise ProfileError(f"ProfileBy={by} requires batter_cluster")

        stand_id = stand_id.astype(np.int64, copy=False)
        runners_state = runners_state.astype(np.int64, copy=False)
        inning_bucket = inning_bucket.astype(np.int64, copy=False)
        score_bucket = score_bucket.astype(np.int64, copy=False)
        prev_type = prev_type.astype(np.int64, copy=False) if prev_type is not None else None
        prev_desc = prev_desc.astype(np.int64, copy=False) if prev_desc is not None else None
        batter_cluster = batter_cluster.astype(np.int64, copy=False) if batter_cluster is not None else None

        # Conservative base sizes to avoid collisions while keeping int64 headroom.
        BASE_BALLS = 10
        BASE_STRIKES = 10
        BASE_STAND = 10
        BASE_RUNNERS = 10
        BASE_INNING_BUCKET = 10
        BASE_SCORE_BUCKET = 10
        BASE_PREV_TYPE = 1_000
        BASE_PREV_DESC = 1_000
        BASE_BATTER_CLUSTER = 100

        # Minimal sanity checks (fail fast, don't silently clip).
        if int(balls.min()) < 0 or int(balls.max()) >= BASE_BALLS:
            raise ProfileError(f"balls out of expected range [0,{BASE_BALLS - 1}]")
        if int(strikes.min()) < 0 or int(strikes.max()) >= BASE_STRIKES:
            raise ProfileError(f"strikes out of expected range [0,{BASE_STRIKES - 1}]")
        if int(stand_id.min()) < 0 or int(stand_id.max()) >= BASE_STAND:
            raise ProfileError(f"stand_id out of expected range [0,{BASE_STAND - 1}]")
        if int(runners_state.min()) < 0 or int(runners_state.max()) >= BASE_RUNNERS:
            raise ProfileError(f"runners_state out of expected range [0,{BASE_RUNNERS - 1}]")
        if int(inning_bucket.min()) < 0 or int(inning_bucket.max()) >= BASE_INNING_BUCKET:
            raise ProfileError(f"inning_bucket out of expected range [0,{BASE_INNING_BUCKET - 1}]")
        if int(score_bucket.min()) < 0 or int(score_bucket.max()) >= BASE_SCORE_BUCKET:
            raise ProfileError(f"score_bucket out of expected range [0,{BASE_SCORE_BUCKET - 1}]")
        if include_prev_type and prev_type is not None and (int(prev_type.min()) < 0 or int(prev_type.max()) >= BASE_PREV_TYPE):
            raise ProfileError(f"prev_type_id out of expected range [0,{BASE_PREV_TYPE - 1}]")
        if include_prev_desc and prev_desc is not None and (int(prev_desc.min()) < 0 or int(prev_desc.max()) >= BASE_PREV_DESC):
            raise ProfileError(f"prev_desc_id out of expected range [0,{BASE_PREV_DESC - 1}]")
        if include_batter_cluster and batter_cluster is not None and (
            int(batter_cluster.min()) < 0 or int(batter_cluster.max()) >= BASE_BATTER_CLUSTER
        ):
            raise ProfileError(f"batter_cluster out of expected range [0,{BASE_BATTER_CLUSTER - 1}]")

        # Pack digits from least-significant to most-significant:
        #   [prev_desc?][prev_type?][score_bucket][inning_bucket][runners][stand][strikes][balls][batter_cluster?]
        offset = np.zeros_like(balls, dtype=np.int64)
        mult = 1
        if include_prev_desc:
            assert prev_desc is not None
            offset += prev_desc
            mult *= BASE_PREV_DESC
        if include_prev_type:
            assert prev_type is not None
            offset += mult * prev_type
            mult *= BASE_PREV_TYPE
        offset += mult * score_bucket
        mult *= BASE_SCORE_BUCKET
        offset += mult * inning_bucket
        mult *= BASE_INNING_BUCKET
        offset += mult * runners_state
        mult *= BASE_RUNNERS
        offset += mult * stand_id
        mult *= BASE_STAND
        offset += mult * strikes
        mult *= BASE_STRIKES
        offset += mult * balls
        mult *= BASE_BALLS

        stride_no_cluster = int(mult)
        if include_batter_cluster:
            assert batter_cluster is not None
            offset += mult * batter_cluster
            mult *= BASE_BATTER_CLUSTER

        stride = int(mult)
        return pitcher_id * stride + offset

    raise ProfileError(f"Unknown ProfileBy: {by}")


def _decode_key(key: int, *, by: ProfileBy) -> dict[str, int]:
    if by == "pitcher":
        return {"pitcher_id": int(key)}

    if by == "pitcher_count":
        pitcher_id = int(key // 100)
        rest = int(key % 100)
        balls = int(rest // 10)
        strikes = int(rest % 10)
        return {"pitcher_id": pitcher_id, "balls": balls, "strikes": strikes}

    if by == "pitcher_count_prev":
        pitcher_id = int(key // 100_000)
        rest = int(key % 100_000)
        balls = int(rest // 10_000)
        rest = int(rest % 10_000)
        strikes = int(rest // 1_000)
        prev_type = int(rest % 1_000)
        return {"pitcher_id": pitcher_id, "balls": balls, "strikes": strikes, "prev_type_id": prev_type}

    if by == "pitcher_count_prev_outcome":
        pitcher_id = int(key // 1_000_000_000)
        rest = int(key % 1_000_000_000)
        balls = int(rest // 100_000_000)
        rest = int(rest % 100_000_000)
        strikes = int(rest // 10_000_000)
        rest = int(rest % 10_000_000)
        prev_type = int(rest // 10_000)
        prev_desc = int(rest % 10_000)
        return {
            "pitcher_id": pitcher_id,
            "balls": balls,
            "strikes": strikes,
            "prev_type_id": prev_type,
            "prev_desc_id": prev_desc,
        }

    if by in {
        "pitcher_situation",
        "pitcher_situation_prev",
        "pitcher_situation_prev_outcome",
        "pitcher_situation_batter_cluster",
        "pitcher_situation_batter_cluster_prev",
        "pitcher_situation_batter_cluster_prev_outcome",
    }:
        BASE_BALLS = 10
        BASE_STRIKES = 10
        BASE_STAND = 10
        BASE_RUNNERS = 10
        BASE_INNING_BUCKET = 10
        BASE_SCORE_BUCKET = 10
        BASE_PREV_TYPE = 1_000
        BASE_PREV_DESC = 1_000
        BASE_BATTER_CLUSTER = 100

        include_prev_type = by in {
            "pitcher_situation_prev",
            "pitcher_situation_prev_outcome",
            "pitcher_situation_batter_cluster_prev",
            "pitcher_situation_batter_cluster_prev_outcome",
        }
        include_prev_desc = by in {"pitcher_situation_prev_outcome", "pitcher_situation_batter_cluster_prev_outcome"}
        include_batter_cluster = by in {
            "pitcher_situation_batter_cluster",
            "pitcher_situation_batter_cluster_prev",
            "pitcher_situation_batter_cluster_prev_outcome",
        }

        stride_no_cluster = (
            BASE_SCORE_BUCKET
            * BASE_INNING_BUCKET
            * BASE_RUNNERS
            * BASE_STAND
            * BASE_STRIKES
            * BASE_BALLS
        )
        if include_prev_type:
            stride_no_cluster *= BASE_PREV_TYPE
        if include_prev_desc:
            stride_no_cluster *= BASE_PREV_DESC

        stride = stride_no_cluster * (BASE_BATTER_CLUSTER if include_batter_cluster else 1)
        pitcher_id = int(key // stride)
        rest = int(key % stride)

        batter_cluster = None
        if include_batter_cluster:
            batter_cluster = int(rest // stride_no_cluster)
            rest = int(rest % stride_no_cluster)

        prev_desc = None
        prev_type = None
        if include_prev_desc:
            rest, prev_desc = divmod(rest, BASE_PREV_DESC)
        if include_prev_type:
            rest, prev_type = divmod(rest, BASE_PREV_TYPE)
        rest, score_bucket = divmod(rest, BASE_SCORE_BUCKET)
        rest, inning_bucket = divmod(rest, BASE_INNING_BUCKET)
        rest, runners_state = divmod(rest, BASE_RUNNERS)
        rest, stand_id = divmod(rest, BASE_STAND)
        rest, strikes = divmod(rest, BASE_STRIKES)
        rest, balls = divmod(rest, BASE_BALLS)

        out = {
            "pitcher_id": pitcher_id,
            "balls": int(balls),
            "strikes": int(strikes),
            "stand_id": int(stand_id),
            "runners_state": int(runners_state),
            "inning_bucket": int(inning_bucket),
            "score_bucket": int(score_bucket),
        }
        if include_prev_type:
            assert prev_type is not None
            out["prev_type_id"] = int(prev_type)
        if include_prev_desc:
            assert prev_desc is not None
            out["prev_desc_id"] = int(prev_desc)
        if include_batter_cluster:
            assert batter_cluster is not None
            out["batter_cluster"] = int(batter_cluster)
        if rest != 0:
            # If this triggers, encoding/decoding bases are inconsistent.
            raise ProfileError(f"Key decode did not terminate cleanly (rest={rest}) for ProfileBy={by}")
        return out

    raise ProfileError(f"Unknown ProfileBy: {by}")


def build_pitcher_profiles(
    paths: Paths,
    *,
    run_id: str,
    split: str,
    by: ProfileBy = "pitcher_count",
    batch_size: int = 8192,
    min_n: int = 25,
    batter_clusters: int = 32,
    device: str | None = None,
) -> pl.DataFrame:
    if split not in {"train", "valid"}:
        raise ValueError("--split must be one of: train, valid")
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if min_n < 1:
        raise ValueError("--min-n must be >= 1")
    if int(batter_clusters) < 2:
        raise ValueError("--batter-clusters must be >= 2")

    prepared = load_prepared(paths.data_prepared)
    meta = prepared.meta
    norms = meta.get("norms")
    if not isinstance(norms, dict):
        raise ProfileError("Prepared meta.json missing norms")
    px_mean = float(norms["plate_x"]["mean"])
    px_std = float(norms["plate_x"]["std"])
    pz_mean = float(norms["plate_z"]["mean"])
    pz_std = float(norms["plate_z"]["std"])

    cont_features = meta.get("cont_features")
    if not isinstance(cont_features, list):
        raise ProfileError("Prepared meta.json missing cont_features")
    cont_features = [str(x) for x in cont_features]
    try:
        idx_balls = cont_features.index("balls")
        idx_strikes = cont_features.index("strikes")
    except ValueError as e:
        raise ProfileError("cont_features must include balls and strikes") from e

    idx_inning = None
    idx_score_diff = None
    idx_on_1b = None
    idx_on_2b = None
    idx_on_3b = None
    if str(by).startswith("pitcher_situation"):
        try:
            idx_inning = cont_features.index("inning")
            idx_score_diff = cont_features.index("score_diff")
            idx_on_1b = cont_features.index("on_1b_occ")
            idx_on_2b = cont_features.index("on_2b_occ")
            idx_on_3b = cont_features.index("on_3b_occ")
        except ValueError as e:
            raise ProfileError(
                "cont_features must include inning, score_diff, on_1b_occ, on_2b_occ, on_3b_occ "
                f"for ProfileBy={by}"
            ) from e

    pitch_type_vocab = Vocab.load(prepared.vocabs_dir / "pitch_type.json")
    desc_vocab_path = prepared.vocabs_dir / "description.json"
    desc_vocab = Vocab.load(desc_vocab_path) if desc_vocab_path.exists() else None
    pitcher_vocab = Vocab.load(prepared.vocabs_dir / "pitcher.json")
    id_to_pitch_type = _id_to_token(pitch_type_vocab)
    id_to_desc = _id_to_token(desc_vocab) if desc_vocab is not None else None
    id_to_pitcher = _id_to_token(pitcher_vocab)

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
        raise ProfileError(f"Unsupported model for profiling: {model_name}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    batter_cluster_map: np.ndarray | None = None
    if by in {
        "pitcher_situation_batter_cluster",
        "pitcher_situation_batter_cluster_prev",
        "pitcher_situation_batter_cluster_prev_outcome",
    }:
        k = int(batter_clusters)
        if k >= 99:
            raise ProfileError("--batter-clusters must be < 99 (encoding base is 100).")
        if not hasattr(model, "batter_emb"):
            raise ProfileError("Selected model does not expose batter_emb for clustering.")
        emb = model.batter_emb.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        if emb.ndim != 2 or emb.shape[0] < 2:
            raise ProfileError(f"Unexpected batter embedding shape: {emb.shape}")
        if (emb.shape[0] - 1) < k:
            raise ProfileError(
                f"Not enough batters to cluster: n_batters={emb.shape[0] - 1} < k={k} (excluding OOV)."
            )
        # Cluster only non-OOV IDs; reserve cluster_id=0 for OOV.
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = km.fit_predict(emb[1:])
        batter_cluster_map = np.zeros((emb.shape[0],), dtype=np.int64)
        batter_cluster_map[1:] = labels.astype(np.int64, copy=False) + 1

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
    dl_kwargs: dict[str, Any] = {
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "num_workers": 2,
        "prefetch_factor": 2,
    }
    loader = DataLoader(ds, batch_size=None, shuffle=False, **dl_kwargs)

    acc: dict[int, _Agg] = {}

    for batch in tqdm(loader, desc=f"profile[{by}]"):
        x_cat_cpu = batch["x_cat"]
        x_cont_raw = batch.get("x_cont_raw")
        if x_cont_raw is None:
            raise ProfileError("Streaming dataset did not return x_cont_raw; internal error.")

        pitcher_id = x_cat_cpu[:, 0].numpy().astype(np.int64, copy=False)
        balls = x_cont_raw[:, idx_balls].numpy().astype(np.int64, copy=False)
        strikes = x_cont_raw[:, idx_strikes].numpy().astype(np.int64, copy=False)

        prev_type = None
        prev_desc = None
        if by in {
            "pitcher_count_prev",
            "pitcher_count_prev_outcome",
            "pitcher_situation_prev",
            "pitcher_situation_prev_outcome",
            "pitcher_situation_batter_cluster_prev",
            "pitcher_situation_batter_cluster_prev_outcome",
        }:
            prev_type = _last_nonzero(batch["hist_type"].numpy().astype(np.int64, copy=False))

        if by in {
            "pitcher_count_prev_outcome",
            "pitcher_situation_prev_outcome",
            "pitcher_situation_batter_cluster_prev_outcome",
        }:
            if "hist_desc" not in batch:
                raise ProfileError(
                    f"ProfileBy={by} requires hist_desc, but the dataset did not provide it. "
                    "Re-run `python -m baseball prepare` with schema_version >= 4."
                )
            prev_desc = _last_nonzero(batch["hist_desc"].numpy().astype(np.int64, copy=False))

        stand_id = None
        runners_state = None
        inning_bucket = None
        score_bucket = None
        batter_cluster = None
        if str(by).startswith("pitcher_situation"):
            if idx_inning is None or idx_score_diff is None or idx_on_1b is None or idx_on_2b is None or idx_on_3b is None:
                raise ProfileError(f"Internal error: missing cont feature indices for ProfileBy={by}")

            stand_id = x_cat_cpu[:, 2].numpy().astype(np.int64, copy=False)

            inning = x_cont_raw[:, idx_inning].numpy().astype(np.int64, copy=False)
            score_diff = x_cont_raw[:, idx_score_diff].numpy().astype(np.int64, copy=False)
            on_1b = x_cont_raw[:, idx_on_1b].numpy().astype(np.int64, copy=False)
            on_2b = x_cont_raw[:, idx_on_2b].numpy().astype(np.int64, copy=False)
            on_3b = x_cont_raw[:, idx_on_3b].numpy().astype(np.int64, copy=False)

            runners_state = (on_1b + 2 * on_2b + 4 * on_3b).astype(np.int64, copy=False)
            inning_bucket = _bucket_inning(inning)
            score_bucket = _bucket_score_diff(score_diff)

            if by in {
                "pitcher_situation_batter_cluster",
                "pitcher_situation_batter_cluster_prev",
                "pitcher_situation_batter_cluster_prev_outcome",
            }:
                if batter_cluster_map is None:
                    raise ProfileError("Internal error: batter_cluster_map was not initialized.")
                batter_id = x_cat_cpu[:, 1].numpy().astype(np.int64, copy=False)
                batter_cluster = batter_cluster_map[batter_id]

        keys = _encode_key(
            by=by,
            pitcher_id=pitcher_id,
            balls=balls,
            strikes=strikes,
            stand_id=stand_id,
            runners_state=runners_state,
            inning_bucket=inning_bucket,
            score_bucket=score_bucket,
            batter_cluster=batter_cluster,
            prev_type=prev_type,
            prev_desc=prev_desc,
        )
        uniq, inv = np.unique(keys, return_inverse=True)

        x_cat = batch["x_cat"].to(dev, non_blocking=True)
        x_cont = batch["x_cont"].to(dev, non_blocking=True)
        hist_type = batch["hist_type"].to(dev, non_blocking=True)
        hist_desc_t = batch.get("hist_desc")
        if hist_desc_t is not None:
            hist_desc_t = hist_desc_t.to(dev, non_blocking=True)
        hist_x = batch["hist_x"].to(dev, non_blocking=True)
        hist_z = batch["hist_z"].to(dev, non_blocking=True)

        with torch.no_grad():
            mb = {"x_cat": x_cat, "x_cont": x_cont, "hist_type": hist_type, "hist_x": hist_x, "hist_z": hist_z}
            if hist_desc_t is not None:
                mb["hist_desc"] = hist_desc_t
            out = model(mb)
            type_logits = out["type_logits"]
            probs = F.softmax(type_logits, dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)

            if model_name in {
                "transformer_mdn",
                "transformer_mdn_mt",
                "transformer_mdn_v2",
                "transformer_mdn_state",
                "transformer_mdn_state_mt",
            }:
                loc_mean_norm = mdn_mean(out["mdn_logit_pi"], out["mdn_mu"]).detach().cpu().numpy()
            else:
                loc_mean_norm = out["loc_mu"].detach().cpu().numpy()

        y_type = batch["y_type"].numpy().astype(np.int64, copy=False)
        y_loc = batch["y_loc"].numpy().astype(np.float64, copy=False)  # normalized

        # Denormalize to feet (plate_x, plate_z).
        pred_loc_ft = np.stack(
            [loc_mean_norm[:, 0] * px_std + px_mean, loc_mean_norm[:, 1] * pz_std + pz_mean],
            axis=1,
        ).astype(np.float64, copy=False)
        true_loc_ft = np.stack(
            [y_loc[:, 0] * px_std + px_mean, y_loc[:, 1] * pz_std + pz_mean],
            axis=1,
        ).astype(np.float64, copy=False)

        T = int(cfg.n_pitch_types)
        prob_sum = np.zeros((uniq.size, T), dtype=np.float64)
        np.add.at(prob_sum, inv, probs)

        true_counts = np.zeros((uniq.size, T), dtype=np.int64)
        np.add.at(true_counts, (inv, y_type), 1)

        pred_loc_sum = np.zeros((uniq.size, 2), dtype=np.float64)
        np.add.at(pred_loc_sum, inv, pred_loc_ft)

        true_loc_sum = np.zeros((uniq.size, 2), dtype=np.float64)
        np.add.at(true_loc_sum, inv, true_loc_ft)

        n_per = np.bincount(inv, minlength=uniq.size).astype(np.int64, copy=False)

        for i, key in enumerate(uniq.tolist()):
            key_i = int(key)
            n_i = int(n_per[i])
            if n_i <= 0:
                continue
            cur = acc.get(key_i)
            if cur is None:
                acc[key_i] = _Agg(
                    n=n_i,
                    prob_sum=prob_sum[i],
                    true_counts=true_counts[i],
                    pred_loc_sum=pred_loc_sum[i],
                    true_loc_sum=true_loc_sum[i],
                )
            else:
                cur.n += n_i
                cur.prob_sum += prob_sum[i]
                cur.true_counts += true_counts[i]
                cur.pred_loc_sum += pred_loc_sum[i]
                cur.true_loc_sum += true_loc_sum[i]

    if not acc:
        raise ProfileError("No profile rows accumulated; is the dataset empty?")

    rows: list[dict[str, Any]] = []
    eps = 1e-12

    for key, a in acc.items():
        if a.n < min_n:
            continue
        dec = _decode_key(key, by=by)
        pitcher_id = int(dec["pitcher_id"])
        pitcher_tok = id_to_pitcher[pitcher_id] if 0 <= pitcher_id < len(id_to_pitcher) else "<OOV>"

        pred = (a.prob_sum / max(1, a.n)).astype(np.float64, copy=False)
        emp = (a.true_counts / max(1, a.n)).astype(np.float64, copy=False)

        # Group-level cross entropy H(emp, pred) (lower is better).
        ce = float(-(emp * np.log(np.clip(pred, eps, 1.0))).sum())

        pred_loc = (a.pred_loc_sum / max(1, a.n)).astype(np.float64, copy=False)
        true_loc = (a.true_loc_sum / max(1, a.n)).astype(np.float64, copy=False)

        rec: dict[str, Any] = {
            "pitcher_id": pitcher_id,
            "pitcher": pitcher_tok,
            "n": int(a.n),
            "type_ce_group": ce,
            "pred_plate_x": float(pred_loc[0]),
            "pred_plate_z": float(pred_loc[1]),
            "emp_plate_x": float(true_loc[0]),
            "emp_plate_z": float(true_loc[1]),
        }
        if "balls" in dec:
            rec["balls"] = int(dec["balls"])
        if "strikes" in dec:
            rec["strikes"] = int(dec["strikes"])
        if "stand_id" in dec:
            sid = int(dec["stand_id"])
            rec["stand_id"] = sid
            rec["stand"] = {0: "<UNK>", 1: "R", 2: "L", 3: "S"}.get(sid, "<UNK>")
        if "runners_state" in dec:
            rs = int(dec["runners_state"])
            rec["runners_state"] = rs
            rec["on_1b_occ"] = int(rs & 1)
            rec["on_2b_occ"] = int((rs >> 1) & 1)
            rec["on_3b_occ"] = int((rs >> 2) & 1)
        if "inning_bucket" in dec:
            ib = int(dec["inning_bucket"])
            rec["inning_bucket"] = ib
            rec["inning_bucket_label"] = {0: "1-3", 1: "4-6", 2: "7-8", 3: "9+"}.get(ib, "UNK")
        if "score_bucket" in dec:
            sb = int(dec["score_bucket"])
            rec["score_bucket"] = sb
            rec["score_bucket_label"] = {
                0: "<=-3",
                1: "-2",
                2: "-1",
                3: "0",
                4: "+1",
                5: "+2",
                6: ">=+3",
            }.get(sb, "UNK")
        if "batter_cluster" in dec:
            rec["batter_cluster"] = int(dec["batter_cluster"])
        if "prev_type_id" in dec:
            pid = int(dec["prev_type_id"])
            rec["prev_pitch_type_id"] = pid
            rec["prev_pitch_type"] = id_to_pitch_type[pid] if 0 <= pid < len(id_to_pitch_type) else "<OOV>"
        if "prev_desc_id" in dec:
            did = int(dec["prev_desc_id"])
            rec["prev_description_id"] = did
            if id_to_desc is None:
                rec["prev_description"] = "<MISSING_DESC_VOCAB>"
            else:
                rec["prev_description"] = id_to_desc[did] if 0 <= did < len(id_to_desc) else "<OOV>"

        for tid in range(int(cfg.n_pitch_types)):
            tok = id_to_pitch_type[tid] if 0 <= tid < len(id_to_pitch_type) else f"id_{tid}"
            col = _safe_col(tok)
            rec[f"pred_prob_{col}"] = float(pred[tid])
            rec[f"emp_prob_{col}"] = float(emp[tid])

        rows.append(rec)

    if not rows:
        raise ProfileError(f"No profile groups met min_n={min_n}.")

    df = pl.DataFrame(rows)
    # Deterministic sort for downstream diffs.
    sort_cols = ["pitcher_id"]
    if by in {
        "pitcher_count",
        "pitcher_count_prev",
        "pitcher_count_prev_outcome",
        "pitcher_situation",
        "pitcher_situation_prev",
        "pitcher_situation_prev_outcome",
        "pitcher_situation_batter_cluster",
        "pitcher_situation_batter_cluster_prev",
        "pitcher_situation_batter_cluster_prev_outcome",
    }:
        sort_cols.extend(["balls", "strikes"])
    if str(by).startswith("pitcher_situation"):
        sort_cols.extend(["stand_id", "runners_state", "inning_bucket", "score_bucket"])
    if by in {
        "pitcher_situation_batter_cluster",
        "pitcher_situation_batter_cluster_prev",
        "pitcher_situation_batter_cluster_prev_outcome",
    }:
        sort_cols.append("batter_cluster")
    if by in {
        "pitcher_count_prev",
        "pitcher_count_prev_outcome",
        "pitcher_situation_prev",
        "pitcher_situation_prev_outcome",
        "pitcher_situation_batter_cluster_prev",
        "pitcher_situation_batter_cluster_prev_outcome",
    }:
        sort_cols.append("prev_pitch_type_id")
    if by in {
        "pitcher_count_prev_outcome",
        "pitcher_situation_prev_outcome",
        "pitcher_situation_batter_cluster_prev_outcome",
    }:
        sort_cols.append("prev_description_id")
    df = df.sort(sort_cols)
    return df


def default_profile_path(paths: Paths, *, run_id: str, split: str, by: ProfileBy) -> Path:
    run_dir = resolve_run_dir(paths.runs, run_id=run_id)
    name = f"pitcher_profiles_{by}_{split}.parquet"
    return run_dir / name


def write_profile(df: pl.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
