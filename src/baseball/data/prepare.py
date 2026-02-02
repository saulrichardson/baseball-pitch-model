from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from baseball.data.vocab import Vocab, build_vocab_from_counts
from baseball.data.raw_schema import RAW_REQUIRED_COLUMNS


class PrepareError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedMeta:
    schema_version: int
    created_at: str
    start: str
    end: str
    history_len: int
    valid_frac: float
    valid_start: str
    split_strategy: str
    prepared_format: str
    train_data: str
    valid_data: str
    cat_features: list[str]
    cont_features: list[str]
    seq_features: dict[str, Any]
    state_targets: dict[str, Any]
    target_pitch_type: str
    target_plate_x: str
    target_plate_z: str
    vocab_paths: dict[str, str]
    vocab_min_counts: dict[str, int]
    norms: dict[str, dict[str, float]]


def _require_columns(df: pl.LazyFrame, cols: list[str]) -> None:
    present = set(df.collect_schema().names())
    missing = [c for c in cols if c not in present]
    if missing:
        raise PrepareError(f"Missing required columns from raw Statcast: {missing}")


def _parse_date_expr(col: str) -> pl.Expr:
    # pybaseball typically yields game_date as string YYYY-MM-DD; normalize to pl.Date
    return pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False)


def _choose_valid_start(counts: pl.DataFrame, valid_frac: float) -> date:
    """
    Choose a split boundary date based on per-day row counts.

    Split strategy:
    - train: game_date < valid_start
    - valid: game_date >= valid_start
    """

    if counts.is_empty():
        raise PrepareError("No rows available to compute a split boundary.")
    if "game_date" not in counts.columns or "count" not in counts.columns:
        raise PrepareError("Internal error: counts missing required columns ['game_date', 'count'].")

    counts = counts.sort("game_date")
    dates = counts.get_column("game_date").to_list()
    day_counts = counts.get_column("count").to_list()
    if len(dates) < 2:
        raise PrepareError("Not enough distinct dates to create a train/valid split.")

    total = float(sum(float(x) for x in day_counts))
    if total <= 0:
        raise PrepareError("Invalid total row count when computing split boundary.")

    # Consider cut positions BETWEEN dates (i = 1..n-1).
    # valid_start = dates[i]
    # train_count = sum(day_counts[:i])
    target_valid = valid_frac
    best_i = None
    best_err = float("inf")

    running = 0.0
    for i in range(1, len(dates)):
        running += float(day_counts[i - 1])
        valid_ratio = 1.0 - (running / total)
        err = abs(valid_ratio - target_valid)
        if err < best_err:
            best_err = err
            best_i = i

    if best_i is None:
        raise PrepareError("Failed to compute split boundary.")
    valid_start = dates[best_i]
    if not isinstance(valid_start, date):
        raise PrepareError("Internal error: game_date was not parsed as pl.Date.")
    return valid_start


def _prepare_dir(prepared_dir: Path, *, overwrite: bool) -> None:
    if prepared_dir.exists() and any(prepared_dir.iterdir()):
        if not overwrite:
            raise PrepareError(
                f"Prepared dir is not empty: {prepared_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)


def prepare_dataset(
    raw_dir: Path,
    prepared_dir: Path,
    start: str,
    end: str,
    history_len: int = 8,
    valid_frac: float = 0.1,
    min_pitch_type_count: int = 50,
    min_description_count: int = 25,
    min_pitcher_count: int = 50,
    min_batter_count: int = 50,
    *,
    overwrite: bool = False,
) -> None:
    if not raw_dir.exists():
        raise PrepareError(f"Raw data directory not found: {raw_dir}")
    _prepare_dir(prepared_dir, overwrite=overwrite)
    if history_len <= 0:
        raise ValueError("--history-len must be > 0")
    if not (0.0 < valid_frac < 0.5):
        raise ValueError("--valid-frac must be between 0 and 0.5")
    for name, v in {
        "min_pitch_type_count": min_pitch_type_count,
        "min_description_count": min_description_count,
        "min_pitcher_count": min_pitcher_count,
        "min_batter_count": min_batter_count,
    }.items():
        if int(v) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")

    try:
        start_d = datetime.strptime(start, "%Y-%m-%d").date()
        end_d = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError("start/end must be YYYY-MM-DD") from e

    # Raw parquet validation (robust to offseason / empty chunks).
    # Some date ranges may contain no MLB games, and some download failures can produce
    # zero-column parquets. We explicitly filter raw files to those that contain the
    # required columns, and write a report of any skipped chunks.
    required = list(RAW_REQUIRED_COLUMNS)
    raw_files_all = sorted(raw_dir.glob("statcast_*.parquet"))
    if not raw_files_all:
        raise PrepareError(f"No raw Statcast parquet files found under: {raw_dir}")

    valid_raw_files: list[Path] = []
    skipped: list[dict[str, Any]] = []
    for raw_path in raw_files_all:
        try:
            schema = pl.scan_parquet(str(raw_path)).collect_schema().names()
            schema_set = set(schema)
        except Exception as e:
            skipped.append({"path": str(raw_path), "error": repr(e)})
            continue
        missing = [c for c in required if c not in schema_set]
        if missing:
            skipped.append({"path": str(raw_path), "missing": missing, "present_cols": list(schema)[:25]})
            continue
        valid_raw_files.append(raw_path)

    if skipped:
        (prepared_dir / "skipped_raw_files.json").write_text(
            json.dumps(skipped, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not valid_raw_files:
        raise PrepareError("All raw parquet files were missing required columns; cannot prepare dataset.")

    lf = pl.scan_parquet([str(p) for p in valid_raw_files])
    _require_columns(lf, required)

    lf = lf.with_columns(game_date=_parse_date_expr("game_date")).filter(pl.col("game_date").is_not_null())
    lf = lf.filter(pl.col("game_date").is_between(start_d, end_d, closed="both"))

    # Drop rows without labels.
    lf = lf.filter(
        pl.col("pitch_type").is_not_null()
        & pl.col("plate_x").is_not_null()
        & pl.col("plate_z").is_not_null()
        & pl.col("pitch_number").is_not_null()
    )

    # Minimal cleanup / derived context features.
    lf = lf.with_columns(
        inning_topbot_id=pl.when(pl.col("inning_topbot") == "Bot")
        .then(pl.lit(1))
        .when(pl.col("inning_topbot") == "Top")
        .then(pl.lit(0))
        .otherwise(pl.lit(0))
        .cast(pl.Int8),
        stand_id=pl.when(pl.col("stand") == "R")
        .then(pl.lit(1))
        .when(pl.col("stand") == "L")
        .then(pl.lit(2))
        .when(pl.col("stand") == "S")
        .then(pl.lit(3))
        .otherwise(pl.lit(0))
        .cast(pl.Int8),
        p_throws_id=pl.when(pl.col("p_throws") == "R")
        .then(pl.lit(1))
        .when(pl.col("p_throws") == "L")
        .then(pl.lit(2))
        .otherwise(pl.lit(0))
        .cast(pl.Int8),
        on_1b_occ=pl.col("on_1b").is_not_null().cast(pl.Int8),
        on_2b_occ=pl.col("on_2b").is_not_null().cast(pl.Int8),
        on_3b_occ=pl.col("on_3b").is_not_null().cast(pl.Int8),
        score_diff=(pl.col("bat_score") - pl.col("fld_score")).cast(pl.Int16),
        pitcher_raw=pl.col("pitcher").cast(pl.Int64),
        batter_raw=pl.col("batter").cast(pl.Int64),
        pitch_type_raw=pl.col("pitch_type").cast(pl.Utf8),
        description_raw=pl.col("description").fill_null("<NULL>").cast(pl.Utf8),
        plate_x=pl.col("plate_x").cast(pl.Float32),
        plate_z=pl.col("plate_z").cast(pl.Float32),
    )

    # Compute split boundary from counts by date (cheap; avoids collecting entire dataset).
    counts = lf.select(pl.col("game_date")).group_by("game_date").len().rename({"len": "count"}).collect()
    valid_start_d = _choose_valid_start(counts, valid_frac=valid_frac)
    valid_start_s = valid_start_d.isoformat()

    # Build vocabs from TRAIN only (prevents validation-only tokens from inflating vocab).
    train_lf = lf.filter(pl.col("game_date") < valid_start_d)

    pitch_type_counts = (
        train_lf.select(token=pl.col("pitch_type_raw"))
        .group_by("token")
        .len()
        .rename({"len": "count"})
        .collect(streaming=True)
    )
    pitcher_counts = (
        train_lf.select(token=pl.col("pitcher_raw").cast(pl.Utf8))
        .group_by("token")
        .len()
        .rename({"len": "count"})
        .collect(streaming=True)
    )
    batter_counts = (
        train_lf.select(token=pl.col("batter_raw").cast(pl.Utf8))
        .group_by("token")
        .len()
        .rename({"len": "count"})
        .collect(streaming=True)
    )
    description_counts = (
        train_lf.select(token=pl.col("description_raw"))
        .group_by("token")
        .len()
        .rename({"len": "count"})
        .collect(streaming=True)
    )

    if (
        pitch_type_counts.is_empty()
        or pitcher_counts.is_empty()
        or batter_counts.is_empty()
        or description_counts.is_empty()
    ):
        raise PrepareError(
            "Train split produced no vocab counts. Adjust --valid-frac or date range."
        )

    pitch_type_vocab = build_vocab_from_counts(pitch_type_counts, min_count=min_pitch_type_count)
    description_vocab = build_vocab_from_counts(description_counts, min_count=min_description_count)
    pitcher_vocab = build_vocab_from_counts(pitcher_counts, min_count=min_pitcher_count)
    batter_vocab = build_vocab_from_counts(batter_counts, min_count=min_batter_count)

    vocab_dir = prepared_dir / "vocabs"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    pitch_type_vocab_path = vocab_dir / "pitch_type.json"
    description_vocab_path = vocab_dir / "description.json"
    pitcher_vocab_path = vocab_dir / "pitcher.json"
    batter_vocab_path = vocab_dir / "batter.json"
    pitch_type_vocab.save(pitch_type_vocab_path)
    description_vocab.save(description_vocab_path)
    pitcher_vocab.save(pitcher_vocab_path)
    batter_vocab.save(batter_vocab_path)

    # Encode vocabs (vectorized via joins; avoids Python callbacks per row).
    pitch_type_map = pl.DataFrame(
        {
            "pitch_type_raw": list(pitch_type_vocab.token_to_id.keys()),
            "pitch_type_id": list(pitch_type_vocab.token_to_id.values()),
        }
    )
    description_map = pl.DataFrame(
        {
            "description_raw": list(description_vocab.token_to_id.keys()),
            "description_id": list(description_vocab.token_to_id.values()),
        }
    )
    pitcher_map = pl.DataFrame(
        {
            "pitcher_token": list(pitcher_vocab.token_to_id.keys()),
            "pitcher_id": list(pitcher_vocab.token_to_id.values()),
        }
    )
    batter_map = pl.DataFrame(
        {
            "batter_token": list(batter_vocab.token_to_id.keys()),
            "batter_id": list(batter_vocab.token_to_id.values()),
        }
    )

    # Normalization stats for continuous features + targets (train only).
    cont_features = [
        "inning",
        "outs_when_up",
        "balls",
        "strikes",
        "pitch_number",
        "score_diff",
        "on_1b_occ",
        "on_2b_occ",
        "on_3b_occ",
        "inning_topbot_id",
    ]

    stats_exprs: list[pl.Expr] = []
    for col in [*cont_features, "plate_x", "plate_z"]:
        s = pl.col(col).cast(pl.Float64)
        stats_exprs.append(s.mean().alias(f"{col}__mean"))
        stats_exprs.append(s.std().alias(f"{col}__std"))

    stats_df = train_lf.select(stats_exprs).collect(streaming=True)
    if stats_df.is_empty():
        raise PrepareError("Train split is empty after applying date boundary. Adjust --valid-frac or date range.")
    stats = stats_df.to_dicts()[0]

    norms: dict[str, dict[str, float]] = {}
    for col in cont_features:
        mean = float(stats.get(f"{col}__mean", 0.0) or 0.0)
        std = float(stats.get(f"{col}__std", 0.0) or 0.0)
        norms[col] = {"mean": mean, "std": std if std > 0 else 1.0}

    for col in ["plate_x", "plate_z"]:
        mean = float(stats.get(f"{col}__mean", 0.0) or 0.0)
        std = float(stats.get(f"{col}__std", 0.0) or 0.0)
        norms[col] = {"mean": mean, "std": std if std > 0 else 1.0}

    # Select modeling columns only.
    cat_features = ["pitcher_id", "batter_id", "stand_id", "p_throws_id"]
    hist_type_cols = [f"hist_pitch_type_id_{i}" for i in range(history_len)]
    hist_desc_cols = [f"hist_description_id_{i}" for i in range(history_len)]
    hist_x_cols = [f"hist_plate_x_{i}" for i in range(history_len)]
    hist_z_cols = [f"hist_plate_z_{i}" for i in range(history_len)]

    # State-transition targets (for simulation without peeking at real next state).
    # These are derived from the *next pitch* within the same game (game_pk).
    # For the final pitch of a game (no next pitch), y_has_next=0 and other targets
    # are set to safe defaults (0). Training should mask those rows.
    state_target_cols = [
        "y_has_next",
        "y_pa_end",
        "y_next_balls",
        "y_next_strikes",
        "y_next_outs_when_up",
        "y_next_on_1b_occ",
        "y_next_on_2b_occ",
        "y_next_on_3b_occ",
        "y_next_inning_topbot_id",
        "y_inning_delta",
        "y_score_diff_delta_id",
    ]
    keep = [
        "split",
        "game_date",
        "game_pk",
        "at_bat_number",
        *cat_features,
        *cont_features,
        *hist_type_cols,
        *hist_desc_cols,
        *hist_x_cols,
        *hist_z_cols,
        *state_target_cols,
        "description_id",
        "pitch_type_id",
        "plate_x",
        "plate_z",
    ]
    # Avoid accidental duplicate projections (e.g., pitch_number appears in cont_features).
    seen: set[str] = set()
    keep_uniq: list[str] = []
    for c in keep:
        if c not in seen:
            keep_uniq.append(c)
            seen.add(c)

    # Sharded prepared data layout (scales; no single giant parquet required).
    train_dir = prepared_dir / "train"
    valid_dir = prepared_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    # Smaller row groups improve streaming throughput + RAM usage during training.
    row_group_size = 50_000

    # Process each raw parquet chunk independently to keep RAM bounded.
    raw_files = valid_raw_files

    train_rows = 0
    valid_rows = 0

    group_cols = ["game_pk", "at_bat_number"]
    hist_exprs: list[pl.Expr] = []
    # Index 0 = oldest (lag=history_len), index L-1 = most recent (lag=1)
    for idx, lag in enumerate(range(history_len, 0, -1)):
        hist_exprs.extend(
            [
                pl.col("pitch_type_id").shift(lag).over(group_cols).fill_null(0).alias(f"hist_pitch_type_id_{idx}"),
                pl.col("description_id")
                .shift(lag)
                .over(group_cols)
                .fill_null(0)
                .alias(f"hist_description_id_{idx}"),
                pl.col("plate_x").shift(lag).over(group_cols).fill_null(0.0).alias(f"hist_plate_x_{idx}"),
                pl.col("plate_z").shift(lag).over(group_cols).fill_null(0.0).alias(f"hist_plate_z_{idx}"),
            ]
        )

    for raw_path in raw_files:
        df = pl.read_parquet(raw_path, columns=required)
        if df.is_empty():
            continue

        df = (
            df.with_columns(game_date=_parse_date_expr("game_date"))
            .filter(pl.col("game_date").is_not_null())
            .filter(pl.col("game_date").is_between(start_d, end_d, closed="both"))
        )
        df = df.filter(
            pl.col("pitch_type").is_not_null()
            & pl.col("plate_x").is_not_null()
            & pl.col("plate_z").is_not_null()
            & pl.col("pitch_number").is_not_null()
        )
        if df.is_empty():
            continue

        df = df.with_columns(
            inning_topbot_id=pl.when(pl.col("inning_topbot") == "Bot")
            .then(pl.lit(1))
            .when(pl.col("inning_topbot") == "Top")
            .then(pl.lit(0))
            .otherwise(pl.lit(0))
            .cast(pl.Int8),
            stand_id=pl.when(pl.col("stand") == "R")
            .then(pl.lit(1))
            .when(pl.col("stand") == "L")
            .then(pl.lit(2))
            .when(pl.col("stand") == "S")
            .then(pl.lit(3))
            .otherwise(pl.lit(0))
            .cast(pl.Int8),
            p_throws_id=pl.when(pl.col("p_throws") == "R")
            .then(pl.lit(1))
            .when(pl.col("p_throws") == "L")
            .then(pl.lit(2))
            .otherwise(pl.lit(0))
            .cast(pl.Int8),
            on_1b_occ=pl.col("on_1b").is_not_null().cast(pl.Int8),
            on_2b_occ=pl.col("on_2b").is_not_null().cast(pl.Int8),
            on_3b_occ=pl.col("on_3b").is_not_null().cast(pl.Int8),
            score_diff=(pl.col("bat_score") - pl.col("fld_score")).cast(pl.Int16),
            pitcher_raw=pl.col("pitcher").cast(pl.Int64),
            batter_raw=pl.col("batter").cast(pl.Int64),
            pitch_type_raw=pl.col("pitch_type").cast(pl.Utf8),
            description_raw=pl.col("description").fill_null("<NULL>").cast(pl.Utf8),
            plate_x=pl.col("plate_x").cast(pl.Float32),
            plate_z=pl.col("plate_z").cast(pl.Float32),
            pitcher_token=pl.col("pitcher").cast(pl.Int64).cast(pl.Utf8),
            batter_token=pl.col("batter").cast(pl.Int64).cast(pl.Utf8),
        )

        df = (
            df.join(pitch_type_map, on="pitch_type_raw", how="left")
            .join(description_map, on="description_raw", how="left")
            .join(pitcher_map, on="pitcher_token", how="left")
            .join(batter_map, on="batter_token", how="left")
            .with_columns(
                pitch_type_id=pl.col("pitch_type_id").fill_null(0).cast(pl.Int32),
                description_id=pl.col("description_id").fill_null(0).cast(pl.Int32),
                pitcher_id=pl.col("pitcher_id").fill_null(0).cast(pl.Int32),
                batter_id=pl.col("batter_id").fill_null(0).cast(pl.Int32),
            )
            .drop(["pitcher_token", "batter_token"])
        )

        df = df.sort(["game_date", "game_pk", "at_bat_number", "pitch_number"]).with_columns(hist_exprs)

        # Next-pitch state targets: shift(-1) within each game_pk.
        game_cols = ["game_pk"]
        next_at_bat = pl.col("at_bat_number").shift(-1).over(game_cols)
        has_next = next_at_bat.is_not_null()

        next_balls = pl.col("balls").shift(-1).over(game_cols).fill_null(0).clip(0, 3).cast(pl.Int8)
        next_strikes = pl.col("strikes").shift(-1).over(game_cols).fill_null(0).clip(0, 2).cast(pl.Int8)
        next_outs = pl.col("outs_when_up").shift(-1).over(game_cols).fill_null(0).clip(0, 2).cast(pl.Int8)

        next_on_1b = pl.col("on_1b_occ").shift(-1).over(game_cols).fill_null(0).clip(0, 1).cast(pl.Int8)
        next_on_2b = pl.col("on_2b_occ").shift(-1).over(game_cols).fill_null(0).clip(0, 1).cast(pl.Int8)
        next_on_3b = pl.col("on_3b_occ").shift(-1).over(game_cols).fill_null(0).clip(0, 1).cast(pl.Int8)

        next_topbot = (
            pl.col("inning_topbot_id").shift(-1).over(game_cols).fill_null(pl.col("inning_topbot_id")).clip(0, 1).cast(pl.Int8)
        )

        next_inning = pl.col("inning").shift(-1).over(game_cols).fill_null(pl.col("inning")).cast(pl.Int16)
        inning_delta = (next_inning - pl.col("inning").cast(pl.Int16)).clip(0, 1).cast(pl.Int8)

        next_score = pl.col("score_diff").shift(-1).over(game_cols).fill_null(pl.col("score_diff")).cast(pl.Int16)
        score_delta = (next_score - pl.col("score_diff").cast(pl.Int16)).clip(-4, 4).cast(pl.Int8)
        score_delta_id = (score_delta + 4).cast(pl.Int8)  # 0..8 for delta=-4..+4

        pa_end = (
            pl.when(has_next)
            .then(next_at_bat != pl.col("at_bat_number"))
            .otherwise(pl.lit(True))
            .cast(pl.Int8)
        )

        df = df.with_columns(
            y_has_next=has_next.cast(pl.Int8),
            y_pa_end=pa_end,
            y_next_balls=next_balls,
            y_next_strikes=next_strikes,
            y_next_outs_when_up=next_outs,
            y_next_on_1b_occ=next_on_1b,
            y_next_on_2b_occ=next_on_2b,
            y_next_on_3b_occ=next_on_3b,
            y_next_inning_topbot_id=next_topbot,
            y_inning_delta=inning_delta,
            y_score_diff_delta_id=score_delta_id,
        )

        df = df.with_columns(
            split=pl.when(pl.col("game_date") < valid_start_d).then(pl.lit("train")).otherwise(pl.lit("valid"))
        )
        df = df.select(keep_uniq)

        out_name = raw_path.name.replace("statcast_", "prepared_")

        df_train = df.filter(pl.col("split") == "train")
        if not df_train.is_empty():
            df_train.write_parquet(
                train_dir / out_name, compression="zstd", row_group_size=row_group_size
            )
            train_rows += df_train.height

        df_valid = df.filter(pl.col("split") == "valid")
        if not df_valid.is_empty():
            df_valid.write_parquet(
                valid_dir / out_name, compression="zstd", row_group_size=row_group_size
            )
            valid_rows += df_valid.height

    if train_rows == 0 or valid_rows == 0:
        raise PrepareError(
            f"Prepared split produced empty data (train_rows={train_rows}, valid_rows={valid_rows}). "
            "Adjust --valid-frac or date range."
        )

    state_targets = {
        "y_has_next": {"col": "y_has_next", "type": "mask", "desc": "1 if next pitch exists within same game_pk."},
        "y_pa_end": {"col": "y_pa_end", "type": "bool", "desc": "1 if this pitch ends the plate appearance (or game)."},
        "y_next_balls": {"col": "y_next_balls", "type": "categorical", "classes": [0, 1, 2, 3]},
        "y_next_strikes": {"col": "y_next_strikes", "type": "categorical", "classes": [0, 1, 2]},
        "y_next_outs_when_up": {"col": "y_next_outs_when_up", "type": "categorical", "classes": [0, 1, 2]},
        "y_next_on_1b_occ": {"col": "y_next_on_1b_occ", "type": "categorical", "classes": [0, 1]},
        "y_next_on_2b_occ": {"col": "y_next_on_2b_occ", "type": "categorical", "classes": [0, 1]},
        "y_next_on_3b_occ": {"col": "y_next_on_3b_occ", "type": "categorical", "classes": [0, 1]},
        "y_next_inning_topbot_id": {
            "col": "y_next_inning_topbot_id",
            "type": "categorical",
            "classes": [0, 1],
            "mapping": {"Top": 0, "Bot": 1},
        },
        "y_inning_delta": {"col": "y_inning_delta", "type": "categorical", "classes": [0, 1]},
        "y_score_diff_delta_id": {
            "col": "y_score_diff_delta_id",
            "type": "categorical",
            "classes": list(range(9)),
            "desc": "score_diff_next - score_diff clipped to [-4,+4], then shifted by +4 to make 0..8.",
            "delta_min": -4,
            "delta_max": 4,
            "delta_offset": 4,
        },
    }

    meta = PreparedMeta(
        schema_version=4,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        start=start,
        end=end,
        history_len=history_len,
        valid_frac=valid_frac,
        valid_start=valid_start_s,
        split_strategy="date_cut",
        prepared_format="sharded_parquet_v1",
        train_data="train",
        valid_data="valid",
        cat_features=cat_features,
        cont_features=cont_features,
        seq_features={
            "hist_pitch_type_id_cols": hist_type_cols,
            "hist_description_id_cols": hist_desc_cols,
            "hist_plate_x_cols": hist_x_cols,
            "hist_plate_z_cols": hist_z_cols,
        },
        state_targets=state_targets,
        target_pitch_type="pitch_type_id",
        target_plate_x="plate_x",
        target_plate_z="plate_z",
        vocab_paths={
            "pitch_type": str(pitch_type_vocab_path.relative_to(prepared_dir)),
            "description": str(description_vocab_path.relative_to(prepared_dir)),
            "pitcher": str(pitcher_vocab_path.relative_to(prepared_dir)),
            "batter": str(batter_vocab_path.relative_to(prepared_dir)),
        },
        vocab_min_counts={
            "pitch_type": int(min_pitch_type_count),
            "description": int(min_description_count),
            "pitcher": int(min_pitcher_count),
            "batter": int(min_batter_count),
        },
        norms=norms,
    )
    with (prepared_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2, sort_keys=True)
