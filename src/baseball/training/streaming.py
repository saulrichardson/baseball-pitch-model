from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class Norm:
    mean: float
    std: float


def _load_norms(meta: dict[str, Any]) -> dict[str, Norm]:
    norms_raw = meta.get("norms")
    if not isinstance(norms_raw, dict):
        raise ValueError("meta.json missing 'norms'")
    out: dict[str, Norm] = {}
    for k, v in norms_raw.items():
        if not isinstance(v, dict) or "mean" not in v or "std" not in v:
            raise ValueError(f"Invalid norm entry for '{k}'")
        out[str(k)] = Norm(mean=float(v["mean"]), std=float(v["std"]))
    return out


class PitchParquetBatchIterable(IterableDataset[dict[str, torch.Tensor]]):
    """
    Stream prepared parquet into pre-batched tensors.

    Motivation: keep RAM usage low on GPU nodes by avoiding loading the full dataset.

    Requirements:
    - Prepared data can be either:
        - a single parquet file, or
        - a directory of parquet shards
    - Prepared parquet schema must have fixed history columns:
        hist_pitch_type_id_{0..L-1}, hist_plate_x_{0..L-1}, hist_plate_z_{0..L-1}
      (legacy list columns are supported but are slower)
    """

    def __init__(
        self,
        path: Path,
        meta: dict[str, Any],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        shuffle_rows_within_rowgroup: bool = True,
        return_raw_cont: bool = False,
    ) -> None:
        super().__init__()
        self.path = path
        if self.path.is_dir():
            self.files = sorted(self.path.glob("*.parquet"))
        else:
            self.files = [self.path]
        if not self.files:
            raise FileNotFoundError(f"No parquet files found at: {self.path}")

        self.meta = meta
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.shuffle_rows_within_rowgroup = bool(shuffle_rows_within_rowgroup)
        self.return_raw_cont = bool(return_raw_cont)

        cat_features = meta.get("cat_features")
        cont_features = meta.get("cont_features")
        history_len = int(meta.get("history_len"))
        if not isinstance(cat_features, list) or not isinstance(cont_features, list):
            raise ValueError("meta.json missing cat_features/cont_features")
        self.cat_features = [str(c) for c in cat_features]
        self.cont_features = [str(c) for c in cont_features]
        self.history_len = history_len
        self.norms = _load_norms(meta)

        # Sequence / history columns come from meta.json so the pipeline can evolve
        # without hardcoding column naming conventions here.
        seq = meta.get("seq_features")
        if isinstance(seq, dict):
            ht = seq.get("hist_pitch_type_id_cols")
            hx = seq.get("hist_plate_x_cols")
            hz = seq.get("hist_plate_z_cols")
            hd = seq.get("hist_description_id_cols")
            if isinstance(ht, list) and isinstance(hx, list) and isinstance(hz, list):
                self.hist_type_cols = [str(c) for c in ht]
                self.hist_x_cols = [str(c) for c in hx]
                self.hist_z_cols = [str(c) for c in hz]
            else:
                raise ValueError("meta.json seq_features missing required hist_*_cols lists")
            if isinstance(hd, list):
                self.hist_desc_cols = [str(c) for c in hd]
            else:
                self.hist_desc_cols = []
        else:
            # Legacy fallback.
            self.hist_type_cols = [f"hist_pitch_type_id_{i}" for i in range(self.history_len)]
            self.hist_x_cols = [f"hist_plate_x_{i}" for i in range(self.history_len)]
            self.hist_z_cols = [f"hist_plate_z_{i}" for i in range(self.history_len)]
            self.hist_desc_cols = []

        vocab_paths = meta.get("vocab_paths")
        self.has_description_target = isinstance(vocab_paths, dict) and "description" in vocab_paths

        # Optional state-transition targets (schema_version >= 3).
        # Keep an explicit stable order so downstream training/eval logic is deterministic.
        state_targets = meta.get("state_targets")
        if isinstance(state_targets, dict):
            ordered = [
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
            cols: list[str] = []
            for name in ordered:
                spec = state_targets.get(name)
                if isinstance(spec, dict) and isinstance(spec.get("col"), str):
                    cols.append(str(spec["col"]))
            self.state_target_cols = cols
        else:
            self.state_target_cols = []

        self._num_rows_cache: int | None = None

        # We open the parquet files lazily per-worker in __iter__.

    @property
    def num_rows(self) -> int:
        if self._num_rows_cache is None:
            total = 0
            for path in self.files:
                pf = pq.ParquetFile(path)
                total += int(pf.metadata.num_rows)
            self._num_rows_cache = total
        return int(self._num_rows_cache)

    @property
    def num_batches(self) -> int:
        return int(math.ceil(self.num_rows / self.batch_size))

    def _files_for_worker(self) -> list[Path]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)

        files = list(self.files)
        if self.shuffle:
            rng = random.Random(self.seed + worker_id)
            rng.shuffle(files)
        return files

    def _row_group_indices(self, pf: pq.ParquetFile) -> list[int]:
        n = pf.num_row_groups
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        rgs = [i for i in range(n) if (i % num_workers) == worker_id]
        if self.shuffle:
            rng = random.Random(self.seed + worker_id)
            rng.shuffle(rgs)
        return rgs

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        cols = [
            *self.cat_features,
            *self.cont_features,
            *self.hist_type_cols,
            *self.hist_desc_cols,
            *self.hist_x_cols,
            *self.hist_z_cols,
            *self.state_target_cols,
            *(["description_id"] if self.has_description_target else []),
            "pitch_type_id",
            "plate_x",
            "plate_z",
        ]

        for path in self._files_for_worker():
            pf = pq.ParquetFile(path)
            rg_indices = self._row_group_indices(pf)

            for rg in rg_indices:
                table = pf.read_row_group(rg, columns=cols).combine_chunks()
                n = table.num_rows
                if n == 0:
                    continue

                # Arrow -> numpy
                def col_np(name: str, dtype: Any) -> np.ndarray:
                    arr = table.column(name).to_numpy(zero_copy_only=False)
                    return arr.astype(dtype, copy=False)

                x_cat = np.stack([col_np(c, np.int64) for c in self.cat_features], axis=1)
                x_cont = np.stack([col_np(c, np.float32) for c in self.cont_features], axis=1)
                x_cont_raw = x_cont.astype(np.int64, copy=True) if self.return_raw_cont else None

                # Normalize cont features
                for j, name in enumerate(self.cont_features):
                    nm = self.norms.get(name)
                    if nm is None:
                        raise ValueError(f"Missing norm for cont feature: {name}")
                    x_cont[:, j] = (x_cont[:, j] - nm.mean) / nm.std

                hist_type = np.stack([col_np(c, np.int64) for c in self.hist_type_cols], axis=1)
                hist_desc = None
                if self.hist_desc_cols:
                    hist_desc = np.stack([col_np(c, np.int64) for c in self.hist_desc_cols], axis=1)
                hist_x = np.stack([col_np(c, np.float32) for c in self.hist_x_cols], axis=1)
                hist_z = np.stack([col_np(c, np.float32) for c in self.hist_z_cols], axis=1)

                # Normalize history locations into the same coordinate system as the targets.
                hist_x = (hist_x - self.norms["plate_x"].mean) / self.norms["plate_x"].std
                hist_z = (hist_z - self.norms["plate_z"].mean) / self.norms["plate_z"].std

                y_type = col_np("pitch_type_id", np.int64)
                y_desc = col_np("description_id", np.int64) if self.has_description_target else None
                plate_x = col_np("plate_x", np.float32)
                plate_z = col_np("plate_z", np.float32)

                plate_x = (plate_x - self.norms["plate_x"].mean) / self.norms["plate_x"].std
                plate_z = (plate_z - self.norms["plate_z"].mean) / self.norms["plate_z"].std
                y_loc = np.stack([plate_x, plate_z], axis=1).astype(np.float32, copy=False)

                if self.shuffle and self.shuffle_rows_within_rowgroup:
                    rng = np.random.default_rng(self.seed + rg + int(os.getpid()))
                    perm = rng.permutation(n)
                    x_cat = x_cat[perm]
                    x_cont = x_cont[perm]
                    if x_cont_raw is not None:
                        x_cont_raw = x_cont_raw[perm]
                    hist_type = hist_type[perm]
                    if hist_desc is not None:
                        hist_desc = hist_desc[perm]
                    hist_x = hist_x[perm]
                    hist_z = hist_z[perm]
                    y_type = y_type[perm]
                    if y_desc is not None:
                        y_desc = y_desc[perm]
                    y_loc = y_loc[perm]
                    state_targets = {c: col_np(c, np.int64)[perm] for c in self.state_target_cols}
                else:
                    state_targets = {c: col_np(c, np.int64) for c in self.state_target_cols}

                for start in range(0, n, self.batch_size):
                    end = min(n, start + self.batch_size)
                    batch = {
                        "x_cat": torch.from_numpy(x_cat[start:end]),
                        "x_cont": torch.from_numpy(x_cont[start:end]),
                        "hist_type": torch.from_numpy(hist_type[start:end]),
                        **(
                            {"hist_desc": torch.from_numpy(hist_desc[start:end])}
                            if hist_desc is not None
                            else {}
                        ),
                        "hist_x": torch.from_numpy(hist_x[start:end]),
                        "hist_z": torch.from_numpy(hist_z[start:end]),
                        "y_type": torch.from_numpy(y_type[start:end]),
                        **({"y_desc": torch.from_numpy(y_desc[start:end])} if y_desc is not None else {}),
                        "y_loc": torch.from_numpy(y_loc[start:end]),
                    }
                    if x_cont_raw is not None:
                        batch["x_cont_raw"] = torch.from_numpy(x_cont_raw[start:end])
                    for name in self.state_target_cols:
                        batch[name] = torch.from_numpy(state_targets[name][start:end])
                    yield batch
