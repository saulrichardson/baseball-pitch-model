from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


def _as_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


def _as_int64(x: np.ndarray) -> np.ndarray:
    return x.astype(np.int64, copy=False)


@dataclass(frozen=True)
class Norm:
    mean: float
    std: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def _load_norms(meta: dict[str, Any]) -> dict[str, Norm]:
    norms_raw = meta.get("norms")
    if not isinstance(norms_raw, dict):
        raise ValueError("meta.json missing 'norms'")
    norms: dict[str, Norm] = {}
    for k, v in norms_raw.items():
        if not isinstance(v, dict) or "mean" not in v or "std" not in v:
            raise ValueError(f"Invalid norm entry for '{k}'")
        norms[k] = Norm(mean=float(v["mean"]), std=float(v["std"]))
    return norms


class PitchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: Path, meta: dict[str, Any]):
        self.path = path
        self.meta = meta

        cat_features = meta.get("cat_features")
        cont_features = meta.get("cont_features")
        history_len = int(meta.get("history_len"))
        if not isinstance(cat_features, list) or not isinstance(cont_features, list):
            raise ValueError("meta.json missing cat_features/cont_features")
        self.cat_features = [str(c) for c in cat_features]
        self.cont_features = [str(c) for c in cont_features]
        self.history_len = history_len
        self.norms = _load_norms(meta)

        df = pl.read_parquet(path)

        self.x_cat = _as_int64(df.select(self.cat_features).to_numpy())
        x_cont = _as_float32(df.select(self.cont_features).to_numpy())
        # Normalize cont features using train-derived norms (stored in meta.json).
        for idx, name in enumerate(self.cont_features):
            n = self.norms.get(name)
            if n is None:
                raise ValueError(f"Missing norm for cont feature: {name}")
            x_cont[:, idx] = _as_float32(n.apply(x_cont[:, idx]))
        self.x_cont = x_cont

        def _hist_to_2d(prefix: str, dtype: Any) -> np.ndarray:
            # New (scale) format: fixed columns prefix_{0..L-1}
            if f"{prefix}_0" in df.columns:
                cols = [f"{prefix}_{i}" for i in range(self.history_len)]
                return df.select(cols).to_numpy().astype(dtype, copy=False)

            # Legacy format: list column
            if prefix in df.columns:
                cols_expr = [pl.col(prefix).list.get(i).alias(f"{prefix}_{i}") for i in range(self.history_len)]
                return df.select(cols_expr).to_numpy().astype(dtype, copy=False)

            raise ValueError(f"Missing history columns for prefix '{prefix}'")

        self.hist_type = _hist_to_2d("hist_pitch_type_id", np.int64)
        self.hist_desc: np.ndarray | None = None
        if f"hist_description_id_0" in df.columns or "hist_description_id" in df.columns:
            self.hist_desc = _hist_to_2d("hist_description_id", np.int64)
        self.hist_x = _hist_to_2d("hist_plate_x", np.float32)
        self.hist_z = _hist_to_2d("hist_plate_z", np.float32)

        # Normalize history locations into the same coordinate system as targets.
        self.hist_x = _as_float32(self.norms["plate_x"].apply(self.hist_x))
        self.hist_z = _as_float32(self.norms["plate_z"].apply(self.hist_z))

        self.y_type = _as_int64(df.select("pitch_type_id").to_numpy().squeeze(1))
        self.y_desc: np.ndarray | None = None
        if "description_id" in df.columns:
            self.y_desc = _as_int64(df.select("description_id").to_numpy().squeeze(1))

        # Normalize location targets too (lets model operate in a stable coordinate system).
        plate_x = _as_float32(df.select("plate_x").to_numpy().squeeze(1))
        plate_z = _as_float32(df.select("plate_z").to_numpy().squeeze(1))
        plate_x = _as_float32(self.norms["plate_x"].apply(plate_x))
        plate_z = _as_float32(self.norms["plate_z"].apply(plate_z))
        self.y_loc = np.stack([plate_x, plate_z], axis=1)

        # Optional state-transition targets (schema_version >= 3).
        state_targets = meta.get("state_targets")
        self.state_targets: dict[str, np.ndarray] = {}
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
            for name in ordered:
                spec = state_targets.get(name)
                if not isinstance(spec, dict):
                    continue
                col = spec.get("col")
                if not isinstance(col, str):
                    continue
                if col not in df.columns:
                    continue
                self.state_targets[name] = _as_int64(df.select(col).to_numpy().squeeze(1))

        if not (
            len(self.x_cat)
            == len(self.x_cont)
            == len(self.hist_type)
            == (len(self.hist_desc) if self.hist_desc is not None else len(self.hist_type))
            == len(self.hist_x)
            == len(self.hist_z)
            == len(self.y_type)
            == (len(self.y_desc) if self.y_desc is not None else len(self.y_type))
            == len(self.y_loc)
        ):
            raise ValueError("Inconsistent dataset lengths")
        for name, arr in self.state_targets.items():
            if len(arr) != len(self.y_type):
                raise ValueError(f"Inconsistent dataset lengths for state target '{name}'")

    def __len__(self) -> int:
        return self.y_type.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "x_cat": torch.from_numpy(self.x_cat[idx]),
            "x_cont": torch.from_numpy(self.x_cont[idx]),
            "hist_type": torch.from_numpy(self.hist_type[idx]),
            **({"hist_desc": torch.from_numpy(self.hist_desc[idx])} if self.hist_desc is not None else {}),
            "hist_x": torch.from_numpy(self.hist_x[idx]),
            "hist_z": torch.from_numpy(self.hist_z[idx]),
            "y_type": torch.tensor(int(self.y_type[idx]), dtype=torch.long),
            **({"y_desc": torch.tensor(int(self.y_desc[idx]), dtype=torch.long)} if self.y_desc is not None else {}),
            "y_loc": torch.from_numpy(self.y_loc[idx]),
        }
        for name, arr in self.state_targets.items():
            out[name] = torch.tensor(int(arr[idx]), dtype=torch.long)
        return out
