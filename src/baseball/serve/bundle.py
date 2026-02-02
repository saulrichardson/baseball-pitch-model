from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from baseball.data.vocab import Vocab
from baseball.training.io import load_prepared
from baseball.training.models import (
    BaselineMLP,
    ModelConfig,
    TransformerMDN,
    TransformerMDNMT,
    TransformerMDNState,
    TransformerMDNStateMT,
    TransformerMDNV2,
)
from baseball.training.runs import latest_completed_run


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    model: torch.nn.Module
    meta: dict[str, Any]
    vocab_pitch_type: Vocab
    vocab_description: Vocab | None
    vocab_pitcher: Vocab
    vocab_batter: Vocab

    @property
    def pitch_type_id_to_token(self) -> list[str]:
        # 0 is OOV.
        size = self.vocab_pitch_type.size
        id_to_token = ["<OOV>"] * size
        for tok, idx in self.vocab_pitch_type.token_to_id.items():
            if 0 <= idx < size:
                id_to_token[idx] = tok
        return id_to_token

    @property
    def plate_x_mean(self) -> float:
        return float(self.meta["norms"]["plate_x"]["mean"])

    @property
    def plate_x_std(self) -> float:
        return float(self.meta["norms"]["plate_x"]["std"])

    @property
    def plate_z_mean(self) -> float:
        return float(self.meta["norms"]["plate_z"]["mean"])

    @property
    def plate_z_std(self) -> float:
        return float(self.meta["norms"]["plate_z"]["std"])


def load_latest_bundle(artifact_root: Path) -> ModelBundle:
    runs_root = artifact_root / "runs"
    run_dir = latest_completed_run(runs_root)
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
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
    model.eval()

    prepared = load_prepared(artifact_root / "data" / "prepared")
    meta = prepared.meta

    vocab_paths = meta["vocab_paths"]
    vocab_pitch_type = Vocab.load(prepared.prepared_dir / vocab_paths["pitch_type"])
    vocab_description = None
    if "description" in vocab_paths:
        vocab_description = Vocab.load(prepared.prepared_dir / vocab_paths["description"])
    if int(getattr(cfg, "n_descriptions", 0)) > 0 and vocab_description is None:
        raise ValueError(
            "Checkpoint ModelConfig.n_descriptions > 0 but prepared meta.json is missing vocab_paths['description']."
        )
    vocab_pitcher = Vocab.load(prepared.prepared_dir / vocab_paths["pitcher"])
    vocab_batter = Vocab.load(prepared.prepared_dir / vocab_paths["batter"])

    return ModelBundle(
        model_name=model_name,
        model=model,
        meta=meta,
        vocab_pitch_type=vocab_pitch_type,
        vocab_description=vocab_description,
        vocab_pitcher=vocab_pitcher,
        vocab_batter=vocab_batter,
    )
