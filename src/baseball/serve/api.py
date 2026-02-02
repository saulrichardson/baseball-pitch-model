from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from baseball.config import Paths
from baseball.serve.bundle import load_latest_bundle
from baseball.training.mdn import mdn_mean


class HistoryPitch(BaseModel):
    pitch_type: str = Field(..., description="Statcast pitch_type code (e.g., FF, SL, CH)")
    plate_x: float = Field(..., description="Feet; Statcast plate_x of the historical pitch")
    plate_z: float = Field(..., description="Feet; Statcast plate_z of the historical pitch")
    description: str | None = Field(
        default=None,
        description="Statcast pitch description for the historical pitch (e.g., ball, called_strike, foul).",
    )


class PredictRequest(BaseModel):
    # Raw IDs (as in Statcast)
    pitcher: int
    batter: int
    stand: Literal["R", "L", "S"] | str
    p_throws: Literal["R", "L"] | str

    # Pre-pitch context
    inning: int
    inning_topbot: Literal["Top", "Bot"] | str
    outs_when_up: int
    balls: int
    strikes: int
    pitch_number: int
    score_diff: int = Field(..., description="bat_score - fld_score (from batting team POV)")
    on_1b: bool = False
    on_2b: bool = False
    on_3b: bool = False

    history: list[HistoryPitch] = Field(default_factory=list, description="Previous pitches within this PA.")


class PredictResponse(BaseModel):
    model: str
    pitch_type_topk: list[dict[str, Any]]
    location_mean: dict[str, float]
    location_components: list[dict[str, float]] | None = None


def _encode_stand(stand: str) -> int:
    return {"R": 1, "L": 2, "S": 3}.get(stand, 0)


def _encode_p_throws(p: str) -> int:
    return {"R": 1, "L": 2}.get(p, 0)


def _encode_inning_topbot(v: str) -> int:
    return {"Bot": 1, "Top": 0}.get(v, 0)


def build_app(artifact_root: Path) -> FastAPI:
    bundle = load_latest_bundle(artifact_root)
    model = bundle.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    app = FastAPI(title="Baseball Pitch Predictor", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": bundle.model_name}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        hist = req.history[-int(bundle.meta["history_len"]) :]
        L = int(bundle.meta["history_len"])

        hist_type = [bundle.vocab_pitch_type.encode(h.pitch_type) for h in hist]
        hist_desc: list[int] | None = None
        if bundle.vocab_description is not None:
            hist_desc = []
            for h in hist:
                if h.description is None:
                    raise ValueError(
                        "This model expects `history[].description` (Statcast pitch description) "
                        "because it was trained with outcome-aware AB history."
                    )
                hist_desc.append(bundle.vocab_description.encode(str(h.description)))
        hist_x = [h.plate_x for h in hist]
        hist_z = [h.plate_z for h in hist]

        # Left-pad with zeros to fixed history length.
        pad = L - len(hist_type)
        if pad > 0:
            hist_type = ([0] * pad) + hist_type
            if hist_desc is not None:
                hist_desc = ([0] * pad) + hist_desc
            hist_x = ([0.0] * pad) + hist_x
            hist_z = ([0.0] * pad) + hist_z

        # Normalize plate coords with train-derived norms.
        hist_x = [(x - bundle.plate_x_mean) / bundle.plate_x_std for x in hist_x]
        hist_z = [(z - bundle.plate_z_mean) / bundle.plate_z_std for z in hist_z]

        pitcher_id = bundle.vocab_pitcher.encode(str(req.pitcher))
        batter_id = bundle.vocab_batter.encode(str(req.batter))
        stand_id = _encode_stand(str(req.stand))
        p_throws_id = _encode_p_throws(str(req.p_throws))

        # Continuous context features must match training meta cont_features order.
        cont_features: list[str] = list(bundle.meta["cont_features"])
        cont_map = {
            "inning": float(req.inning),
            "outs_when_up": float(req.outs_when_up),
            "balls": float(req.balls),
            "strikes": float(req.strikes),
            "pitch_number": float(req.pitch_number),
            "score_diff": float(req.score_diff),
            "on_1b_occ": float(1.0 if req.on_1b else 0.0),
            "on_2b_occ": float(1.0 if req.on_2b else 0.0),
            "on_3b_occ": float(1.0 if req.on_3b else 0.0),
            "inning_topbot_id": float(_encode_inning_topbot(str(req.inning_topbot))),
        }

        x_cont = []
        for name in cont_features:
            if name not in cont_map:
                raise ValueError(f"Missing cont feature for inference: {name}")
            n = bundle.meta["norms"][name]
            mean = float(n["mean"])
            std = float(n["std"])
            x_cont.append((cont_map[name] - mean) / std)

        batch = {
            "x_cat": torch.tensor([[pitcher_id, batter_id, stand_id, p_throws_id]], dtype=torch.long, device=device),
            "x_cont": torch.tensor([x_cont], dtype=torch.float32, device=device),
            "hist_type": torch.tensor([hist_type], dtype=torch.long, device=device),
            **(
                {"hist_desc": torch.tensor([hist_desc], dtype=torch.long, device=device)}
                if hist_desc is not None
                else {}
            ),
            "hist_x": torch.tensor([hist_x], dtype=torch.float32, device=device),
            "hist_z": torch.tensor([hist_z], dtype=torch.float32, device=device),
        }

        with torch.no_grad():
            out = model(batch)
            logits = out["type_logits"].squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=min(5, probs.numel()))
            id_to_tok = bundle.pitch_type_id_to_token
            pitch_type_topk = [
                {"pitch_type": id_to_tok[int(i)], "prob": float(p)} for p, i in zip(topk.values, topk.indices)
            ]

            if bundle.model_name in {
                "transformer_mdn",
                "transformer_mdn_mt",
                "transformer_mdn_v2",
                "transformer_mdn_state",
                "transformer_mdn_state_mt",
            }:
                mu_norm = mdn_mean(out["mdn_logit_pi"], out["mdn_mu"]).squeeze(0)
                mu_x = float(mu_norm[0] * bundle.plate_x_std + bundle.plate_x_mean)
                mu_z = float(mu_norm[1] * bundle.plate_z_std + bundle.plate_z_mean)

                pi = torch.softmax(out["mdn_logit_pi"].squeeze(0), dim=-1)
                mu = out["mdn_mu"].squeeze(0)
                sx = torch.exp(out["mdn_log_sx"].squeeze(0))
                sz = torch.exp(out["mdn_log_sz"].squeeze(0))
                rho = out["mdn_rho"].squeeze(0)

                comps = []
                for k in range(pi.numel()):
                    comps.append(
                        {
                            "weight": float(pi[k]),
                            "mu_x": float(mu[k, 0] * bundle.plate_x_std + bundle.plate_x_mean),
                            "mu_z": float(mu[k, 1] * bundle.plate_z_std + bundle.plate_z_mean),
                            "sx": float(sx[k] * bundle.plate_x_std),
                            "sz": float(sz[k] * bundle.plate_z_std),
                            "rho": float(rho[k]),
                        }
                    )
                location_components = comps
            else:
                mu_norm = out["loc_mu"].squeeze(0)
                mu_x = float(mu_norm[0] * bundle.plate_x_std + bundle.plate_x_mean)
                mu_z = float(mu_norm[1] * bundle.plate_z_std + bundle.plate_z_mean)
                location_components = None

        return PredictResponse(
            model=bundle.model_name,
            pitch_type_topk=pitch_type_topk,
            location_mean={"plate_x": mu_x, "plate_z": mu_z},
            location_components=location_components,
        )

    return app


def run_server(paths: Paths, host: str, port: int) -> None:
    import uvicorn

    app = build_app(paths.root)
    uvicorn.run(app, host=host, port=port)
