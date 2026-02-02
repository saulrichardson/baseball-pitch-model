from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import torch
import torch.nn.functional as F

from baseball.config import Paths
from baseball.data.vocab import Vocab
from baseball.training.io import load_prepared
from baseball.training.mdn import mdn_mean, mdn_nll
from baseball.training.models import ModelConfig, TransformerMDNState, TransformerMDNStateMT
from baseball.training.runs import resolve_run_dir


class SimulationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Norm:
    mean: float
    std: float

    def norm(self, x: float) -> float:
        return (x - self.mean) / self.std

    def denorm(self, x: float) -> float:
        return x * self.std + self.mean


def _load_norms(meta: dict[str, Any]) -> dict[str, Norm]:
    norms_raw = meta.get("norms")
    if not isinstance(norms_raw, dict):
        raise SimulationError("meta.json missing 'norms'")
    out: dict[str, Norm] = {}
    for k, v in norms_raw.items():
        if not isinstance(v, dict) or "mean" not in v or "std" not in v:
            raise SimulationError(f"Invalid norm entry for '{k}'")
        mean = float(v["mean"])
        std = float(v["std"])
        out[str(k)] = Norm(mean=mean, std=std if std > 0 else 1.0)
    return out


def _scan_split(prepared_split_path: Path) -> pl.LazyFrame:
    if prepared_split_path.is_dir():
        return pl.scan_parquet(str(prepared_split_path / "*.parquet"))
    return pl.scan_parquet(str(prepared_split_path))


def _list_game_pks(prepared_split_path: Path, *, max_games: int) -> list[int]:
    lf = _scan_split(prepared_split_path).select(pl.col("game_pk"))
    # Use streaming collection to avoid materializing all columns.
    df = lf.unique().collect(streaming=True)
    if "game_pk" not in df.columns:
        raise SimulationError("Prepared data missing 'game_pk'")
    pks = [int(x) for x in df.get_column("game_pk").to_list()]
    pks = sorted(set(pks))
    return pks[: max(0, int(max_games))]


def _load_game(prepared_split_path: Path, *, game_pk: int) -> pl.DataFrame:
    lf = _scan_split(prepared_split_path)
    cols_present = set(lf.collect_schema().names())
    desc_col = "description_id" if "description_id" in cols_present else None
    df = (
        lf.filter(pl.col("game_pk") == int(game_pk))
        .select(
            [
                "game_pk",
                "at_bat_number",
                "pitch_number",
                *( [desc_col] if desc_col is not None else [] ),
                "pitch_type_id",
                "plate_x",
                "plate_z",
                "pitcher_id",
                "batter_id",
                "stand_id",
                "p_throws_id",
                # Pre-pitch state features (raw, not normalized)
                "inning",
                "outs_when_up",
                "balls",
                "strikes",
                "score_diff",
                "on_1b_occ",
                "on_2b_occ",
                "on_3b_occ",
                "inning_topbot_id",
                # Next-state labels for validation
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
        )
        .collect(streaming=True)
    )
    if df.is_empty():
        raise SimulationError(f"No rows found for game_pk={game_pk}")
    return df.sort(["at_bat_number", "pitch_number"])


def _torch_topk_acc(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    k = min(int(k), int(logits.size(-1)))
    if k <= 0:
        return 0.0
    topk = torch.topk(logits, k=k, dim=-1).indices
    return float((topk == y.unsqueeze(-1)).any(dim=-1).float().mean().item())


def _id_to_token(vocab: Vocab) -> list[str]:
    # 0 is OOV.
    size = vocab.size
    out = ["<OOV>"] * size
    for tok, idx in vocab.token_to_id.items():
        if 0 <= idx < size:
            out[idx] = str(tok)
    return out


def _apply_desc_count_rules(desc: str, *, balls: int, strikes: int) -> tuple[int, int, int, bool]:
    """
    Deterministic within-PA count transitions based on Statcast `description`.

    Returns:
      (next_balls, next_strikes, pa_end, used_rules)

    Notes:
    - We only model balls/strikes + PA ending here. Other state (outs/bases/score)
      remains predicted by the model heads.
    - If the PA ends via walk/strikeout/HBP/ball in play, next pitch in the game
      is from the next PA, so balls/strikes reset to 0.
    """

    b = int(balls)
    s = int(strikes)
    d = str(desc)

    # Ball-like outcomes.
    if d in {"ball", "blocked_ball", "pitchout"}:
        b2 = b + 1
        if b2 >= 4:
            return (0, 0, 1, True)  # walk -> next PA
        return (min(3, b2), s, 0, True)

    # Immediate PA ending outcomes.
    if d in {"hit_by_pitch", "hit_into_play"}:
        return (0, 0, 1, True)

    # Strike-like outcomes.
    if d in {"called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}:
        s2 = s + 1
        if s2 >= 3:
            return (0, 0, 1, True)  # strikeout -> next PA
        return (b, min(2, s2), 0, True)

    # Fouls.
    if d == "foul":
        # With 2 strikes, fouls do not increment strikes.
        if s < 2:
            return (b, s + 1, 0, True)
        return (b, s, 0, True)

    if d == "foul_bunt":
        # Fouled bunt with 2 strikes is a strikeout.
        if s >= 2:
            return (0, 0, 1, True)
        return (b, s + 1, 0, True)

    return (b, s, 0, False)


def _clamped_count_decode(
    *,
    next_balls_logits: torch.Tensor,
    next_strikes_logits: torch.Tensor,
    balls: int,
    strikes: int,
) -> tuple[int, int]:
    """
    Jointly decode (next_balls, next_strikes) under simple within-PA constraints:
    - counts never decrease
    - at most one of balls/strikes increments per pitch
    - increments are at most +1

    This does not require a description head.
    """

    if next_balls_logits.ndim != 1 or int(next_balls_logits.numel()) != 4:
        raise SimulationError("next_balls_logits must be rank-1 [4]")
    if next_strikes_logits.ndim != 1 or int(next_strikes_logits.numel()) != 3:
        raise SimulationError("next_strikes_logits must be rank-1 [3]")

    b0 = int(max(0, min(3, int(balls))))
    s0 = int(max(0, min(2, int(strikes))))
    b1 = int(min(3, b0 + 1))
    s1 = int(min(2, s0 + 1))

    b_logp = F.log_softmax(next_balls_logits, dim=-1)
    s_logp = F.log_softmax(next_strikes_logits, dim=-1)

    # Candidate transitions: stay, ball++, strike++
    cand = [
        (b0, s0, float(b_logp[b0] + s_logp[s0])),
        (b1, s0, float(b_logp[b1] + s_logp[s0])),
        (b0, s1, float(b_logp[b0] + s_logp[s1])),
    ]
    cand.sort(key=lambda x: x[2], reverse=True)
    return (int(cand[0][0]), int(cand[0][1]))


def _constrained_count_decode(
    *,
    desc_logits: torch.Tensor,
    pa_end_logits: torch.Tensor,
    next_balls_logits: torch.Tensor,
    next_strikes_logits: torch.Tensor,
    desc_id_to_tok: list[str],
    balls: int,
    strikes: int,
    alpha_desc: float = 0.35,
    alpha_pa_end: float = 1.0,
    alpha_balls: float = 1.0,
    alpha_strikes: float = 1.0,
) -> tuple[int, int, int, int, bool]:
    """
    Pick a stable count transition in open-loop rollouts.

    Returns:
      next_balls, next_strikes, pa_end, chosen_desc_id, used_rules

    Strategy:
    - Candidate A ("heads"): use the argmax of the model's next_* heads.
    - Candidate B ("rules"): for each description token that implies a deterministic
      (next_balls, next_strikes, pa_end), score that transition under:
        log p(desc) + log p(pa_end) + log p(next_balls) + log p(next_strikes)
      and take the best.
    - Choose whichever candidate has higher score.

    This is a constrained decode, not a true joint model; it reduces obviously
    inconsistent transitions without forcing rules when the description head is weak.
    """

    if desc_logits.ndim != 1:
        raise SimulationError("desc_logits must be rank-1 [D]")
    if pa_end_logits.ndim != 1 or int(pa_end_logits.numel()) != 2:
        raise SimulationError("pa_end_logits must be rank-1 [2]")
    if next_balls_logits.ndim != 1 or int(next_balls_logits.numel()) != 4:
        raise SimulationError("next_balls_logits must be rank-1 [4]")
    if next_strikes_logits.ndim != 1 or int(next_strikes_logits.numel()) != 3:
        raise SimulationError("next_strikes_logits must be rank-1 [3]")

    dlogp = F.log_softmax(desc_logits, dim=-1)
    pa_logp = F.log_softmax(pa_end_logits, dim=-1)
    b_logp = F.log_softmax(next_balls_logits, dim=-1)
    s_logp = F.log_softmax(next_strikes_logits, dim=-1)

    # Candidate from heads.
    heads_pa = int(pa_end_logits.argmax(dim=-1).item())
    heads_b = int(next_balls_logits.argmax(dim=-1).item())
    heads_s = int(next_strikes_logits.argmax(dim=-1).item())
    heads_score = float(
        alpha_pa_end * pa_logp[heads_pa] + alpha_balls * b_logp[heads_b] + alpha_strikes * s_logp[heads_s]
    )

    best_rule_score = float("-inf")
    best_rule: tuple[int, int, int, int] | None = None  # (b, s, pa, desc_id)

    D = int(desc_logits.numel())
    for desc_id in range(D):
        if not (0 <= desc_id < len(desc_id_to_tok)):
            continue
        tok = str(desc_id_to_tok[desc_id])
        nb, ns, pe, used = _apply_desc_count_rules(tok, balls=balls, strikes=strikes)
        if not used:
            continue
        nb = int(max(0, min(3, nb)))
        ns = int(max(0, min(2, ns)))
        pe = int(1 if int(pe) == 1 else 0)
        score = float(
            alpha_desc * dlogp[desc_id]
            + alpha_pa_end * pa_logp[pe]
            + alpha_balls * b_logp[nb]
            + alpha_strikes * s_logp[ns]
        )
        if score > best_rule_score:
            best_rule_score = score
            best_rule = (nb, ns, pe, desc_id)

    if best_rule is not None and best_rule_score > heads_score:
        nb, ns, pe, desc_id = best_rule
        return (int(nb), int(ns), int(pe), int(desc_id), True)

    # Default to heads; still return argmax description for history tokens.
    return (
        int(heads_b),
        int(heads_s),
        int(heads_pa),
        int(desc_logits.argmax(dim=-1).item()),
        False,
    )


def simulate(
    paths: Paths,
    *,
    run_id: str,
    split: str,
    mode: str,
    count_mode: str = "heads",
    game_pks: list[int] | None,
    max_games: int,
    device: str | None,
    events_out: str | None = None,
    events_topk: int = 5,
    events_max: int = 0,
) -> dict[str, Any]:
    """
    Simulate / replay held-out games.

    Modes:
      - replay: teacher-forced. Uses the real pre-pitch state from the dataset.
      - rollout: open-loop. Feeds the model its own predicted next state (while still
                taking the real pitcher/batter IDs per pitch, since we do not model
                lineup/substitution decisions yet).
    """

    if split not in {"train", "valid"}:
        raise ValueError("--split must be one of: train, valid")
    if mode not in {"replay", "rollout"}:
        raise ValueError("--mode must be one of: replay, rollout")
    if count_mode not in {"heads", "clamp", "rules", "constrained"}:
        raise ValueError("--count-mode must be one of: heads, clamp, rules, constrained")

    prepared = load_prepared(paths.data_prepared)
    meta = prepared.meta
    norms = _load_norms(meta)

    events_path: Path | None = None
    events_fp = None
    events_written = 0

    if events_out is not None and str(events_out).strip():
        if str(events_out).strip() == "-":
            raise SimulationError("--events-out must be a file path (not '-') to avoid mixing with summary JSON.")
        events_path = Path(str(events_out))
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_fp = events_path.open("w", encoding="utf-8")

    pitch_id_to_tok: list[str] | None = None
    if events_fp is not None:
        pitch_vocab_path = prepared.vocabs_dir / "pitch_type.json"
        if not pitch_vocab_path.exists():
            raise SimulationError(f"--events-out requires pitch type vocab at: {pitch_vocab_path}")
        pitch_vocab = Vocab.load(pitch_vocab_path)
        pitch_id_to_tok = _id_to_token(pitch_vocab)

    desc_id_to_tok: list[str] | None = None
    if "vocab_paths" in meta and isinstance(meta["vocab_paths"], dict) and "description" in meta["vocab_paths"]:
        vocab_paths = meta["vocab_paths"]
        rel = vocab_paths.get("description")
        if isinstance(rel, str) and rel:
            desc_vocab = Vocab.load(prepared.prepared_dir / rel)
            desc_id_to_tok = _id_to_token(desc_vocab)

    run_dir = resolve_run_dir(paths.runs, run_id=run_id)
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise SimulationError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_name = str(ckpt.get("model_name"))
    if model_name not in {"transformer_mdn_state", "transformer_mdn_state_mt"}:
        raise SimulationError(
            "simulate requires a state-capable model "
            "(transformer_mdn_state or transformer_mdn_state_mt). "
            f"Got: {model_name}"
        )

    if mode == "rollout" and count_mode in {"rules", "constrained"}:
        if model_name != "transformer_mdn_state_mt":
            raise SimulationError(
                f"--count-mode={count_mode} requires transformer_mdn_state_mt (outcome-aware state model). "
                f"Got: {model_name}"
            )
        if desc_id_to_tok is None:
            raise SimulationError(
                f"--count-mode={count_mode} requires a description vocab in prepared data. "
                "Re-run `python -m baseball prepare` with schema_version >= 4."
            )

    cfg = ModelConfig(**ckpt["model_config"])
    if model_name == "transformer_mdn_state":
        model: torch.nn.Module = TransformerMDNState(cfg)
    else:
        model = TransformerMDNStateMT(cfg)
    model.load_state_dict(ckpt["state_dict"])

    if device is None or device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    model.to(dev)
    model.eval()

    split_path = prepared.valid_path if split == "valid" else prepared.train_path

    if game_pks is None or len(game_pks) == 0:
        game_pks = _list_game_pks(split_path, max_games=max_games)
    else:
        game_pks = [int(x) for x in game_pks]

    if not game_pks:
        raise SimulationError("No game_pks selected for simulation.")

    # Aggregate metrics.
    total_pitches = 0
    sum_type_acc = 0.0
    sum_type_top3 = 0.0
    sum_loc_mse_ft = 0.0
    sum_desc_acc = 0.0
    sum_desc_top3 = 0.0
    desc_total = 0

    # State (next-step) metrics only defined where y_has_next=1.
    state_total = 0
    state_correct: dict[str, int] = {}

    # In rollout mode, track how often the *current* simulated state matches the real state.
    curr_total = 0
    curr_match: dict[str, int] = {}
    rollout_count: dict[str, Any] = {
        "count_mode": str(count_mode),
        "rules_applied_n": 0,
        "constrained_rules_chosen_n": 0,
        "constrained_heads_chosen_n": 0,
        "constrained_desc_override_n": 0,
    }

    def _tok(id_to_tok: list[str] | None, idx: int) -> str:
        if id_to_tok is None:
            return str(idx)
        if 0 <= int(idx) < len(id_to_tok):
            return str(id_to_tok[int(idx)])
        return "<OOV>"

    try:
        for game_pk in game_pks:
            df = _load_game(split_path, game_pk=game_pk)

            # Simulated state (raw units).
            sim = {
                "inning": int(df[0, "inning"]),
                "outs_when_up": int(df[0, "outs_when_up"]),
                "balls": int(df[0, "balls"]),
                "strikes": int(df[0, "strikes"]),
                "score_diff": int(df[0, "score_diff"]),
                "on_1b_occ": int(df[0, "on_1b_occ"]),
                "on_2b_occ": int(df[0, "on_2b_occ"]),
                "on_3b_occ": int(df[0, "on_3b_occ"]),
                "inning_topbot_id": int(df[0, "inning_topbot_id"]),
                "pitch_number": int(df[0, "pitch_number"]),
            }

            # Rolling history in normalized coords (most recent last).
            hist_type: list[int] = []
            hist_desc: list[int] = []
            hist_x: list[float] = []
            hist_z: list[float] = []

            for row in df.iter_rows(named=True):
                if mode == "rollout":
                    curr_total += 1

                    def _curr(name: str, actual_key: str) -> None:
                        if int(sim[name]) == int(row[actual_key]):
                            curr_match[name] = curr_match.get(name, 0) + 1

                    _curr("inning", "inning")
                    _curr("outs_when_up", "outs_when_up")
                    _curr("balls", "balls")
                    _curr("strikes", "strikes")
                    _curr("pitch_number", "pitch_number")
                    _curr("score_diff", "score_diff")
                    _curr("on_1b_occ", "on_1b_occ")
                    _curr("on_2b_occ", "on_2b_occ")
                    _curr("on_3b_occ", "on_3b_occ")
                    _curr("inning_topbot_id", "inning_topbot_id")

                # Current pitch context is either teacher-forced (replay) or simulated (rollout).
                if mode == "replay":
                    sim["inning"] = int(row["inning"])
                    sim["outs_when_up"] = int(row["outs_when_up"])
                    sim["balls"] = int(row["balls"])
                    sim["strikes"] = int(row["strikes"])
                    sim["score_diff"] = int(row["score_diff"])
                    sim["on_1b_occ"] = int(row["on_1b_occ"])
                    sim["on_2b_occ"] = int(row["on_2b_occ"])
                    sim["on_3b_occ"] = int(row["on_3b_occ"])
                    sim["inning_topbot_id"] = int(row["inning_topbot_id"])
                    sim["pitch_number"] = int(row["pitch_number"])

                    # Use the *real* within-PA history by reconstructing from the sequence so far.
                    # (We intentionally do not read prepared hist_* columns here.)
                    if int(row["pitch_number"]) == 1:
                        hist_type = []
                        hist_desc = []
                        hist_x = []
                        hist_z = []

                # Build x_cat from prepared ids (we are not simulating lineup decisions yet).
                x_cat = torch.tensor(
                    [[int(row["pitcher_id"]), int(row["batter_id"]), int(row["stand_id"]), int(row["p_throws_id"])]],
                    dtype=torch.long,
                    device=dev,
                )

                cont_features: list[str] = list(meta["cont_features"])
                cont_map = {
                    "inning": float(sim["inning"]),
                    "outs_when_up": float(sim["outs_when_up"]),
                    "balls": float(sim["balls"]),
                    "strikes": float(sim["strikes"]),
                    "pitch_number": float(sim["pitch_number"]),
                    "score_diff": float(sim["score_diff"]),
                    "on_1b_occ": float(sim["on_1b_occ"]),
                    "on_2b_occ": float(sim["on_2b_occ"]),
                    "on_3b_occ": float(sim["on_3b_occ"]),
                    "inning_topbot_id": float(sim["inning_topbot_id"]),
                }
                x_cont = []
                for name in cont_features:
                    if name not in cont_map:
                        raise SimulationError(f"Missing cont feature in simulate(): {name}")
                    n = norms[name]
                    x_cont.append(n.norm(cont_map[name]))
                x_cont_t = torch.tensor([x_cont], dtype=torch.float32, device=dev)

                # Left-pad history to fixed length.
                L = int(meta["history_len"])
                ht = hist_type[-L:]
                hd = hist_desc[-L:]
                hx = hist_x[-L:]
                hz = hist_z[-L:]
                pad = L - len(ht)
                if pad > 0:
                    ht = ([0] * pad) + ht
                    hd = ([0] * pad) + hd
                    hx = ([0.0] * pad) + hx
                    hz = ([0.0] * pad) + hz

                batch = {
                    "x_cat": x_cat,
                    "x_cont": x_cont_t,
                    "hist_type": torch.tensor([ht], dtype=torch.long, device=dev),
                    **(
                        {"hist_desc": torch.tensor([hd], dtype=torch.long, device=dev)}
                        if int(getattr(cfg, "n_descriptions", 0)) > 0
                        else {}
                    ),
                    "hist_x": torch.tensor([hx], dtype=torch.float32, device=dev),
                    "hist_z": torch.tensor([hz], dtype=torch.float32, device=dev),
                }

                with torch.no_grad():
                    out = model(batch)
                    type_logits = out["type_logits"].squeeze(0)
                    y_type = torch.tensor(int(row["pitch_type_id"]), dtype=torch.long, device=dev)
                    pred_type = int(type_logits.argmax(dim=-1).item())

                    type_acc = float((pred_type == int(row["pitch_type_id"])))
                    type_top3 = _torch_topk_acc(type_logits.unsqueeze(0), y_type.unsqueeze(0), k=3)

                    desc_logits = out.get("desc_logits")
                    pred_desc_argmax = 0
                    pred_desc = 0
                    if desc_logits is not None:
                        desc_logits = desc_logits.squeeze(0)
                        pred_desc_argmax = int(desc_logits.argmax(dim=-1).item())
                        pred_desc = pred_desc_argmax
                        if "description_id" in row:
                            y_desc = torch.tensor(int(row.get("description_id", 0) or 0), dtype=torch.long, device=dev)
                            sum_desc_acc += float(pred_desc_argmax == int(y_desc.item()))
                            sum_desc_top3 += _torch_topk_acc(desc_logits.unsqueeze(0), y_desc.unsqueeze(0), k=3)
                            desc_total += 1

                    loc_norm = mdn_mean(out["mdn_logit_pi"], out["mdn_mu"]).squeeze(0)
                    pred_x_ft = norms["plate_x"].denorm(float(loc_norm[0].item()))
                    pred_z_ft = norms["plate_z"].denorm(float(loc_norm[1].item()))
                    dx = pred_x_ft - float(row["plate_x"])
                    dz = pred_z_ft - float(row["plate_z"])
                    loc_mse = 0.5 * (dx * dx + dz * dz)

                    # Per-pitch location NLL (normalized units) for debugging.
                    y_loc_norm = torch.tensor(
                        [[norms["plate_x"].norm(float(row["plate_x"])), norms["plate_z"].norm(float(row["plate_z"]))]],
                        dtype=torch.float32,
                        device=dev,
                    )
                    loc_nll = float(
                        mdn_nll(
                            y_loc_norm,
                            out["mdn_logit_pi"],
                            out["mdn_mu"],
                            out["mdn_log_sx"],
                            out["mdn_log_sz"],
                            out["mdn_rho"],
                        ).item()
                    )

                    # Next-state predictions (argmax per head).
                    pa_end_logits = out["pa_end_logits"].squeeze(0)
                    next_balls_logits = out["next_balls_logits"].squeeze(0)
                    next_strikes_logits = out["next_strikes_logits"].squeeze(0)
                    pa_end = int(pa_end_logits.argmax(dim=-1).item())
                    next_balls = int(next_balls_logits.argmax(dim=-1).item())
                    next_strikes = int(next_strikes_logits.argmax(dim=-1).item())
                    next_outs = int(out["next_outs_when_up_logits"].argmax(dim=-1).item())
                    next_on_1b = int(out["next_on_1b_logits"].argmax(dim=-1).item())
                    next_on_2b = int(out["next_on_2b_logits"].argmax(dim=-1).item())
                    next_on_3b = int(out["next_on_3b_logits"].argmax(dim=-1).item())
                    next_topbot = int(out["next_inning_topbot_logits"].argmax(dim=-1).item())
                    inning_delta = int(out["inning_delta_logits"].argmax(dim=-1).item())
                    score_delta_id = int(out["score_diff_delta_logits"].argmax(dim=-1).item())

                # Optional: emit pitch-by-pitch replay / rollout trace (JSONL).
                if events_fp is not None:
                    if int(events_max) > 0 and events_written >= int(events_max):
                        pass
                    else:
                        k = max(1, int(events_topk))
                        probs = torch.softmax(type_logits, dim=-1)
                        k = min(k, int(probs.numel()))
                        top = torch.topk(probs, k=k, dim=-1)
                        top_items = []
                        for j in range(k):
                            tid = int(top.indices[j].item())
                            top_items.append(
                                {
                                    "pitch_type_id": tid,
                                    "pitch_type": _tok(pitch_id_to_tok, tid),
                                    "prob": float(top.values[j].item()),
                                }
                            )

                        actual_state = {
                            "inning": int(row["inning"]),
                            "outs_when_up": int(row["outs_when_up"]),
                            "balls": int(row["balls"]),
                            "strikes": int(row["strikes"]),
                            "pitch_number": int(row["pitch_number"]),
                            "score_diff": int(row["score_diff"]),
                            "on_1b_occ": int(row["on_1b_occ"]),
                            "on_2b_occ": int(row["on_2b_occ"]),
                            "on_3b_occ": int(row["on_3b_occ"]),
                            "inning_topbot_id": int(row["inning_topbot_id"]),
                        }
                        sim_state = dict(actual_state) if mode == "replay" else dict(sim)
                        state_match = {}
                        if mode == "rollout":
                            for k2, v2 in actual_state.items():
                                state_match[k2] = int(int(sim_state[k2]) == int(v2))

                        y_desc_id = int(row.get("description_id", 0) or 0) if "description_id" in row else None
                        rec = {
                            "game_pk": int(row["game_pk"]),
                            "at_bat_number": int(row["at_bat_number"]),
                            "pitch_number": int(row["pitch_number"]),
                            "mode": str(mode),
                            "pitcher_id": int(row["pitcher_id"]),
                            "batter_id": int(row["batter_id"]),
                            "stand_id": int(row["stand_id"]),
                            "p_throws_id": int(row["p_throws_id"]),
                            "state": sim_state,
                            **({"actual_state": actual_state, "state_match": state_match} if mode == "rollout" else {}),
                            "y": {
                                "pitch_type_id": int(row["pitch_type_id"]),
                                "pitch_type": _tok(pitch_id_to_tok, int(row["pitch_type_id"])),
                                "plate_x": float(row["plate_x"]),
                                "plate_z": float(row["plate_z"]),
                                **(
                                    {
                                        "description_id": int(y_desc_id),
                                        "description": _tok(desc_id_to_tok, int(y_desc_id)),
                                    }
                                    if y_desc_id is not None
                                    else {}
                                ),
                            },
                            "pred": {
                                "pitch_type_id": int(pred_type),
                                "pitch_type": _tok(pitch_id_to_tok, int(pred_type)),
                                "pitch_type_topk": top_items,
                                "plate_x": float(pred_x_ft),
                                "plate_z": float(pred_z_ft),
                                "loc_nll": float(loc_nll),
                                "pa_end": int(pa_end),
                                "next_balls": int(next_balls),
                                "next_strikes": int(next_strikes),
                            },
                            "metrics": {
                                "type_acc": float(type_acc),
                                "type_top3": float(type_top3),
                                "loc_se_ft2": float(2.0 * loc_mse),
                            },
                        }
                        events_fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
                        events_written += 1

                total_pitches += 1
                sum_type_acc += type_acc
                sum_type_top3 += type_top3
                sum_loc_mse_ft += float(loc_mse)

                if int(row["y_has_next"]) == 1:
                    # Next-state accuracy is only meaningful in replay mode (teacher-forced).
                    if mode == "replay":
                        state_total += 1

                    def _count(name: str, pred: int, y_key: str) -> None:
                        yv = int(row[y_key])
                        if pred == yv:
                            state_correct[name] = state_correct.get(name, 0) + 1

                    if mode == "replay":
                        _count("pa_end", pa_end, "y_pa_end")
                        _count("next_balls", next_balls, "y_next_balls")
                        _count("next_strikes", next_strikes, "y_next_strikes")
                        _count("next_outs_when_up", next_outs, "y_next_outs_when_up")
                        _count("next_on_1b", next_on_1b, "y_next_on_1b_occ")
                        _count("next_on_2b", next_on_2b, "y_next_on_2b_occ")
                        _count("next_on_3b", next_on_3b, "y_next_on_3b_occ")
                        _count("next_inning_topbot", next_topbot, "y_next_inning_topbot_id")
                        _count("inning_delta", inning_delta, "y_inning_delta")
                        _count("score_diff_delta", score_delta_id, "y_score_diff_delta_id")

                # Roll forward state/history (only in rollout mode; replay mode overwrites from data).
                if mode == "rollout":
                    if count_mode == "clamp":
                        # Always trust pa_end, but decode counts with simple constraints.
                        if pa_end == 1:
                            next_balls = 0
                            next_strikes = 0
                        else:
                            nb, ns = _clamped_count_decode(
                                next_balls_logits=next_balls_logits,
                                next_strikes_logits=next_strikes_logits,
                                balls=int(sim["balls"]),
                                strikes=int(sim["strikes"]),
                            )
                            next_balls = int(nb)
                            next_strikes = int(ns)
                    elif count_mode == "rules":
                        if desc_logits is None or desc_id_to_tok is None:
                            raise SimulationError("--count-mode=rules requires a description head + vocab.")
                        if 0 <= int(pred_desc) < len(desc_id_to_tok):
                            tok = desc_id_to_tok[int(pred_desc)]
                            nb, ns, pe, used = _apply_desc_count_rules(
                                tok, balls=int(sim["balls"]), strikes=int(sim["strikes"])
                            )
                            if used:
                                rollout_count["rules_applied_n"] += 1
                                next_balls = int(nb)
                                next_strikes = int(ns)
                                pa_end = int(pe)
                    elif count_mode == "constrained":
                        if desc_logits is None or desc_id_to_tok is None:
                            raise SimulationError("--count-mode=constrained requires a description head + vocab.")
                        nb, ns, pe, chosen_desc, used_rules = _constrained_count_decode(
                            desc_logits=desc_logits,
                            pa_end_logits=pa_end_logits,
                            next_balls_logits=next_balls_logits,
                            next_strikes_logits=next_strikes_logits,
                            desc_id_to_tok=desc_id_to_tok,
                            balls=int(sim["balls"]),
                            strikes=int(sim["strikes"]),
                        )
                        if int(chosen_desc) != int(pred_desc_argmax):
                            rollout_count["constrained_desc_override_n"] += 1
                        pred_desc = int(chosen_desc)
                        next_balls = int(nb)
                        next_strikes = int(ns)
                        pa_end = int(pe)
                        if used_rules:
                            rollout_count["constrained_rules_chosen_n"] += 1
                        else:
                            rollout_count["constrained_heads_chosen_n"] += 1

                    # Update history first (history is within PA).
                    if pa_end == 1:
                        hist_type = []
                        hist_desc = []
                        hist_x = []
                        hist_z = []
                        sim["pitch_number"] = 1
                    else:
                        hist_type.append(pred_type)
                        hist_desc.append(pred_desc)
                        # Store normalized for model consumption.
                        hist_x.append(norms["plate_x"].norm(pred_x_ft))
                        hist_z.append(norms["plate_z"].norm(pred_z_ft))
                        sim["pitch_number"] = int(sim["pitch_number"]) + 1

                    sim["balls"] = next_balls
                    sim["strikes"] = next_strikes
                    sim["outs_when_up"] = next_outs
                    sim["on_1b_occ"] = next_on_1b
                    sim["on_2b_occ"] = next_on_2b
                    sim["on_3b_occ"] = next_on_3b
                    sim["inning_topbot_id"] = next_topbot
                    sim["inning"] = int(sim["inning"]) + inning_delta
                    sim["score_diff"] = int(sim["score_diff"]) + (score_delta_id - 4)

                # Replay history update (teacher-forced) after scoring.
                if mode == "replay":
                    # Use true previous pitch labels and locations to build history.
                    # Only update history if the next pitch is within the same PA.
                    if int(row["y_pa_end"]) == 1:
                        hist_type = []
                        hist_desc = []
                        hist_x = []
                        hist_z = []
                    else:
                        hist_type.append(int(row["pitch_type_id"]))
                        desc_id = int(row.get("description_id", 0) or 0)
                        hist_desc.append(desc_id)
                        hist_x.append(norms["plate_x"].norm(float(row["plate_x"])))
                        hist_z.append(norms["plate_z"].norm(float(row["plate_z"])))
    finally:
        if events_fp is not None:
            events_fp.close()

    # Aggregate summary.
    out: dict[str, Any] = {
        "run_id": run_dir.name,
        "model": model_name,
        "split": split,
        "mode": mode,
        **({"rollout_count": rollout_count} if mode == "rollout" else {}),
        **({"events_out": str(events_path), "events_written": int(events_written)} if events_path is not None else {}),
        "games": len(game_pks),
        "pitches": total_pitches,
        "pitch_type_acc": float(sum_type_acc / max(1, total_pitches)),
        "pitch_type_acc_top3": float(sum_type_top3 / max(1, total_pitches)),
        **(
            {
                "desc_acc": float(sum_desc_acc / max(1, desc_total)),
                "desc_acc_top3": float(sum_desc_top3 / max(1, desc_total)),
                "desc_n": int(desc_total),
            }
            if desc_total > 0
            else {}
        ),
        "loc_rmse_ft": float((sum_loc_mse_ft / max(1, total_pitches)) ** 0.5),
        "state_next_n": int(state_total),
        "state_next_acc": {k: float(v / max(1, state_total)) for k, v in sorted(state_correct.items())},
        "state_curr_n": int(curr_total) if mode == "rollout" else 0,
        "state_curr_match": (
            {k: float(v / max(1, curr_total)) for k, v in sorted(curr_match.items())} if mode == "rollout" else {}
        ),
    }
    return out


def write_json(path: str | None, payload: dict[str, Any]) -> None:
    s = json.dumps(payload, indent=2, sort_keys=True)
    if path is None or path == "-" or path == "":
        print(s)
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(s + "\n", encoding="utf-8")
