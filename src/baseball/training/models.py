from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    history_len: int
    n_pitch_types: int
    n_pitchers: int
    n_batters: int
    n_descriptions: int = 0
    n_stand: int = 4
    n_p_throws: int = 3

    cont_dim: int = 0

    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    mdn_components: int = 8
    gradient_checkpointing: bool = False


class CheckpointedTransformerEncoder(nn.Module):
    """
    Thin wrapper around TransformerEncoderLayer stacks that optionally applies
    gradient checkpointing to each layer during training to reduce activation memory.

    This keeps state_dict keys compatible with `nn.TransformerEncoder` because we
    expose a `.layers` ModuleList with identical layer submodules.
    """

    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, *, num_layers: int, gradient_checkpointing: bool):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(int(num_layers))])
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, src: torch.Tensor, *, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = src
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                if src_key_padding_mask is None:
                    # We never hit this in our models (we always pass a padding mask),
                    # but fail loudly if used incorrectly.
                    raise RuntimeError("gradient_checkpointing requires src_key_padding_mask to be a Tensor.")

                def _f(x_in: torch.Tensor, mask_in: torch.Tensor) -> torch.Tensor:
                    return layer(x_in, src_key_padding_mask=mask_in)

                x = checkpoint(_f, x, src_key_padding_mask, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class BaselineMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        emb_dim = 64
        self.pitcher_emb = nn.Embedding(cfg.n_pitchers, emb_dim)
        self.batter_emb = nn.Embedding(cfg.n_batters, emb_dim)
        self.stand_emb = nn.Embedding(cfg.n_stand, 8)
        self.p_throws_emb = nn.Embedding(cfg.n_p_throws, 8)

        # History: flatten pitch type embeddings + (x,z)
        self.hist_type_emb = nn.Embedding(cfg.n_pitch_types, 16)
        hist_dim = cfg.history_len * (16 + 2)

        hidden = 512
        in_dim = emb_dim + emb_dim + 8 + 8 + cfg.cont_dim + hist_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        self.type_head = nn.Linear(hidden, cfg.n_pitch_types)
        # Location head: simple Gaussian (single component) for baseline.
        self.loc_mu = nn.Linear(hidden, 2)
        self.loc_log_s = nn.Linear(hidden, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_cat = batch["x_cat"]  # [B,4]
        x_cont = batch["x_cont"]  # [B,C]
        hist_type = batch["hist_type"]  # [B,L]
        hist_x = batch["hist_x"]  # [B,L]
        hist_z = batch["hist_z"]  # [B,L]

        pitcher_id = x_cat[:, 0]
        batter_id = x_cat[:, 1]
        stand_id = x_cat[:, 2]
        p_throws_id = x_cat[:, 3]

        e = torch.cat(
            [
                self.pitcher_emb(pitcher_id),
                self.batter_emb(batter_id),
                self.stand_emb(stand_id),
                self.p_throws_emb(p_throws_id),
            ],
            dim=-1,
        )

        h_type = self.hist_type_emb(hist_type)  # [B,L,16]
        h_loc = torch.stack([hist_x, hist_z], dim=-1)  # [B,L,2]
        h = torch.cat([h_type, h_loc], dim=-1).flatten(1)  # [B, L*(18)]

        feats = torch.cat([e, x_cont, h], dim=-1)
        z = self.backbone(feats)

        type_logits = self.type_head(z)
        mu = self.loc_mu(z)
        # Keep sigmas in a numerically stable range (in normalized coordinates).
        log_s = self.loc_log_s(z).clamp(min=-4.0, max=2.0)
        return {"type_logits": type_logits, "loc_mu": mu, "loc_log_s": log_s}


class TransformerMDN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Context embeddings
        self.pitcher_emb = nn.Embedding(cfg.n_pitchers, 64)
        self.batter_emb = nn.Embedding(cfg.n_batters, 64)
        self.stand_emb = nn.Embedding(cfg.n_stand, 8)
        self.p_throws_emb = nn.Embedding(cfg.n_p_throws, 8)

        ctx_dim = 64 + 64 + 8 + 8 + cfg.cont_dim
        self.ctx_proj = nn.Linear(ctx_dim, cfg.d_model)

        # Sequence token embedding: pitch_type + location -> d_model
        self.type_emb = nn.Embedding(cfg.n_pitch_types, cfg.d_model)
        self.desc_emb = nn.Embedding(cfg.n_descriptions, cfg.d_model) if int(cfg.n_descriptions) > 0 else None
        self.loc_proj = nn.Linear(2, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.history_len + 1, cfg.d_model)
        self.token_ln = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = CheckpointedTransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers, gradient_checkpointing=cfg.gradient_checkpointing
        )

        self.final_ln = nn.LayerNorm(cfg.d_model)

        # Heads
        self.type_head = nn.Linear(cfg.d_model, cfg.n_pitch_types)

        # Couple location prediction to the model's pitch-type belief by embedding the
        # predicted type distribution and feeding it into the location head.
        self.type_cond_emb = nn.Embedding(cfg.n_pitch_types, cfg.d_model)
        self.type_cond_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.loc_ln = nn.LayerNorm(cfg.d_model)

        K = cfg.mdn_components
        self.mdn_pi = nn.Linear(cfg.d_model, K)
        self.mdn_mu = nn.Linear(cfg.d_model, K * 2)
        self.mdn_log_s = nn.Linear(cfg.d_model, K * 2)
        self.mdn_rho = nn.Linear(cfg.d_model, K)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_cat = batch["x_cat"]
        x_cont = batch["x_cont"]
        hist_type = batch["hist_type"]
        hist_desc = batch.get("hist_desc")
        hist_x = batch["hist_x"]
        hist_z = batch["hist_z"]

        pitcher_id = x_cat[:, 0]
        batter_id = x_cat[:, 1]
        stand_id = x_cat[:, 2]
        p_throws_id = x_cat[:, 3]

        ctx = torch.cat(
            [
                self.pitcher_emb(pitcher_id),
                self.batter_emb(batter_id),
                self.stand_emb(stand_id),
                self.p_throws_emb(p_throws_id),
                x_cont,
            ],
            dim=-1,
        )
        ctx = self.ctx_proj(ctx)  # [B,d]

        # Build sequence tokens. 0 pitch_type_id denotes padding / no-history.
        loc = torch.stack([hist_x, hist_z], dim=-1)  # [B,L,2]
        hist_tok = self.type_emb(hist_type) + self.loc_proj(loc)
        if self.desc_emb is not None:
            if hist_desc is None:
                raise KeyError(
                    "ModelConfig.n_descriptions > 0 but batch is missing 'hist_desc'. "
                    "Re-run `python -m baseball prepare` with schema_version >= 4 and use the updated loaders."
                )
            hist_tok = hist_tok + self.desc_emb(hist_desc)

        # Prepend a non-masked context token so the transformer always has at least
        # one valid token (important for rows with zero history / all padding).
        tok = torch.cat([ctx.unsqueeze(1), hist_tok], dim=1)  # [B,1+L,d]

        # Add positional embeddings (0 = ctx token, 1..L = history positions).
        pos = torch.arange(tok.size(1), device=tok.device)
        tok = tok + self.pos_emb(pos)[None, :, :]
        tok = self.token_ln(tok)

        pad_mask = torch.cat(
            [torch.zeros((hist_type.size(0), 1), dtype=torch.bool, device=hist_type.device), hist_type.eq(0)],
            dim=1,
        )  # [B,1+L] True where padding

        enc = self.encoder(tok, src_key_padding_mask=pad_mask)  # [B,1+L,d]

        # Use the context token output (position 0) as the pooled representation.
        h = self.final_ln(enc[:, 0, :])

        type_logits = self.type_head(h)

        # Location head conditioning: E[pitch_type] under the model distribution.
        probs = torch.softmax(type_logits, dim=-1)  # [B,T]
        type_ctx = probs @ self.type_cond_emb.weight  # [B,d]
        h_loc = self.loc_ln(h + self.type_cond_proj(type_ctx))

        K = self.cfg.mdn_components
        logit_pi = self.mdn_pi(h_loc)  # [B,K]
        mu = self.mdn_mu(h_loc).view(-1, K, 2)
        # Keep sigmas in a numerically stable range (in normalized coordinates).
        log_s = self.mdn_log_s(h_loc).view(-1, K, 2).clamp(min=-4.0, max=2.0)
        # Avoid extreme correlations which can destabilize likelihood.
        rho = 0.95 * torch.tanh(self.mdn_rho(h_loc))  # [B,K]

        return {
            "type_logits": type_logits,
            "mdn_logit_pi": logit_pi,
            "mdn_mu": mu,
            "mdn_log_sx": log_s[:, :, 0],
            "mdn_log_sz": log_s[:, :, 1],
            "mdn_rho": rho,
        }


class TransformerMDNMT(nn.Module):
    """
    Multi-task TransformerMDN:
    - pitch type classification
    - pitch location (MDN)
    - pitch outcome/description classification (Statcast `description`)

    This enables open-loop rollouts to evolve within-PA outcome tokens without
    peeking at ground truth (and often improves representations for the main tasks).
    Requires cfg.n_descriptions > 0 and `hist_desc` + `y_desc` in the batch.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if int(cfg.n_descriptions) <= 0:
            raise ValueError("TransformerMDNMT requires ModelConfig.n_descriptions > 0 (schema_version >= 4).")

        # Context embeddings
        self.pitcher_emb = nn.Embedding(cfg.n_pitchers, 64)
        self.batter_emb = nn.Embedding(cfg.n_batters, 64)
        self.stand_emb = nn.Embedding(cfg.n_stand, 8)
        self.p_throws_emb = nn.Embedding(cfg.n_p_throws, 8)

        ctx_dim = 64 + 64 + 8 + 8 + cfg.cont_dim
        self.ctx_proj = nn.Linear(ctx_dim, cfg.d_model)

        # Sequence token embedding: type + location + outcome -> d_model
        self.type_emb = nn.Embedding(cfg.n_pitch_types, cfg.d_model)
        self.desc_emb = nn.Embedding(cfg.n_descriptions, cfg.d_model)
        self.loc_proj = nn.Linear(2, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.history_len + 1, cfg.d_model)
        self.token_ln = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = CheckpointedTransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers, gradient_checkpointing=cfg.gradient_checkpointing
        )
        self.final_ln = nn.LayerNorm(cfg.d_model)

        # Pitch type head
        self.type_head = nn.Linear(cfg.d_model, cfg.n_pitch_types)

        # Condition location prediction on pitch-type belief.
        self.type_cond_emb = nn.Embedding(cfg.n_pitch_types, cfg.d_model)
        self.type_cond_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.loc_ln = nn.LayerNorm(cfg.d_model)

        # Location MDN
        K = cfg.mdn_components
        self.mdn_pi = nn.Linear(cfg.d_model, K)
        self.mdn_mu = nn.Linear(cfg.d_model, K * 2)
        self.mdn_log_s = nn.Linear(cfg.d_model, K * 2)
        self.mdn_rho = nn.Linear(cfg.d_model, K)

        # Description head: condition on type belief + predicted location mean.
        self.loc_mean_proj = nn.Linear(2, cfg.d_model)
        self.desc_ln = nn.LayerNorm(cfg.d_model)
        self.desc_head = nn.Linear(cfg.d_model, cfg.n_descriptions)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_cat = batch["x_cat"]
        x_cont = batch["x_cont"]
        hist_type = batch["hist_type"]
        hist_desc = batch.get("hist_desc")
        hist_x = batch["hist_x"]
        hist_z = batch["hist_z"]

        if hist_desc is None:
            raise KeyError(
                "TransformerMDNMT requires 'hist_desc' (outcome-aware AB history). "
                "Re-run `python -m baseball prepare` with schema_version >= 4 and use the updated loaders."
            )

        pitcher_id = x_cat[:, 0]
        batter_id = x_cat[:, 1]
        stand_id = x_cat[:, 2]
        p_throws_id = x_cat[:, 3]

        ctx = torch.cat(
            [
                self.pitcher_emb(pitcher_id),
                self.batter_emb(batter_id),
                self.stand_emb(stand_id),
                self.p_throws_emb(p_throws_id),
                x_cont,
            ],
            dim=-1,
        )
        ctx = self.ctx_proj(ctx)  # [B,d]

        loc = torch.stack([hist_x, hist_z], dim=-1)  # [B,L,2]
        hist_tok = self.type_emb(hist_type) + self.loc_proj(loc) + self.desc_emb(hist_desc)

        tok = torch.cat([ctx.unsqueeze(1), hist_tok], dim=1)  # [B,1+L,d]
        pos = torch.arange(tok.size(1), device=tok.device)
        tok = self.token_ln(tok + self.pos_emb(pos)[None, :, :])

        pad_mask = torch.cat(
            [torch.zeros((hist_type.size(0), 1), dtype=torch.bool, device=hist_type.device), hist_type.eq(0)],
            dim=1,
        )
        enc = self.encoder(tok, src_key_padding_mask=pad_mask)
        h = self.final_ln(enc[:, 0, :])

        type_logits = self.type_head(h)

        probs = torch.softmax(type_logits, dim=-1)
        type_ctx = probs @ self.type_cond_emb.weight
        type_ctx = self.type_cond_proj(type_ctx)

        h_loc = self.loc_ln(h + type_ctx)
        K = self.cfg.mdn_components
        logit_pi = self.mdn_pi(h_loc)
        mu = self.mdn_mu(h_loc).view(-1, K, 2)
        log_s = self.mdn_log_s(h_loc).view(-1, K, 2).clamp(min=-4.0, max=2.0)
        rho = 0.95 * torch.tanh(self.mdn_rho(h_loc))

        pi = torch.softmax(logit_pi, dim=-1)
        loc_mean = torch.einsum("bk,bkd->bd", pi, mu)  # [B,2] (normalized)

        h_desc = self.desc_ln(h + type_ctx + self.loc_mean_proj(loc_mean))
        desc_logits = self.desc_head(h_desc)

        return {
            "type_logits": type_logits,
            "desc_logits": desc_logits,
            "mdn_logit_pi": logit_pi,
            "mdn_mu": mu,
            "mdn_log_sx": log_s[:, :, 0],
            "mdn_log_sz": log_s[:, :, 1],
            "mdn_rho": rho,
        }


class TransformerMDNV2(nn.Module):
    """
    Transformer + MDN model with a more expressive history tokenization.

    Instead of summing embeddings (type + location + outcome), we concatenate
    per-field embeddings/projections and project back to d_model. This lets the
    model preserve information from each field without forcing it into a single
    shared embedding space prematurely.

    Supports optional outcome history via `hist_desc` when cfg.n_descriptions > 0.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Context embeddings
        self.pitcher_emb = nn.Embedding(cfg.n_pitchers, 64)
        self.batter_emb = nn.Embedding(cfg.n_batters, 64)
        self.stand_emb = nn.Embedding(cfg.n_stand, 8)
        self.p_throws_emb = nn.Embedding(cfg.n_p_throws, 8)

        ctx_dim = 64 + 64 + 8 + 8 + cfg.cont_dim
        self.ctx_proj = nn.Linear(ctx_dim, cfg.d_model)

        # History tokenization: concat(type_emb, loc_proj, desc_emb) -> project to d_model
        loc_dim = max(16, cfg.d_model // 4)
        desc_dim = max(16, cfg.d_model // 4) if int(cfg.n_descriptions) > 0 else 0
        type_dim = int(cfg.d_model) - int(loc_dim) - int(desc_dim)
        if type_dim <= 0:
            raise ValueError(
                f"Invalid token dims (d_model={cfg.d_model}, loc_dim={loc_dim}, desc_dim={desc_dim})."
            )

        self.type_tok_emb = nn.Embedding(cfg.n_pitch_types, type_dim)
        self.loc_tok_proj = nn.Linear(2, loc_dim)
        self.desc_tok_emb = nn.Embedding(cfg.n_descriptions, desc_dim) if int(cfg.n_descriptions) > 0 else None
        tok_in_dim = type_dim + loc_dim + desc_dim
        self.tok_proj = nn.Linear(tok_in_dim, cfg.d_model)

        self.pos_emb = nn.Embedding(cfg.history_len + 1, cfg.d_model)
        self.token_ln = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = CheckpointedTransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers, gradient_checkpointing=cfg.gradient_checkpointing
        )
        self.final_ln = nn.LayerNorm(cfg.d_model)

        # Heads
        self.type_head = nn.Linear(cfg.d_model, cfg.n_pitch_types)

        # Condition location prediction on pitch-type belief.
        self.type_cond_emb = nn.Embedding(cfg.n_pitch_types, cfg.d_model)
        self.type_cond_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.loc_ln = nn.LayerNorm(cfg.d_model)

        K = cfg.mdn_components
        self.mdn_pi = nn.Linear(cfg.d_model, K)
        self.mdn_mu = nn.Linear(cfg.d_model, K * 2)
        self.mdn_log_s = nn.Linear(cfg.d_model, K * 2)
        self.mdn_rho = nn.Linear(cfg.d_model, K)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_cat = batch["x_cat"]
        x_cont = batch["x_cont"]
        hist_type = batch["hist_type"]
        hist_desc = batch.get("hist_desc")
        hist_x = batch["hist_x"]
        hist_z = batch["hist_z"]

        pitcher_id = x_cat[:, 0]
        batter_id = x_cat[:, 1]
        stand_id = x_cat[:, 2]
        p_throws_id = x_cat[:, 3]

        ctx = torch.cat(
            [
                self.pitcher_emb(pitcher_id),
                self.batter_emb(batter_id),
                self.stand_emb(stand_id),
                self.p_throws_emb(p_throws_id),
                x_cont,
            ],
            dim=-1,
        )
        ctx = self.ctx_proj(ctx)  # [B,d]

        loc = torch.stack([hist_x, hist_z], dim=-1)  # [B,L,2]
        parts = [self.type_tok_emb(hist_type), self.loc_tok_proj(loc)]
        if self.desc_tok_emb is not None:
            if hist_desc is None:
                raise KeyError(
                    "ModelConfig.n_descriptions > 0 but batch is missing 'hist_desc'. "
                    "Re-run `python -m baseball prepare` with schema_version >= 4."
                )
            parts.append(self.desc_tok_emb(hist_desc))
        tok_in = torch.cat(parts, dim=-1)
        hist_tok = self.tok_proj(tok_in)

        tok = torch.cat([ctx.unsqueeze(1), hist_tok], dim=1)  # [B,1+L,d]
        pos = torch.arange(tok.size(1), device=tok.device)
        tok = self.token_ln(tok + self.pos_emb(pos)[None, :, :])

        pad_mask = torch.cat(
            [torch.zeros((hist_type.size(0), 1), dtype=torch.bool, device=hist_type.device), hist_type.eq(0)],
            dim=1,
        )
        enc = self.encoder(tok, src_key_padding_mask=pad_mask)
        h = self.final_ln(enc[:, 0, :])

        type_logits = self.type_head(h)

        probs = torch.softmax(type_logits, dim=-1)
        type_ctx = probs @ self.type_cond_emb.weight
        h_loc = self.loc_ln(h + self.type_cond_proj(type_ctx))

        K = self.cfg.mdn_components
        logit_pi = self.mdn_pi(h_loc)
        mu = self.mdn_mu(h_loc).view(-1, K, 2)
        log_s = self.mdn_log_s(h_loc).view(-1, K, 2).clamp(min=-4.0, max=2.0)
        rho = 0.95 * torch.tanh(self.mdn_rho(h_loc))

        return {
            "type_logits": type_logits,
            "mdn_logit_pi": logit_pi,
            "mdn_mu": mu,
            "mdn_log_sx": log_s[:, :, 0],
            "mdn_log_sz": log_s[:, :, 1],
            "mdn_rho": rho,
        }


class TransformerMDNState(nn.Module):
    """
    Multi-task transformer:
    - pitch type classification
    - pitch location (MDN)
    - next-state prediction (for simulation without peeking at the real next pitch state)

    The next-state heads are trained with a mask (y_has_next) because the final pitch
    of a game has no next state.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Context embeddings
        self.pitcher_emb = nn.Embedding(cfg.n_pitchers, 64)
        self.batter_emb = nn.Embedding(cfg.n_batters, 64)
        self.stand_emb = nn.Embedding(cfg.n_stand, 8)
        self.p_throws_emb = nn.Embedding(cfg.n_p_throws, 8)

        ctx_dim = 64 + 64 + 8 + 8 + cfg.cont_dim
        self.ctx_proj = nn.Linear(ctx_dim, cfg.d_model)

        # Sequence token embedding: pitch_type + location -> d_model
        self.type_emb = nn.Embedding(cfg.n_pitch_types, cfg.d_model)
        self.desc_emb = nn.Embedding(cfg.n_descriptions, cfg.d_model) if int(cfg.n_descriptions) > 0 else None
        self.loc_proj = nn.Linear(2, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.history_len + 1, cfg.d_model)
        self.token_ln = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = CheckpointedTransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers, gradient_checkpointing=cfg.gradient_checkpointing
        )
        self.final_ln = nn.LayerNorm(cfg.d_model)

        # Pitch type head
        self.type_head = nn.Linear(cfg.d_model, cfg.n_pitch_types)

        # Condition both location + state on the model's pitch-type belief.
        self.type_cond_emb = nn.Embedding(cfg.n_pitch_types, cfg.d_model)
        self.type_cond_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # Location head (MDN)
        self.loc_ln = nn.LayerNorm(cfg.d_model)
        K = cfg.mdn_components
        self.mdn_pi = nn.Linear(cfg.d_model, K)
        self.mdn_mu = nn.Linear(cfg.d_model, K * 2)
        self.mdn_log_s = nn.Linear(cfg.d_model, K * 2)
        self.mdn_rho = nn.Linear(cfg.d_model, K)

        # State head
        self.loc_mean_proj = nn.Linear(2, cfg.d_model)
        self.state_ln = nn.LayerNorm(cfg.d_model)

        # Discrete next-state heads (class counts are fixed by design).
        self.pa_end_head = nn.Linear(cfg.d_model, 2)
        self.next_balls_head = nn.Linear(cfg.d_model, 4)
        self.next_strikes_head = nn.Linear(cfg.d_model, 3)
        self.next_outs_head = nn.Linear(cfg.d_model, 3)
        self.next_on_1b_head = nn.Linear(cfg.d_model, 2)
        self.next_on_2b_head = nn.Linear(cfg.d_model, 2)
        self.next_on_3b_head = nn.Linear(cfg.d_model, 2)
        self.next_topbot_head = nn.Linear(cfg.d_model, 2)
        self.inning_delta_head = nn.Linear(cfg.d_model, 2)
        self.score_delta_head = nn.Linear(cfg.d_model, 9)  # [-4..4] shifted by +4

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_cat = batch["x_cat"]
        x_cont = batch["x_cont"]
        hist_type = batch["hist_type"]
        hist_desc = batch.get("hist_desc")
        hist_x = batch["hist_x"]
        hist_z = batch["hist_z"]

        pitcher_id = x_cat[:, 0]
        batter_id = x_cat[:, 1]
        stand_id = x_cat[:, 2]
        p_throws_id = x_cat[:, 3]

        ctx = torch.cat(
            [
                self.pitcher_emb(pitcher_id),
                self.batter_emb(batter_id),
                self.stand_emb(stand_id),
                self.p_throws_emb(p_throws_id),
                x_cont,
            ],
            dim=-1,
        )
        ctx = self.ctx_proj(ctx)  # [B,d]

        # Build sequence tokens. 0 pitch_type_id denotes padding / no-history.
        loc = torch.stack([hist_x, hist_z], dim=-1)  # [B,L,2]
        hist_tok = self.type_emb(hist_type) + self.loc_proj(loc)
        if self.desc_emb is not None:
            if hist_desc is None:
                raise KeyError(
                    "ModelConfig.n_descriptions > 0 but batch is missing 'hist_desc'. "
                    "Re-run `python -m baseball prepare` with schema_version >= 4 and use the updated loaders."
                )
            hist_tok = hist_tok + self.desc_emb(hist_desc)

        # Prepend context token.
        tok = torch.cat([ctx.unsqueeze(1), hist_tok], dim=1)  # [B,1+L,d]

        # Add positional embeddings (0 = ctx token, 1..L = history positions).
        pos = torch.arange(tok.size(1), device=tok.device)
        tok = tok + self.pos_emb(pos)[None, :, :]
        tok = self.token_ln(tok)

        pad_mask = torch.cat(
            [torch.zeros((hist_type.size(0), 1), dtype=torch.bool, device=hist_type.device), hist_type.eq(0)],
            dim=1,
        )  # [B,1+L] True where padding

        enc = self.encoder(tok, src_key_padding_mask=pad_mask)  # [B,1+L,d]
        h = self.final_ln(enc[:, 0, :])  # pooled ctx token

        type_logits = self.type_head(h)

        # Pitch-type belief embedding (distribution-weighted).
        probs = torch.softmax(type_logits, dim=-1)  # [B,T]
        type_ctx = probs @ self.type_cond_emb.weight  # [B,d]
        type_ctx = self.type_cond_proj(type_ctx)

        # Location MDN conditioned on type belief.
        h_loc = self.loc_ln(h + type_ctx)
        K = self.cfg.mdn_components
        logit_pi = self.mdn_pi(h_loc)  # [B,K]
        mu = self.mdn_mu(h_loc).view(-1, K, 2)
        log_s = self.mdn_log_s(h_loc).view(-1, K, 2).clamp(min=-4.0, max=2.0)
        rho = 0.95 * torch.tanh(self.mdn_rho(h_loc))  # [B,K]

        # Approximate location mean in normalized coords (differentiable).
        pi = torch.softmax(logit_pi, dim=-1)  # [B,K]
        loc_mean = torch.einsum("bk,bkd->bd", pi, mu)  # [B,2]

        # State prediction conditioned on type belief + predicted location mean.
        h_state = self.state_ln(h + type_ctx + self.loc_mean_proj(loc_mean))

        return {
            # Pitch prediction
            "type_logits": type_logits,
            "mdn_logit_pi": logit_pi,
            "mdn_mu": mu,
            "mdn_log_sx": log_s[:, :, 0],
            "mdn_log_sz": log_s[:, :, 1],
            "mdn_rho": rho,
            # State prediction
            "pa_end_logits": self.pa_end_head(h_state),
            "next_balls_logits": self.next_balls_head(h_state),
            "next_strikes_logits": self.next_strikes_head(h_state),
            "next_outs_when_up_logits": self.next_outs_head(h_state),
            "next_on_1b_logits": self.next_on_1b_head(h_state),
            "next_on_2b_logits": self.next_on_2b_head(h_state),
            "next_on_3b_logits": self.next_on_3b_head(h_state),
            "next_inning_topbot_logits": self.next_topbot_head(h_state),
            "inning_delta_logits": self.inning_delta_head(h_state),
            "score_diff_delta_logits": self.score_delta_head(h_state),
        }


class TransformerMDNStateMT(nn.Module):
    """
    TransformerMDNState + description head:
    - pitch type
    - pitch location (MDN)
    - next-state (for open-loop simulation)
    - pitch outcome/description (enables updating within-PA outcome tokens during rollout)

    Requires cfg.n_descriptions > 0 and `hist_desc` + `y_desc` in the batch.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if int(cfg.n_descriptions) <= 0:
            raise ValueError(
                "TransformerMDNStateMT requires ModelConfig.n_descriptions > 0 (schema_version >= 4)."
            )

        # Context embeddings
        self.pitcher_emb = nn.Embedding(cfg.n_pitchers, 64)
        self.batter_emb = nn.Embedding(cfg.n_batters, 64)
        self.stand_emb = nn.Embedding(cfg.n_stand, 8)
        self.p_throws_emb = nn.Embedding(cfg.n_p_throws, 8)

        ctx_dim = 64 + 64 + 8 + 8 + cfg.cont_dim
        self.ctx_proj = nn.Linear(ctx_dim, cfg.d_model)

        # Sequence token embedding: pitch_type + location + outcome -> d_model
        self.type_emb = nn.Embedding(cfg.n_pitch_types, cfg.d_model)
        self.desc_emb = nn.Embedding(cfg.n_descriptions, cfg.d_model)
        self.loc_proj = nn.Linear(2, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.history_len + 1, cfg.d_model)
        self.token_ln = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = CheckpointedTransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers, gradient_checkpointing=cfg.gradient_checkpointing
        )
        self.final_ln = nn.LayerNorm(cfg.d_model)

        # Pitch type head
        self.type_head = nn.Linear(cfg.d_model, cfg.n_pitch_types)

        # Condition both location + state on the model's pitch-type belief.
        self.type_cond_emb = nn.Embedding(cfg.n_pitch_types, cfg.d_model)
        self.type_cond_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # Location head (MDN)
        self.loc_ln = nn.LayerNorm(cfg.d_model)
        K = cfg.mdn_components
        self.mdn_pi = nn.Linear(cfg.d_model, K)
        self.mdn_mu = nn.Linear(cfg.d_model, K * 2)
        self.mdn_log_s = nn.Linear(cfg.d_model, K * 2)
        self.mdn_rho = nn.Linear(cfg.d_model, K)

        # Approximate MDN mean -> representation.
        self.loc_mean_proj = nn.Linear(2, cfg.d_model)

        # State head
        self.state_ln = nn.LayerNorm(cfg.d_model)
        self.pa_end_head = nn.Linear(cfg.d_model, 2)
        self.next_balls_head = nn.Linear(cfg.d_model, 4)
        self.next_strikes_head = nn.Linear(cfg.d_model, 3)
        self.next_outs_head = nn.Linear(cfg.d_model, 3)
        self.next_on_1b_head = nn.Linear(cfg.d_model, 2)
        self.next_on_2b_head = nn.Linear(cfg.d_model, 2)
        self.next_on_3b_head = nn.Linear(cfg.d_model, 2)
        self.next_topbot_head = nn.Linear(cfg.d_model, 2)
        self.inning_delta_head = nn.Linear(cfg.d_model, 2)
        self.score_delta_head = nn.Linear(cfg.d_model, 9)

        # Description head (trained on current pitch description_id).
        self.desc_head = nn.Linear(cfg.d_model, cfg.n_descriptions)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_cat = batch["x_cat"]
        x_cont = batch["x_cont"]
        hist_type = batch["hist_type"]
        hist_desc = batch.get("hist_desc")
        hist_x = batch["hist_x"]
        hist_z = batch["hist_z"]

        if hist_desc is None:
            raise KeyError(
                "TransformerMDNStateMT requires 'hist_desc' (outcome-aware AB history). "
                "Re-run `python -m baseball prepare` with schema_version >= 4 and use the updated loaders."
            )

        pitcher_id = x_cat[:, 0]
        batter_id = x_cat[:, 1]
        stand_id = x_cat[:, 2]
        p_throws_id = x_cat[:, 3]

        ctx = torch.cat(
            [
                self.pitcher_emb(pitcher_id),
                self.batter_emb(batter_id),
                self.stand_emb(stand_id),
                self.p_throws_emb(p_throws_id),
                x_cont,
            ],
            dim=-1,
        )
        ctx = self.ctx_proj(ctx)

        loc = torch.stack([hist_x, hist_z], dim=-1)
        hist_tok = self.type_emb(hist_type) + self.loc_proj(loc) + self.desc_emb(hist_desc)

        tok = torch.cat([ctx.unsqueeze(1), hist_tok], dim=1)
        pos = torch.arange(tok.size(1), device=tok.device)
        tok = self.token_ln(tok + self.pos_emb(pos)[None, :, :])

        pad_mask = torch.cat(
            [torch.zeros((hist_type.size(0), 1), dtype=torch.bool, device=hist_type.device), hist_type.eq(0)],
            dim=1,
        )
        enc = self.encoder(tok, src_key_padding_mask=pad_mask)
        h = self.final_ln(enc[:, 0, :])

        type_logits = self.type_head(h)

        probs = torch.softmax(type_logits, dim=-1)
        type_ctx = probs @ self.type_cond_emb.weight
        type_ctx = self.type_cond_proj(type_ctx)

        h_loc = self.loc_ln(h + type_ctx)
        K = self.cfg.mdn_components
        logit_pi = self.mdn_pi(h_loc)
        mu = self.mdn_mu(h_loc).view(-1, K, 2)
        log_s = self.mdn_log_s(h_loc).view(-1, K, 2).clamp(min=-4.0, max=2.0)
        rho = 0.95 * torch.tanh(self.mdn_rho(h_loc))

        pi = torch.softmax(logit_pi, dim=-1)
        loc_mean = torch.einsum("bk,bkd->bd", pi, mu)

        h_state = self.state_ln(h + type_ctx + self.loc_mean_proj(loc_mean))
        desc_logits = self.desc_head(h_state)

        return {
            "type_logits": type_logits,
            "desc_logits": desc_logits,
            "mdn_logit_pi": logit_pi,
            "mdn_mu": mu,
            "mdn_log_sx": log_s[:, :, 0],
            "mdn_log_sz": log_s[:, :, 1],
            "mdn_rho": rho,
            "pa_end_logits": self.pa_end_head(h_state),
            "next_balls_logits": self.next_balls_head(h_state),
            "next_strikes_logits": self.next_strikes_head(h_state),
            "next_outs_when_up_logits": self.next_outs_head(h_state),
            "next_on_1b_logits": self.next_on_1b_head(h_state),
            "next_on_2b_logits": self.next_on_2b_head(h_state),
            "next_on_3b_logits": self.next_on_3b_head(h_state),
            "next_inning_topbot_logits": self.next_topbot_head(h_state),
            "inning_delta_logits": self.inning_delta_head(h_state),
            "score_diff_delta_logits": self.score_delta_head(h_state),
        }
