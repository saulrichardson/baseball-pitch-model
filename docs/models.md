# Models

This repo contains several model families; the “main line” is the transformer + MDN family.

Implementation lives in: `src/baseball/training/models.py`

## Model families

- `baseline_mlp`
  - A context + flattened-history MLP baseline.
  - Useful as a sanity check and for measuring “sequence model lift”.

- `transformer_mdn`
  - Transformer encoder over within-PA pitch history tokens.
  - Outputs:
    - pitch type logits
    - 2D MDN parameters for `(plate_x, plate_z)`

- `transformer_mdn_mt`
  - Multi-task variant with an extra `description_id` head.
  - Helps the learned representation, and enables outcome-aware history.

- `transformer_mdn_state_mt`
  - Multi-task + state-capable variant.
  - Adds next-state heads so we can do open-loop rollouts (simulation).

## Location head (MDN)

Pitch location is trained as a probabilistic prediction:
- mixture weights
- per-component means in feet
- per-component (diagonal) covariance

This enables:
- `loc_nll` as a proper scoring rule
- `loc_rmse_ft` as a human-readable error metric

