# Dataset

Prepared datasets are stored as sharded parquet under:

- `$BASEBALL_ARTIFACT_ROOT/data/prepared/train/*.parquet`
- `$BASEBALL_ARTIFACT_ROOT/data/prepared/valid/*.parquet`

They are designed to be streamable (low RAM) while still being model-friendly (fixed-width history).

## Core columns (examples)

Per-pitch identifiers / labels:
- `pitch_type_id` (target class)
- `plate_x`, `plate_z` (target location in feet)

Pre-pitch context:
- `balls`, `strikes`
- `outs_when_up`
- `inning`, `inning_topbot_id`
- `score_diff`
- `on_1b_occ`, `on_2b_occ`, `on_3b_occ`
- `stand_id` (batter stance), `p_throws_id` (pitcher handedness)

Within-PA history (fixed width, padded with 0/OOV):
- `hist_pitch_type_id_0..L-1`
- `hist_plate_x_0..L-1`
- `hist_plate_z_0..L-1`

Optional outcome-aware history (schema v4):
- `hist_description_id_0..L-1`

## Vocabularies

Prepared data includes vocabulary JSON files under `data/prepared/vocabs/`, and `meta.json` records the paths.

See also:
- `docs/simulation.md` (why `description_id` matters for rollouts)
- `docs/profiling.md` (profile key variants that use previous pitch / outcome tokens)

