# Visuals

This repo is intentionally **visual-first** at the landing page level: the root `README.md` embeds SVGs generated from model artifacts.

These visuals are produced from **profile parquet outputs**, not notebooks.

## Generate pitcher visuals

First generate a profile parquet (on Greene recommended):

```bash
python -m baseball profile --run-id <RUN_ID> --split train --by pitcher_situation_prev_outcome
```

Then generate SVGs (count-policy grid + zone heatmaps):

```bash
python -m baseball viz \
  --run-id <RUN_ID> \
  --by pitcher_situation_prev_outcome \
  --split train \
  --pitcher 543037 \
  --pitch-types FF,SL \
  --out-dir docs/assets
```

Alternatively, point directly at an existing profile parquet:

```bash
python -m baseball viz \
  --profile-parquet /path/to/pitcher_profiles_*.parquet \
  --pitcher 543037 \
  --pitch-types FF,SL \
  --out-dir docs/assets
```

Outputs:
- `docs/assets/profile_<pitcher>_count_policy.svg`
- `docs/assets/profile_<pitcher>_zone_heatmaps.svg`

## What these plots mean

- **Count policy grid**: a 4×3 grid (balls × strikes) where each cell shows the **top pitch type** and its probability.
- **Zone heatmaps**: strike-zone heatmaps built from profile-group mean locations, smoothed and weighted by pitch-type probability.

