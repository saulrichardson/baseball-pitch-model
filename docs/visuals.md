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

## Pitch-by-pitch replay visuals

If you want something more granular than pitcher-level aggregates, generate a **single-game trace** and render it.

1) Produce a JSONL trace (one object per pitch):

```bash
python -m baseball simulate \
  --run-id <RUN_ID> \
  --split valid \
  --mode replay \
  --game-pk <GAME_PK> \
  --events-out /tmp/pitch_trace.jsonl
```

2) Render either:

- a shareable HTML scroll report:

```bash
python -m baseball trace --events /tmp/pitch_trace.jsonl --format html --out /tmp/replay.html
```

- or a single at‑bat SVG “strip” (embed-friendly for GitHub READMEs):

```bash
python -m baseball trace --events /tmp/pitch_trace.jsonl --format svg --at-bat 17 --out /tmp/replay_strip.svg
```

See also: `docs/simulation.md` (replay vs rollout, drift, and trace outputs).
