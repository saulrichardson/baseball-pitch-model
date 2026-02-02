# Baseball Pitch Model

Transformer-first modeling of **what pitch gets thrown next**.

At each pitch, the model predicts:
- **Pitch type** as a probability distribution (classification)
- **Pitch location** as a probability density over the batter box (probabilistic regression via a 2D mixture model)

Those predictions are conditioned on:
- within‑AB pitch history (types + locations, optionally outcomes),
- pre‑pitch game state (count / outs / runners / score / inning),
- pitcher/batter identity + handedness/stance.

![Overview diagram](docs/assets/overview.svg)

## A quick tour (what’s in the repo)

This repo is structured around the idea that “pitch prediction” is a **policy modeling** problem:
you want a model you can *interrogate* (not just a single accuracy number).

You can:
- train transformer models on large Statcast datasets (streaming parquet; GPU-friendly),
- generate writeup-ready reports (calibration + slice metrics + strong empirical baselines),
- run open‑loop rollouts on held‑out games to measure drift,
- build pitcher “policy tables” (what a pitcher throws next by count/situation),
- export bundles + serve predictions via FastAPI.

## What’s unique here

- **Sequence policy modeling**: the transformer learns “AB context”, not just pitcher priors.
- **Probabilistic location**: location is scored with `loc_nll` (proper scoring rule), not only RMSE.
- **Open-loop rollouts**: the repo can simulate held-out games to study drift (`replay` vs `rollout`).
- **Pitcher policies**: export per-pitcher conditional “what do they throw next in this situation?” tables.
- **Writeup-ready evaluation**: calibration + slice metrics + strong empirical baselines.

## Pipeline (end-to-end)

```mermaid
flowchart LR
  A[Statcast via pybaseball] --> B[Raw chunked parquet]
  B --> C[Prepared dataset<br/>sharded parquet]
  C --> D[Train transformer + MDN]
  D --> E[Report<br/>metrics + calibration + baselines + slices]
  D --> F[Simulate<br/>replay vs rollout]
  D --> G[Profile<br/>pitcher policies]
  D --> H[Export bundle]
  H --> I[FastAPI /predict]
```

## Results snapshot (example run)

One strong 2023-only run (`final_state_mt_4665878`, valid, teacher-forced):

| Metric | Value |
|---|---:|
| Pitch type top‑1 | 0.478 |
| Pitch type top‑3 | 0.897 |
| Location RMSE (ft) | 0.875 |
| Calibration ECE (15 bins) | 0.041 |

Open-loop rollouts (50 held-out games) predictably drop due to drift:
| Setting | Top‑1 | Top‑3 |
|---|---:|---:|
| rollout (heads) | 0.402 | 0.830 |
| rollout (clamp count) | 0.403 | 0.835 |

## Pitcher profile example (the “policy view”)

Example pitcher: **Gerrit Cole** (MLBAM `543037`).

Count-policy heatmap: for each count (balls × strikes), show the **top pitch type** and its probability.
- left = model prediction
- right = empirical distribution from the dataset

![Gerrit Cole count policy](docs/assets/profile_543037_count_policy.svg)

Zone heatmaps: batter‑box location density for specific pitch types (here: FF + SL).
- left = predicted location density
- right = empirical location density

![Gerrit Cole zone heatmaps](docs/assets/profile_543037_zone_heatmaps.svg)

## Docs

- Start here: `docs/README.md`
- Setup: `docs/setup.md`
- Greene workflows: `docs/workflows-greene.md`
- Visual generation: `docs/visuals.md`
- Narrative writeup: `docs/writeup.md`
