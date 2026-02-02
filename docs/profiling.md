# Profiling (pitcher policies)

The `profile` command generates “pitcher policy tables” from a trained model:
- predicted pitch-type distribution by context bucket
- empirical pitch-type distribution (ground truth) by bucket
- predicted vs empirical mean location by bucket

Implementation: `src/baseball/profile.py`

## Why this exists

If the goal is “what is this pitcher likely to throw next in this situation?”, you want a *conditional policy* view — not just aggregate accuracy.

## Profile keys (`--by`)

Profiles can be built at different granularities.

Less sparse (recommended start):
- `pitcher`
- `pitcher_count`
- `pitcher_count_prev`
- `pitcher_situation`
- `pitcher_situation_prev`
- `pitcher_situation_batter_cluster`
- `pitcher_situation_batter_cluster_prev`

More granular (can be sparse, but aligns with “AB history”):
- `pitcher_count_prev_outcome`
- `pitcher_situation_prev_outcome`
- `pitcher_situation_batter_cluster_prev_outcome`

## Example

```bash
python -m baseball profile \
  --run-id <RUN_ID> \
  --split train \
  --by pitcher_situation_prev \
  --min-n 10
```

