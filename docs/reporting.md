# Reporting

The `report` command is the “writeup-ready” evaluation pass:

- overall metrics (pitch type + location)
- calibration (ECE bins + Brier) for pitch type probabilities
- slice metrics (count / inning / runners / score / batter stance / pitcher handedness)
- empirical baselines (fit on `train`, evaluated on a split)

## Run

```bash
python -m baseball report --run-id <RUN_ID> --split valid
```

## Outputs

Written under:

```
$BASEBALL_ARTIFACT_ROOT/runs/<RUN_ID>/report/
```

Files:
- `report.json` — summary pointer file
- `model_metrics.json` — primary metrics
- `baselines.json` — fitted baselines + eval
- `slices.json` — slice breakdown tables

## Baselines (what they mean)

Baselines are designed to answer: “how far above pitcher tendencies is the model?”

Examples:
- `global`: one pitch mix for everyone
- `pitcher`: per-pitcher pitch mix
- `pitcher_count`: per-pitcher-by-count pitch mix
- `pitcher_count_prev`: per-pitcher-by-count-by-prev pitch mix

They are fitted by streaming parquet to stay RAM-efficient.

