# Serving

This repo includes:
- an export format for trained runs (so inference doesn’t need training code)
- a FastAPI server that serves `/predict`

## Export a bundle

```bash
python -m baseball export --run-id <RUN_ID>
```

This writes to:

```
$BASEBALL_ARTIFACT_ROOT/models/exported/<RUN_ID>/
```

The bundle contains:
- `best.pt` (weights)
- `meta.json` (config + norms)
- vocab JSONs (pitch types, ids, optional description vocab)

## Serve

```bash
python -m baseball serve --host 0.0.0.0 --port 8000
```

Then `POST /predict` with the `PredictRequest` payload defined in:
`src/baseball/serve/api.py`

