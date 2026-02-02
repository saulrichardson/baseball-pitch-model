# Artifact layout

All outputs are written under:

```bash
export BASEBALL_ARTIFACT_ROOT="..."
```

Expected layout:

```
data/
  raw/            # cached Statcast downloads (chunked)
  prepared/       # sharded parquet datasets + vocabs + meta.json
runs/
  <run_id>/
    config.json
    metrics.jsonl
    checkpoints/
models/
  exported/       # exported inference bundles
```

Notes:
- `runs/<run_id>/report/` is created by `python -m baseball report`
- simulation outputs are written under `runs/<run_id>/sim/` (see `docs/simulation.md`)
- profiles are written under `runs/<run_id>/profiles/` (see `docs/profiling.md`)

