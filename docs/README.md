# Documentation

This folder holds the “how to run it” and “how it works” material. The repo root `README.md` is intentionally high-level and visual.

## Getting started

- `docs/setup.md` — install + first run (local or Greene)
- `docs/workflows-greene.md` — sbatch workflows + $VAST conventions

## How it works

- `docs/artifacts.md` — artifact layout under `$BASEBALL_ARTIFACT_ROOT`
- `docs/models.md` — model families (Transformer + MDN, multi-task, state-capable)
- `docs/dataset.md` — prepared dataset schema + feature conventions

## Evaluation / analysis

- `docs/reporting.md` — writeup-ready metrics + calibration + baselines + slices
- `docs/simulation.md` — replay vs rollout, count modes, drift measurement
- `docs/profiling.md` — pitcher policy tables + grouping keys
- `docs/visuals.md` — generate SVG pitcher visuals

## Deployment

- `docs/serving.md` — export bundles + FastAPI `/predict`

## Narrative writeup

- `docs/writeup.md` — deeper technical narrative + example results

## Examples

- `docs/examples/` — small, shareable artifacts (pitch-by-pitch replay HTML + JSONL snippets)
