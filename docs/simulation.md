# Simulation (replay vs rollout)

The `simulate` command exists to measure out-of-sample behavior in *game-like* settings, not just teacher forcing.

Implementation: `src/baseball/simulate.py`

## Modes

- `--mode replay`
  - Teacher-forced.
  - Uses the real pre-pitch state from the dataset every step.

- `--mode rollout`
  - Open-loop.
  - Feeds the model its predicted next-state into the next pitch prediction.
  - Errors compound; accuracy drops are expected and are the point of the experiment.

## Count evolution (`--count-mode`)

Rollouts can update balls/strikes in different ways:

- `heads`
  - Use model next-balls / next-strikes heads directly.

- `clamp`
  - Simple constraints: no decreases, +1 max, and only one of balls/strikes increments.
  - Works even without a `description_id` head.

- `rules`
  - Deterministic transitions from predicted Statcast `description` token.
  - Requires outcome-aware models / vocab.

- `constrained`
  - Scores rules-consistent transitions vs head transitions and takes the better one.
  - Requires outcome-aware models / vocab.

## Example

```bash
python -m baseball simulate \
  --run-id <RUN_ID> \
  --split valid \
  --mode rollout \
  --count-mode clamp \
  --max-games 50 \
  --device cuda
```

## Pitch-by-pitch replay / trace

`simulate` can also emit a **pitch-by-pitch JSONL trace** (one JSON object per pitch) containing:
- the current state used by the model (teacher-forced in `replay`, simulated in `rollout`)
- the actual pitch type + location
- the model’s top‑K pitch type probabilities
- the model’s predicted location mean + per-pitch location NLL

Example: replay a single game and write a trace:

```bash
python -m baseball simulate \
  --run-id <RUN_ID> \
  --split valid \
  --mode replay \
  --game-pk <GAME_PK> \
  --events-out /tmp/pitch_trace.jsonl \
  --events-topk 5
```
