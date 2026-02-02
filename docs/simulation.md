# Simulation (replay vs rollout)

The `simulate` command exists to answer a simple question:

> Does the model still behave sensibly when you run it forward through real games?

That’s different from the usual “teacher-forced” evaluation where the model always gets the **true** game state.

Implementation: `src/baseball/simulate.py`

## Two modes: `replay` vs `rollout`

### `--mode replay` (teacher-forced)

- Uses the real pre‑pitch state from the dataset every step.
- This isolates **pure next‑pitch prediction** quality:
  - pitch type (`top‑1`, `top‑3`, log loss)
  - pitch location (`loc_nll`, RMSE)

### `--mode rollout` (open-loop)

- Feeds the model its predicted next state into the next prediction.
- Errors compound across time → accuracy drops are expected.
- This is how you measure **drift**: does the model stay “on-policy” over long sequences?

## Count evolution (`--count-mode`)

In rollout mode, balls/strikes must be updated without peeking at the real next pitch.

- `heads`
  - Uses the model’s state heads (`next_balls`, `next_strikes`, `pa_end`) directly.

- `clamp`
  - Minimal constraints for count sanity: no decreases, +1 max, and only one of balls/strikes increments per pitch.
  - Works even without an outcome (`description_id`) head.

- `rules`
  - Deterministic count transitions from predicted Statcast `description`.
  - Requires outcome-aware models / vocab (e.g. `transformer_mdn_state_mt`).

- `constrained`
  - Chooses the better of `heads` vs `rules` under hard constraints.
  - Requires outcome-aware models / vocab.

## Pitch-by-pitch replay (visual)

This repo can emit a per‑pitch trace (JSONL) and then render it into a readable “replay strip”.

Each pitch panel shows:
- the pre‑pitch state (count / inning / bases),
- top pitch‑type probabilities,
- actual location (dot) vs predicted mean (cross),
- per‑pitch `loc_nll`.

![Pitch-by-pitch replay strip](assets/replay_example.svg)

The HTML version is in `docs/examples/replay_game716494_example.html` (open locally in a browser).

## How to generate a trace

Replay a single held‑out game and write a JSONL trace:

```bash
python -m baseball simulate \
  --run-id <RUN_ID> \
  --split valid \
  --mode replay \
  --game-pk <GAME_PK> \
  --events-out /tmp/pitch_trace.jsonl \
  --events-topk 5
```

Render that JSONL into an HTML report:

```bash
python -m baseball trace \
  --events /tmp/pitch_trace.jsonl \
  --format html \
  --out /tmp/replay.html \
  --max-at-bats 10
```

Or render a single at‑bat as an embed‑friendly SVG strip:

```bash
python -m baseball trace \
  --events /tmp/pitch_trace.jsonl \
  --format svg \
  --at-bat 17 \
  --out /tmp/replay_strip.svg
```

## Example rollout (scale)

```bash
python -m baseball simulate \
  --run-id <RUN_ID> \
  --split valid \
  --mode rollout \
  --count-mode clamp \
  --max-games 50 \
  --device cuda
```
