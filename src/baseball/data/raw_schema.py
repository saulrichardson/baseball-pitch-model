from __future__ import annotations

# Central, explicit definition of the raw Statcast columns we require downstream.
#
# This is used both for:
# - downloads: to write a schema-stable empty parquet for date ranges with no games
# - prepare: to validate raw parquet chunks and build prepared datasets
#
# Keep this list in sync with the modeling features/targets in `baseball.data.prepare`.

RAW_TARGET_COLS = [
    "pitch_type",
    "plate_x",
    "plate_z",
]

RAW_ID_COLS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
]

RAW_OUTCOME_COLS = [
    # Post-pitch outcome token (used only as history for later pitches in the same PA).
    "description",
]

RAW_CONTEXT_COLS = [
    # Pre-pitch context.
    "game_date",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "balls",
    "strikes",
    "pitcher",
    "batter",
    "stand",
    "p_throws",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "fld_score",
]

RAW_REQUIRED_COLUMNS = [
    *RAW_TARGET_COLS,
    *RAW_ID_COLS,
    *RAW_OUTCOME_COLS,
    *RAW_CONTEXT_COLS,
]

