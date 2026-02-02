from __future__ import annotations

import numpy as np
import pytest

from baseball.profile import _decode_key, _encode_key


@pytest.mark.parametrize(
    "by",
    [
        "pitcher",
        "pitcher_count",
        "pitcher_count_prev",
        "pitcher_count_prev_outcome",
        "pitcher_situation",
        "pitcher_situation_prev",
        "pitcher_situation_prev_outcome",
        "pitcher_situation_batter_cluster",
        "pitcher_situation_batter_cluster_prev",
        "pitcher_situation_batter_cluster_prev_outcome",
    ],
)
def test_profile_key_roundtrip(by: str) -> None:
    rng = np.random.default_rng(0)
    B = 64

    pitcher_id = rng.integers(0, 5000, size=B, dtype=np.int64)
    balls = rng.integers(0, 4, size=B, dtype=np.int64)
    strikes = rng.integers(0, 3, size=B, dtype=np.int64)

    stand_id = rng.integers(0, 4, size=B, dtype=np.int64)
    runners_state = rng.integers(0, 8, size=B, dtype=np.int64)
    inning_bucket = rng.integers(0, 4, size=B, dtype=np.int64)
    score_bucket = rng.integers(0, 7, size=B, dtype=np.int64)

    prev_type = rng.integers(0, 64, size=B, dtype=np.int64)
    prev_desc = rng.integers(0, 128, size=B, dtype=np.int64)
    batter_cluster = rng.integers(0, 33, size=B, dtype=np.int64)

    kwargs: dict[str, object] = {"by": by, "pitcher_id": pitcher_id}

    if by != "pitcher":
        kwargs["balls"] = balls
        kwargs["strikes"] = strikes

    if by in {
        "pitcher_count_prev",
        "pitcher_count_prev_outcome",
        "pitcher_situation_prev",
        "pitcher_situation_prev_outcome",
        "pitcher_situation_batter_cluster_prev",
        "pitcher_situation_batter_cluster_prev_outcome",
    }:
        kwargs["prev_type"] = prev_type

    if by in {"pitcher_count_prev_outcome", "pitcher_situation_prev_outcome", "pitcher_situation_batter_cluster_prev_outcome"}:
        kwargs["prev_desc"] = prev_desc

    if by.startswith("pitcher_situation"):
        kwargs["stand_id"] = stand_id
        kwargs["runners_state"] = runners_state
        kwargs["inning_bucket"] = inning_bucket
        kwargs["score_bucket"] = score_bucket

    if by in {
        "pitcher_situation_batter_cluster",
        "pitcher_situation_batter_cluster_prev",
        "pitcher_situation_batter_cluster_prev_outcome",
    }:
        kwargs["batter_cluster"] = batter_cluster

    keys = _encode_key(**kwargs)  # type: ignore[arg-type]
    assert keys.shape == (B,)

    for i in [0, 1, 2, 7, 31, 63]:
        dec = _decode_key(int(keys[i]), by=by)
        assert dec["pitcher_id"] == int(pitcher_id[i])

        if by != "pitcher":
            assert dec["balls"] == int(balls[i])
            assert dec["strikes"] == int(strikes[i])

        if by.startswith("pitcher_situation"):
            assert dec["stand_id"] == int(stand_id[i])
            assert dec["runners_state"] == int(runners_state[i])
            assert dec["inning_bucket"] == int(inning_bucket[i])
            assert dec["score_bucket"] == int(score_bucket[i])

        if by in {
            "pitcher_situation_batter_cluster",
            "pitcher_situation_batter_cluster_prev",
            "pitcher_situation_batter_cluster_prev_outcome",
        }:
            assert dec["batter_cluster"] == int(batter_cluster[i])

        if by in {
            "pitcher_count_prev",
            "pitcher_count_prev_outcome",
            "pitcher_situation_prev",
            "pitcher_situation_prev_outcome",
            "pitcher_situation_batter_cluster_prev",
            "pitcher_situation_batter_cluster_prev_outcome",
        }:
            assert dec["prev_type_id"] == int(prev_type[i])

        if by in {"pitcher_count_prev_outcome", "pitcher_situation_prev_outcome", "pitcher_situation_batter_cluster_prev_outcome"}:
            assert dec["prev_desc_id"] == int(prev_desc[i])

