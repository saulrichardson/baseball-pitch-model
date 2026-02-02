from __future__ import annotations

import torch

from baseball.simulate import _clamped_count_decode


def test_clamped_count_decode_respects_constraints() -> None:
    torch.manual_seed(0)
    for balls in [0, 1, 2, 3]:
        for strikes in [0, 1, 2]:
            for _ in range(50):
                b_logits = torch.randn(4)
                s_logits = torch.randn(3)
                nb, ns = _clamped_count_decode(
                    next_balls_logits=b_logits,
                    next_strikes_logits=s_logits,
                    balls=balls,
                    strikes=strikes,
                )
                assert 0 <= nb <= 3
                assert 0 <= ns <= 2
                assert nb >= balls
                assert ns >= strikes
                assert nb <= min(3, balls + 1)
                assert ns <= min(2, strikes + 1)
                # Only one of balls/strikes can increment.
                assert (nb - balls) + (ns - strikes) <= 1

