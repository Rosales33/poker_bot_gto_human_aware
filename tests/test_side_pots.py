from __future__ import annotations

from poker_ai.engine.payout import compute_side_pots


def test_compute_side_pots_basic() -> None:
    # contrib: A=10, B=30, C=30
    contrib = [10, 30, 30]
    active = [True, True, True]
    pots = compute_side_pots(contrib, active)

    assert len(pots) == 2
    assert int(pots[0].amount) == 30  # 10*3
    assert pots[0].eligible == (0, 1, 2)

    assert int(pots[1].amount) == 40  # (30-10)*2
    assert pots[1].eligible == (1, 2)
