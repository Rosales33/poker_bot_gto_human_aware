from __future__ import annotations

from poker_ai.engine.dealing import deal_nlhe


def test_deal_is_deterministic() -> None:
    d1 = deal_nlhe(num_players=6, hand_id=123, base_seed=0)
    d2 = deal_nlhe(num_players=6, hand_id=123, base_seed=0)
    assert d1 == d2

    d3 = deal_nlhe(num_players=6, hand_id=124, base_seed=0)
    assert d1 != d3
