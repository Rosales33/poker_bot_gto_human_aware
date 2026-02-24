from __future__ import annotations

from poker_ai.engine.dealing import deal_nlhe
from poker_ai.engine.observe import observe
from poker_ai.engine.state import GameConfig, initial_state


def test_observation_hides_other_hole_cards() -> None:
    cfg = GameConfig(num_players=6)
    deal = deal_nlhe(cfg.num_players, hand_id=0, base_seed=0)
    s = initial_state(cfg, deal=deal, dealer_pos=0)

    obs0 = observe(s, 0)
    obs1 = observe(s, 1)

    assert obs0.hole_cards != obs1.hole_cards  # almost surely true
    # importantly, Observation has no attribute like "all_hole_cards"
    assert not hasattr(obs0, "hole_cards_all")
    assert not hasattr(obs0, "all_hole_cards")
