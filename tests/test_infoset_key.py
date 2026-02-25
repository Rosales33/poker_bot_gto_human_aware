from __future__ import annotations

from poker_ai.abstraction.infoset import make_infoset_key
from poker_ai.engine.dealing import deal_nlhe
from poker_ai.engine.observe import observe
from poker_ai.engine.state import GameConfig, initial_state


def test_infoset_key_is_deterministic() -> None:
    cfg = GameConfig(num_players=6)
    deal = deal_nlhe(cfg.num_players, hand_id=0, base_seed=0)
    s = initial_state(cfg, deal=deal, dealer_pos=0)

    obs = observe(s, s.to_act)
    k1 = make_infoset_key(obs, bb=int(cfg.big_blind))
    k2 = make_infoset_key(obs, bb=int(cfg.big_blind))
    assert k1 == k2
    assert obs.dealer_pos == s.dealer_pos
    assert obs.num_players == s.n
