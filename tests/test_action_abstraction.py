from __future__ import annotations

from poker_ai.abstraction.action_abstraction import (
    ActionAbstractionConfig,
    abstract_to_action,
    get_abstract_actions,
)
from poker_ai.engine.dealing import deal_nlhe
from poker_ai.engine.rules import legal_actions
from poker_ai.engine.state import GameConfig, initial_state


def test_action_abstraction_maps_to_legal_actions() -> None:
    cfg = GameConfig(num_players=6)
    deal = deal_nlhe(cfg.num_players, hand_id=0, base_seed=0)
    state = initial_state(cfg, deal=deal, dealer_pos=0)

    abs_cfg = ActionAbstractionConfig()
    abs_acts = get_abstract_actions(state, abs_cfg)
    assert len(abs_acts) > 0

    legals = legal_actions(state)
    legal_set = {(a.type, int(a.amount)) for a in legals}

    for aa in abs_acts:
        a = abstract_to_action(state, aa)
        assert (a.type, int(a.amount)) in legal_set
