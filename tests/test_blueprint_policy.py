from __future__ import annotations

from poker_ai.abstraction.action_abstraction import ActionAbstractionConfig
from poker_ai.agent.blueprint_agent import BlueprintPolicy
from poker_ai.engine.dealing import deal_nlhe
from poker_ai.engine.observe import observe
from poker_ai.engine.rules import legal_actions
from poker_ai.engine.state import GameConfig, initial_state
from poker_ai.solver.strategy_store import StoredStrategy


def test_blueprint_policy_fallback_is_legal() -> None:
    cfg = GameConfig(num_players=2)
    deal = deal_nlhe(cfg.num_players, hand_id=0, base_seed=0)
    state = initial_state(cfg, deal=deal, dealer_pos=0)

    # empty stored strategy -> fallback uniform
    stored = StoredStrategy(
        game_cfg=cfg,
        action_abs_cfg=ActionAbstractionConfig(),
        action_lists={},
        strategy_sum={},
    )
    policy = BlueprintPolicy(stored, seed=0)

    obs = observe(state, state.to_act)
    act = policy.act(state, obs)

    legal = {(a.type, int(a.amount)) for a in legal_actions(state)}
    assert (act.type, int(act.amount)) in legal
