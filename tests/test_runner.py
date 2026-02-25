from __future__ import annotations

from poker_ai.agent.random_legal import RandomLegalAgent
from poker_ai.agent.random_policy import RandomPolicy
from poker_ai.engine.state import GameConfig
from poker_ai.experiments.runner import play_match


def test_play_match_runs_multiple_hands() -> None:
    cfg = GameConfig(num_players=6)
    policies = [RandomPolicy(RandomLegalAgent(seed=42 + i)) for i in range(cfg.num_players)]
    results = play_match(cfg, policies, num_hands=50, dealer_start=0)

    assert len(results) == 50
    assert all(r.num_actions > 0 for r in results)
    # sanity: terminal reached
    assert all(r.final_state.is_terminal() for r in results)
