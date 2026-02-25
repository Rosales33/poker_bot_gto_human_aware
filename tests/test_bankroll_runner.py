from __future__ import annotations

from poker_ai.agent.random_legal import RandomLegalAgent
from poker_ai.agent.random_policy import RandomPolicy
from poker_ai.engine.state import GameConfig
from poker_ai.engine.types import Chips
from poker_ai.experiments.runner import play_match_bankroll


def test_play_match_bankroll_conserves_total() -> None:
    cfg = GameConfig(num_players=2, starting_stack=Chips(200))
    policies = [RandomPolicy(RandomLegalAgent(seed=1)), RandomPolicy(RandomLegalAgent(seed=2))]

    traj = play_match_bankroll(cfg, policies, num_hands=50, dealer_start=0, base_seed=0)
    assert len(traj) == 50

    total0 = 2 * int(cfg.starting_stack)
    for _hid, stacks in traj:
        assert sum(int(s) for s in stacks) == total0
