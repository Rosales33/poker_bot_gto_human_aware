from __future__ import annotations

from poker_ai.agent.random_legal import RandomLegalAgent
from poker_ai.engine.state import GameConfig
from poker_ai.experiments.runner import play_hand


def test_payout_conserves_total_chips() -> None:
    cfg = GameConfig(num_players=6)
    agents = [RandomLegalAgent(seed=100 + i) for i in range(cfg.num_players)]

    r = play_hand(cfg, agents, dealer_pos=0, hand_id=0, base_seed=0)
    final = r.final_state

    total = (
        sum(int(s) for s in final.stacks)
        + int(final.pot)
        + sum(int(x) for x in final.street_commit)
    )
    expected = cfg.num_players * int(cfg.starting_stack)
    assert total == expected
    assert len(r.winners) >= 1
