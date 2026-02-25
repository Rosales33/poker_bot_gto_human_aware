from __future__ import annotations

from poker_ai.engine.state import GameConfig
from poker_ai.engine.types import Chips
from poker_ai.solver.mccfr import MCCFRConfig, MCCFRTrainer


def test_mccfr_runs_smoke() -> None:
    # Keep it small/fast for CI-like testing
    cfg = GameConfig(num_players=2, starting_stack=Chips(200))  # HU is faster
    trainer = MCCFRTrainer(cfg, MCCFRConfig(iterations=200, seed=1, base_deal_seed=1))
    trainer.run()

    assert len(trainer.regret_sum) > 0
    assert len(trainer.strategy_sum) > 0
