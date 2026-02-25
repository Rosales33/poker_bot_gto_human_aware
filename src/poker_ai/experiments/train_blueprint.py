from __future__ import annotations

import argparse
from pathlib import Path

from poker_ai.engine.state import GameConfig
from poker_ai.engine.types import Chips
from poker_ai.solver.mccfr import MCCFRConfig, MCCFRTrainer
from poker_ai.solver.strategy_store import StoredStrategy, save_strategy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--iters", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deal-seed", type=int, default=1)
    p.add_argument("--out", type=str, default="data/blueprints/blueprint.pkl")
    args = p.parse_args()

    cfg = GameConfig(num_players=args.players, starting_stack=Chips(200))
    trainer = MCCFRTrainer(
        cfg,
        MCCFRConfig(iterations=args.iters, seed=args.seed, base_deal_seed=args.deal_seed),
    )
    trainer.run()

    stored = StoredStrategy(
        game_cfg=cfg,
        action_abs_cfg=trainer.cfg.action_abs,
        action_lists=trainer.action_lists,
        strategy_sum=trainer.strategy_sum,
    )

    out = Path(args.out)
    save_strategy(out, stored)
    print(f"Saved blueprint to {out}")


if __name__ == "__main__":
    main()
