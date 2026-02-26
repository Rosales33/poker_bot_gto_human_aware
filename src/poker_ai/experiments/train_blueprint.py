from __future__ import annotations

import argparse
from pathlib import Path

from poker_ai.abstraction.action_abstraction import ActionAbstractionConfig
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
    p.add_argument("--bet-fracs", type=str, default="0.33,0.75")
    p.add_argument("--raise-mults", type=str, default="2.5,4.0")
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--checkpoint-prefix", type=str, default="data/blueprints/ckpt")
    args = p.parse_args()

    cfg = GameConfig(num_players=args.players, starting_stack=Chips(200))
    bet_fracs = tuple(float(x.strip()) for x in args.bet_fracs.split(",") if x.strip())
    raise_mults = tuple(float(x.strip()) for x in args.raise_mults.split(",") if x.strip())

    mccfg = MCCFRConfig(
        iterations=args.iters,
        seed=args.seed,
        base_deal_seed=args.deal_seed,
        action_abs=ActionAbstractionConfig(
            bet_fractions=bet_fracs,
            raise_multipliers=raise_mults,
            include_all_in=True,
        ),
    )
    trainer = MCCFRTrainer(
        cfg,
        mccfg,
    )
    remaining = args.iters
    chunk = (
        args.checkpoint_every if args.checkpoint_every and args.checkpoint_every > 0 else args.iters
    )
    done = 0

    while remaining > 0:
        step = min(chunk, remaining)
        trainer.run(iterations=step)
        done += step
        remaining -= step

        stored = StoredStrategy(
            game_cfg=cfg,
            action_abs_cfg=trainer.cfg.action_abs,
            action_lists=trainer.action_lists,
            strategy_sum=trainer.strategy_sum,
        )

        # save checkpoint or final
        out_path = (
            Path(args.out) if remaining == 0 else Path(f"{args.checkpoint_prefix}_{done}.pkl")
        )
        save_strategy(out_path, stored)
        print(f"Saved {'final' if remaining==0 else 'checkpoint'} to {out_path}")

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
