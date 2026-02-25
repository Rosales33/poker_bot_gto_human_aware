from __future__ import annotations

import argparse

from poker_ai.agent.blueprint_agent import BlueprintPolicy
from poker_ai.agent.random_legal import RandomLegalAgent
from poker_ai.agent.random_policy import RandomPolicy
from poker_ai.experiments.runner import play_match_bankroll
from poker_ai.solver.strategy_store import load_strategy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blueprint", type=str, required=True)
    p.add_argument("--hands", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    stored = load_strategy(args.blueprint)
    cfg = stored.game_cfg
    if cfg.num_players != 2:
        raise ValueError("This evaluator is for HU (2 players).")

    policies = [
        BlueprintPolicy(stored=stored, seed=args.seed),
        RandomPolicy(RandomLegalAgent(seed=12345)),
    ]

    traj = play_match_bankroll(
        cfg, policies, num_hands=args.hands, dealer_start=0, base_seed=args.seed
    )

    start0 = int(cfg.starting_stack)
    end0 = int(traj[-1][1][0])
    delta = end0 - start0
    bb = int(cfg.big_blind)
    bb100 = (delta / bb) / (args.hands / 100.0)

    print(f"Hands: {args.hands}")
    print(f"Seat0 delta: {delta} chips")
    print(f"BB/100: {bb100:.2f}")


if __name__ == "__main__":
    main()
