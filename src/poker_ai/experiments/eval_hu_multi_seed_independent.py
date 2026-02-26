# src/poker_ai/experiments/eval_hu_multi_seed_independent.py
from __future__ import annotations

import argparse
import math
from statistics import mean, pstdev

from poker_ai.agent.blueprint_agent import BlueprintPolicy
from poker_ai.agent.random_legal import RandomLegalAgent
from poker_ai.agent.random_policy import RandomPolicy
from poker_ai.experiments.runner import play_hand
from poker_ai.solver.strategy_store import load_strategy


def _bb100(total_delta_chips: int, bb: int, hands: int) -> float:
    return (total_delta_chips / bb) / (hands / 100.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blueprint", type=str, required=True)
    p.add_argument("--hands", type=int, default=10000)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--start-seed", type=int, default=0)
    args = p.parse_args()

    stored = load_strategy(args.blueprint)
    cfg = stored.game_cfg
    if cfg.num_players != 2:
        raise ValueError("HU only")

    bb = int(cfg.big_blind)
    results: list[float] = []

    for s in range(args.start_seed, args.start_seed + args.seeds):
        policies = [
            BlueprintPolicy(stored=stored, seed=s),
            RandomPolicy(RandomLegalAgent(seed=10_000 + s)),
        ]

        total_delta = 0
        dealer = 0
        for hand_id in range(args.hands):
            r = play_hand(cfg, policies, dealer_pos=dealer, hand_id=hand_id, base_seed=s)
            total_delta += int(r.final_state.stacks[0]) - int(cfg.starting_stack)
            dealer = (dealer + 1) % cfg.num_players

        results.append(_bb100(total_delta, bb, args.hands))

    m = mean(results)
    if len(results) > 1:
        sd = pstdev(results) * math.sqrt(len(results) / (len(results) - 1))
        se = sd / math.sqrt(len(results))
        tcrit = 2.26 if len(results) <= 10 else 2.09
        lo, hi = m - tcrit * se, m + tcrit * se
    else:
        sd = 0.0
        lo = hi = m

    print(f"Seeds: {args.seeds} | Hands/seed: {args.hands}")
    print(f"BB/100 mean: {m:.3f} | std: {sd:.3f} | ~95% CI: [{lo:.3f}, {hi:.3f}]")
    print("Per-seed:", ", ".join(f"{x:.3f}" for x in results))


if __name__ == "__main__":
    main()
