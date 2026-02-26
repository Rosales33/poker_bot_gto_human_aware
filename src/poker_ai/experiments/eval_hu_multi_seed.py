from __future__ import annotations

import argparse
import math
from statistics import mean, pstdev

from poker_ai.agent.blueprint_agent import BlueprintPolicy
from poker_ai.agent.random_legal import RandomLegalAgent
from poker_ai.agent.random_policy import RandomPolicy
from poker_ai.experiments.runner import play_match_bankroll
from poker_ai.solver.strategy_store import load_strategy


def _bb100(delta_chips: int, bb: int, hands: int) -> float:
    return (delta_chips / bb) / (hands / 100.0)


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
        raise ValueError("This evaluator is for HU (2 players).")

    bb = int(cfg.big_blind)
    results: list[float] = []

    for s in range(args.start_seed, args.start_seed + args.seeds):
        policies = [
            BlueprintPolicy(stored=stored, seed=s),
            RandomPolicy(RandomLegalAgent(seed=10000 + s)),
        ]
        traj = play_match_bankroll(cfg, policies, num_hands=args.hands, dealer_start=0, base_seed=s)
        start0 = int(cfg.starting_stack)
        end0 = int(traj[-1][1][0])
        delta = end0 - start0
        results.append(_bb100(delta, bb, args.hands))

    m = mean(results)
    # sample std (use pstdev then adjust) — ok for quick CI
    if len(results) > 1:
        sd = pstdev(results) * math.sqrt(len(results) / (len(results) - 1))
        se = sd / math.sqrt(len(results))
        # t≈2.262 for df=9, 2.093 for df=19; use 2.26 as safe-ish default for ~10 seeds
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
