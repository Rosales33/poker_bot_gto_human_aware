from __future__ import annotations

import argparse

from poker_ai.agent.blueprint_agent import BlueprintPolicy
from poker_ai.agent.policy import Policy
from poker_ai.agent.random_legal import RandomLegalAgent
from poker_ai.agent.random_policy import RandomPolicy
from poker_ai.engine.types import Chips
from poker_ai.experiments.runner import play_match
from poker_ai.solver.strategy_store import load_strategy


def _total_chips(stacks: tuple[Chips, ...]) -> int:
    return sum(int(s) for s in stacks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blueprint", type=str, required=True)
    p.add_argument("--hands", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    stored = load_strategy(args.blueprint)
    cfg = stored.game_cfg

    # Seat 0: blueprint, others: random
    policies: list[Policy] = []
    policies.append(BlueprintPolicy(stored=stored, seed=args.seed))
    for i in range(1, cfg.num_players):
        policies.append(RandomPolicy(RandomLegalAgent(seed=1000 + i)))

    results = play_match(cfg, policies, num_hands=args.hands, dealer_start=0, base_seed=args.seed)

    # Evaluate by final stacks delta aggregated across hands (simple)
    start_total = cfg.num_players * int(cfg.starting_stack)
    # We don't track per-hand bankroll evolution yet, so we approximate by summing final states deltas.
    # Better: extend runner to carry bankroll across hands (Phase 1.6).
    # For now, check we don't break conservation.
    assert all(_total_chips(r.final_state.stacks) == start_total for r in results)

    print(f"Ran {len(results)} hands. (Note: bankroll reset each hand in current runner.)")
    print("Next step: add match bankroll carry-over for proper BB/100.")


if __name__ == "__main__":
    main()
