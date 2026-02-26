from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from poker_ai.agent.interfaces import Agent, AgentContext
from poker_ai.engine.observation import Observation
from poker_ai.engine.types import Action


def _mix_seed(
    seed: int, hand_id: int, seat: int, history_len: int, pot: int, current_bet: int
) -> int:
    x = seed & 0xFFFFFFFF
    x = (x * 1000003) ^ (hand_id & 0xFFFFFFFF)
    x = (x * 1000003) ^ (seat & 0xFFFFFFFF)
    x = (x * 1000003) ^ (history_len & 0xFFFFFFFF)
    x = (x * 1000003) ^ (pot & 0xFFFFFFFF)
    x = (x * 1000003) ^ (current_bet & 0xFFFFFFFF)
    return x & 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True, slots=True)
class RandomLegalAgent(Agent):
    seed: int = 0

    def choose_action(
        self, obs: Observation, legal_actions: Sequence[Action], ctx: AgentContext
    ) -> Action:
        if not legal_actions:
            raise RuntimeError("No legal actions passed to agent.")

        # Deterministic but state-dependent seed to avoid identical trajectories
        s = _mix_seed(
            seed=self.seed,
            hand_id=obs.hand_id,
            seat=obs.seat,
            history_len=len(obs.history),
            pot=int(obs.pot),
            current_bet=int(obs.current_bet),
        )
        rng = random.Random(s)
        return rng.choice(list(legal_actions))
