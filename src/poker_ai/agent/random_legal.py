from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from poker_ai.agent.interfaces import Agent, AgentContext
from poker_ai.engine.observation import Observation
from poker_ai.engine.types import Action


def _mix_seed(seed: int, hand_id: int, to_act: int, history_len: int) -> int:
    x = seed & 0xFFFFFFFF
    x = (x * 1000003) ^ (hand_id & 0xFFFFFFFF)
    x = (x * 1000003) ^ (to_act & 0xFFFFFFFF)
    x = (x * 1000003) ^ (history_len & 0xFFFFFFFF)
    return x & 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True, slots=True)
class RandomLegalAgent(Agent):
    seed: int = 0

    def choose_action(
        self, obs: Observation, legal_actions: Sequence[Action], ctx: AgentContext
    ) -> Action:
        if not legal_actions:
            raise RuntimeError("No legal actions passed to agent.")
        # Use obs.history length for determinism; hand_id comes from ctx later (v1),
        # for now we derive from history + seat (still deterministic enough)
        s = _mix_seed(self.seed, history_len=len(obs.history), hand_id=0, to_act=obs.seat)
        rng = random.Random(s)
        return rng.choice(list(legal_actions))
