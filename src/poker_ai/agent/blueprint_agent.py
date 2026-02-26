from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from poker_ai.abstraction.action_abstraction import (
    AbstractAction,
    abstract_to_action,
    get_abstract_actions,
)
from poker_ai.abstraction.infoset import make_infoset_key
from poker_ai.agent.interfaces import Agent, AgentContext
from poker_ai.engine.observation import Observation
from poker_ai.engine.state import GameState
from poker_ai.engine.types import Action
from poker_ai.solver.strategy_store import StoredStrategy


def _normalize(probs: NDArray[np.float64]) -> NDArray[np.float64]:
    s = float(probs.sum())
    if s <= 1e-12:
        n = int(probs.size)
        return np.full((n,), 1.0 / n, dtype=np.float64)
    return probs / s


def _mix_seed(seed: int, hand_id: int, seat: int, history_len: int, key_hash: int) -> int:
    x = seed & 0xFFFFFFFF
    x = (x * 1000003) ^ (hand_id & 0xFFFFFFFF)
    x = (x * 1000003) ^ (seat & 0xFFFFFFFF)
    x = (x * 1000003) ^ (history_len & 0xFFFFFFFF)
    x = (x * 1000003) ^ (key_hash & 0xFFFFFFFF)
    return x & 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True, slots=True)
class BlueprintPolicy:
    stored: StoredStrategy
    seed: int = 0

    def act(self, state: GameState, obs: Observation) -> Action:
        """
        Choose an action from the blueprint (average strategy) given the current state.
        """
        bb = int(self.stored.game_cfg.big_blind)
        key = make_infoset_key(obs, bb=bb, history_len=8)

        legal_abs = get_abstract_actions(state, self.stored.action_abs_cfg)
        if not legal_abs:
            raise RuntimeError("No legal abstract actions available.")

        # Deterministic RNG that depends on hand + infoset + local history length
        rng = random.Random(
            _mix_seed(self.seed, obs.hand_id, obs.seat, len(obs.history), hash(key))
        )

        # If we never visited this infoset in training: play uniform over legal abstract actions
        if key not in self.stored.action_lists or key not in self.stored.strategy_sum:
            aa = rng.choice(list(legal_abs))
            return abstract_to_action(state, aa)

        action_list = self.stored.action_lists[key]
        strat_sum = self.stored.strategy_sum[key].astype(np.float64, copy=False)
        probs_full = _normalize(strat_sum)

        legal_set = {(a.type, a.param) for a in legal_abs}
        legal_idx = [i for i, a in enumerate(action_list) if (a.type, a.param) in legal_set]

        if not legal_idx:
            # fallback if mismatch between stored action list and current legal set
            aa = rng.choice(list(legal_abs))
            return abstract_to_action(state, aa)

        probs = np.array([probs_full[i] for i in legal_idx], dtype=np.float64)
        probs = _normalize(probs)

        x = rng.random()
        c = 0.0
        chosen_idx = legal_idx[-1]
        for p, idx in zip(probs, legal_idx, strict=True):
            c += float(p)
            if x <= c:
                chosen_idx = idx
                break

        chosen_aa: AbstractAction = action_list[chosen_idx]
        return abstract_to_action(state, chosen_aa)


@dataclass(frozen=True, slots=True)
class BlueprintAgent(Agent):
    policy: BlueprintPolicy

    def choose_action(
        self,
        obs: Observation,
        legal_actions: Sequence[Action],
        ctx: AgentContext,
    ) -> Action:
        # We need the full state to map abstract actions -> engine Action.
        # The runner will call policy.act(...) directly and pass the chosen action.
        raise RuntimeError("Use BlueprintPolicy.act(state, obs) from runner for BlueprintAgent.")
