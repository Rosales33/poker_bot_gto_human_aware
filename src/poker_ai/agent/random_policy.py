from __future__ import annotations

from dataclasses import dataclass

from poker_ai.agent.interfaces import AgentContext
from poker_ai.agent.policy import Policy
from poker_ai.agent.random_legal import RandomLegalAgent
from poker_ai.engine.observation import Observation
from poker_ai.engine.rules import legal_actions as legal_actions_fn
from poker_ai.engine.state import GameState
from poker_ai.engine.types import Action


@dataclass(frozen=True, slots=True)
class RandomPolicy(Policy):
    agent: RandomLegalAgent

    def act(self, state: GameState, obs: Observation) -> Action:
        acts = legal_actions_fn(state)
        return self.agent.choose_action(obs, acts, AgentContext(seat=obs.seat))
