from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from poker_ai.engine.observation import Observation
from poker_ai.engine.types import Action


@dataclass(frozen=True, slots=True)
class AgentContext:
    seat: int


class Agent(Protocol):
    def choose_action(
        self, obs: Observation, legal_actions: Sequence[Action], ctx: AgentContext
    ) -> Action: ...
