from __future__ import annotations

from typing import Protocol

from poker_ai.engine.observation import Observation
from poker_ai.engine.state import GameState
from poker_ai.engine.types import Action


class Policy(Protocol):
    def act(self, state: GameState, obs: Observation) -> Action: ...
