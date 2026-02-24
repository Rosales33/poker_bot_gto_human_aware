from __future__ import annotations

from dataclasses import dataclass

from poker_ai.engine.types import ActionType, Chips, Street


@dataclass(frozen=True, slots=True)
class ActionRecord:
    street: Street
    player: int
    action_type: ActionType
    amount: Chips
