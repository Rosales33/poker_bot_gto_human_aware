from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType


class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


Chips = NewType("Chips", int)


@dataclass(frozen=True, slots=True)
class Action:
    type: ActionType
    # amount is the total amount the acting player puts in THIS action (incremental),
    # not "raise to". For CHECK/FOLD amount must be 0.
    amount: Chips = Chips(0)
