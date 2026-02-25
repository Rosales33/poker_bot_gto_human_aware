from __future__ import annotations

from dataclasses import dataclass

from poker_ai.engine.cards import Card
from poker_ai.engine.history import ActionRecord
from poker_ai.engine.types import Chips, Street


@dataclass(frozen=True, slots=True)
class Observation:
    seat: int
    dealer_pos: int
    num_players: int

    street: Street
    pot: Chips
    current_bet: Chips
    last_raise: Chips
    active: tuple[bool, ...]
    stacks: tuple[Chips, ...]
    street_commit: tuple[Chips, ...]
    history: tuple[ActionRecord, ...]

    board: tuple[Card, ...]  # only revealed portion
    hole_cards: tuple[Card, Card]  # only for this seat
