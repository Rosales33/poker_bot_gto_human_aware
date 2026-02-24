from __future__ import annotations

import random
from dataclasses import dataclass

from poker_ai.engine.cards import Card, full_deck


@dataclass(frozen=True, slots=True)
class Deal:
    hole_cards: tuple[tuple[Card, Card], ...]  # per seat
    board: tuple[Card, ...]  # 0..5


def deal_nlhe(num_players: int, hand_id: int, base_seed: int = 0) -> Deal:
    if num_players < 2:
        raise ValueError("Need at least 2 players.")

    deck = full_deck()
    rng = random.Random((base_seed << 32) ^ hand_id)
    rng.shuffle(deck)

    # Deal 2 cards to each seat (round-robin like real dealing)
    hole: list[list[Card]] = [[] for _ in range(num_players)]
    idx = 0
    for _ in range(2):
        for p in range(num_players):
            hole[p].append(deck[idx])
            idx += 1

    hole_cards = tuple((hc[0], hc[1]) for hc in hole)

    # Deal 5 board cards (ignore burn for now; can add later)
    board = tuple(deck[idx : idx + 5])

    return Deal(hole_cards=hole_cards, board=board)
