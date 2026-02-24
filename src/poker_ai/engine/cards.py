from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass(frozen=True, slots=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        rank_map = {
            Rank.TWO: "2",
            Rank.THREE: "3",
            Rank.FOUR: "4",
            Rank.FIVE: "5",
            Rank.SIX: "6",
            Rank.SEVEN: "7",
            Rank.EIGHT: "8",
            Rank.NINE: "9",
            Rank.TEN: "T",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }
        suit_map = {Suit.CLUBS: "c", Suit.DIAMONDS: "d", Suit.HEARTS: "h", Suit.SPADES: "s"}
        return f"{rank_map[self.rank]}{suit_map[self.suit]}"


def full_deck() -> list[Card]:
    return [Card(rank, suit) for suit in Suit for rank in Rank]


def cards_to_str(cards: Iterable[Card]) -> str:
    return " ".join(str(c) for c in cards)
