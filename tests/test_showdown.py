from __future__ import annotations

from poker_ai.engine.cards import Card, Rank, Suit
from poker_ai.engine.showdown import winners


def test_showdown_winner_simple() -> None:
    # Board: Ah Kh Qh Jh 2c
    board = (
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.TWO, Suit.CLUBS),
    )

    # P0 has Th 9h -> makes royal-ish straight flush (A K Q J T hearts)
    # P1 has 2d 2s -> just a set of twos? actually board has 2c so it’s trips twos.
    hole_cards = (
        (Card(Rank.TEN, Suit.HEARTS), Card(Rank.NINE, Suit.HEARTS)),
        (Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.TWO, Suit.SPADES)),
    )
    active = (True, True)
    win = winners(hole_cards, board, active)
    assert win == [0]
