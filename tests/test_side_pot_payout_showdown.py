from __future__ import annotations

from dataclasses import replace

from poker_ai.engine.cards import Card, Rank, Suit
from poker_ai.engine.dealing import Deal
from poker_ai.engine.payout import settle_terminal_hand
from poker_ai.engine.state import GameConfig, initial_state
from poker_ai.engine.types import Chips, Street


def test_side_pot_payout_showdown() -> None:
    cfg = GameConfig(
        num_players=3, small_blind=Chips(1), big_blind=Chips(2), starting_stack=Chips(100)
    )

    # Board: Ah Kh Qh Jh 2c
    board = (
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.TWO, Suit.CLUBS),
    )

    # P0: Th 9h => straight flush (A K Q J T hearts)
    # P1: 2d 2s => trips twos (since 2c on board)
    # P2: 3d 4s => nothing strong
    hole = (
        (Card(Rank.TEN, Suit.HEARTS), Card(Rank.NINE, Suit.HEARTS)),
        (Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.TWO, Suit.SPADES)),
        (Card(Rank.THREE, Suit.DIAMONDS), Card(Rank.FOUR, Suit.SPADES)),
    )
    deal = Deal(hole_cards=hole, board=board)

    s = initial_state(cfg, deal=deal, dealer_pos=0)

    # Force terminal showdown state with contributions:
    # P0 contributed 10, P1 30, P2 30
    # => main pot 30 among all, side pot 40 among P1,P2
    s = replace(
        s,
        street=Street.SHOWDOWN,
        revealed_board_cards=5,
        active=(True, True, True),
        stacks=(Chips(90), Chips(70), Chips(70)),  # after putting money in (for test)
        total_commit=(Chips(10), Chips(30), Chips(30)),
        street_commit=(Chips(0), Chips(0), Chips(0)),
        pot=Chips(0),
    )

    paid, main_winners = settle_terminal_hand(s)

    # Main winners should be player 0
    assert main_winners == (0,)

    # P0 gets main pot 30 => 90+30 = 120
    # Side pot 40 between P1,P2; P1 wins => 70+40 = 110; P2 stays 70
    assert int(paid.stacks[0]) == 120
    assert int(paid.stacks[1]) == 110
    assert int(paid.stacks[2]) == 70

    # commits cleared
    assert sum(int(x) for x in paid.total_commit) == 0
    assert sum(int(x) for x in paid.street_commit) == 0
