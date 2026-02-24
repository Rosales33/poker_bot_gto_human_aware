from __future__ import annotations

from dataclasses import dataclass

from poker_ai.engine.cards import Card
from poker_ai.engine.dealing import Deal
from poker_ai.engine.history import ActionRecord
from poker_ai.engine.types import Chips, Street


@dataclass(frozen=True, slots=True)
class GameConfig:
    num_players: int = 6
    small_blind: Chips = Chips(1)
    big_blind: Chips = Chips(2)
    starting_stack: Chips = Chips(200)  # 100bb if BB=2
    min_raise_multiplier: float = 2.0  # no-limit: min raise = last_raise_size (handled in rules)


@dataclass(frozen=True, slots=True)
class GameState:
    cfg: GameConfig
    street: Street
    dealer_pos: int  # index of dealer button
    to_act: int  # index of player whose turn it is
    # players still in hand (not folded). all-in players remain "active" but have stack=0
    active: tuple[bool, ...]
    stacks: tuple[Chips, ...]
    # amount each player has put into the pot for the current street (for calling)
    street_commit: tuple[Chips, ...]
    # total pot size (including street_commit already moved in previous streets)
    pot: Chips
    # current highest commitment on this street (i.e., bet to call)
    current_bet: Chips
    # last raise size on this street (used to enforce min-raise)
    last_raise: Chips
    # index of first player to act on this street (for end-of-round detection)
    first_to_act: int
    # index of last aggressor (bettor/raiser) on this street, for enforcing end-of-round rules
    last_aggressor: int | None  # player index who last bet/raised on this street
    # unique identifier for the hand (can be used to group states from the same hand together)
    hand_id: int
    # immutable history of actions leading to this state (for training data, etc)
    history: tuple[ActionRecord, ...]
    # hole cards for each player (empty cards for players who have folded or not yet dealt)
    hole_cards: tuple[tuple[Card, Card], ...]
    # the 5 board cards (only first N are "revealed" based on street)
    board: tuple[Card, ...]
    # number of board cards currently revealed (3 on flop, 4 on turn, 5 on river; 0 preflop)
    revealed_board_cards: int

    @property
    def n(self) -> int:
        return self.cfg.num_players

    def is_terminal(self) -> bool:
        # terminal if only one active (not folded)
        if sum(1 for a in self.active if a) <= 1:
            return True
        return self.street == Street.SHOWDOWN


def initial_state(cfg: GameConfig, deal: Deal, dealer_pos: int = 0) -> GameState:
    n = cfg.num_players
    if n < 2:
        raise ValueError("Need at least 2 players.")
    if not (0 <= dealer_pos < n):
        raise ValueError("Invalid dealer_pos.")

    stacks = [cfg.starting_stack] * n
    active = [True] * n
    street_commit = [Chips(0)] * n

    if n == 2:
        # Heads-up: dealer is SB, other is BB
        sb_pos = dealer_pos
        bb_pos = (dealer_pos + 1) % n
    else:
        sb_pos = (dealer_pos + 1) % n
        bb_pos = (dealer_pos + 2) % n

    def post(pos: int, amt: Chips) -> None:
        pay = min(int(amt), int(stacks[pos]))
        stacks[pos] = Chips(int(stacks[pos]) - pay)
        street_commit[pos] = Chips(int(street_commit[pos]) + pay)

    post(sb_pos, cfg.small_blind)
    post(bb_pos, cfg.big_blind)

    pot = Chips(0)
    current_bet = cfg.big_blind
    last_raise = cfg.big_blind

    # Preflop first to act:
    # - HU: SB (dealer) acts first preflop
    # - 3+ players: UTG (left of BB)
    if n == 2:
        first_to_act = sb_pos
    else:
        first_to_act = (bb_pos + 1) % n

    to_act = first_to_act

    # Treat BB as last_aggressor initially (forced bet to call)
    last_aggressor = bb_pos

    return GameState(
        cfg=cfg,
        street=Street.PREFLOP,
        dealer_pos=dealer_pos,
        to_act=to_act,
        active=tuple(active),
        stacks=tuple(stacks),
        street_commit=tuple(street_commit),
        pot=pot,
        current_bet=current_bet,
        last_raise=last_raise,
        first_to_act=first_to_act,
        last_aggressor=last_aggressor,
        hand_id=0,
        history=tuple(),
        hole_cards=deal.hole_cards,
        board=deal.board,
        revealed_board_cards=0,
    )
