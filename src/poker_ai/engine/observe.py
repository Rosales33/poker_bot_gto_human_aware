from __future__ import annotations

from poker_ai.engine.observation import Observation
from poker_ai.engine.state import GameState


def observe(state: GameState, seat: int) -> Observation:
    if not (0 <= seat < state.n):
        raise ValueError("Invalid seat")

    revealed_n = state.revealed_board_cards
    board = state.board[:revealed_n]

    return Observation(
        seat=seat,
        street=state.street,
        pot=state.pot,
        current_bet=state.current_bet,
        last_raise=state.last_raise,
        active=state.active,
        stacks=state.stacks,
        street_commit=state.street_commit,
        history=state.history,
        board=board,
        hole_cards=state.hole_cards[seat],
    )
