from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from poker_ai.engine.showdown import winners_among
from poker_ai.engine.state import GameState
from poker_ai.engine.types import Chips, Street


@dataclass(frozen=True, slots=True)
class SidePot:
    amount: Chips
    eligible: tuple[int, ...]  # seats that can win this pot (not folded)


def _final_contrib(state: GameState) -> list[int]:
    return [int(state.total_commit[i]) + int(state.street_commit[i]) for i in range(state.n)]


def compute_side_pots(contrib: Sequence[int], active: Sequence[bool]) -> list[SidePot]:
    """
    Build side pots from final contributions.
    active[i] indicates player not folded (eligible for showdown winnings).
    """
    levels = sorted({c for c in contrib if c > 0})
    if not levels:
        return []

    pots: list[SidePot] = []
    prev = 0
    for lvl in levels:
        participants = [i for i, c in enumerate(contrib) if c >= lvl]
        tranche = (lvl - prev) * len(participants)
        if tranche <= 0:
            prev = lvl
            continue

        eligible = tuple(i for i in participants if active[i])
        # Even if eligible is empty (shouldn't happen in normal play), we keep pot for conservation.
        pots.append(SidePot(amount=Chips(tranche), eligible=eligible))
        prev = lvl

    return pots


def _pay_split(stacks: list[Chips], seats: Sequence[int], amount: int) -> None:
    if amount <= 0 or not seats:
        return
    share = amount // len(seats)
    rem = amount - share * len(seats)
    for s in seats:
        stacks[s] = Chips(int(stacks[s]) + share)
    # deterministic remainder: lowest seats
    for s in sorted(seats)[:rem]:
        stacks[s] = Chips(int(stacks[s]) + 1)


def settle_terminal_hand(state: GameState) -> tuple[GameState, tuple[int, ...]]:
    """
    Settle a terminal hand with correct side pots.
    Returns (updated_state, winners_of_main_pot_or_fold).
    For fold case: single winner.
    For showdown: winners of main pot (first side pot).
    """
    if not state.is_terminal():
        raise ValueError("Hand is not terminal.")

    contrib = _final_contrib(state)
    pots = compute_side_pots(contrib, state.active)

    stacks = list(state.stacks)

    # Fold case: only one active remains => that player wins every pot
    active_seats = [i for i, a in enumerate(state.active) if a]
    if len(active_seats) == 1:
        winner = active_seats[0]
        total = sum(int(p.amount) for p in pots)
        _pay_split(stacks, [winner], total)
        cleared = _clear_commits(state)
        return replace(cleared, stacks=tuple(stacks)), (winner,)

    # Showdown case
    if state.street != Street.SHOWDOWN:
        raise RuntimeError("Terminal state is neither fold nor showdown.")

    main_winners: tuple[int, ...] = tuple()
    for idx, pot in enumerate(pots):
        w: tuple[int, ...]
        if int(pot.amount) == 0:
            continue
        if len(pot.eligible) == 1:
            w = (pot.eligible[0],)
        else:
            w = tuple(winners_among(pot.eligible, state.hole_cards, state.board))

        if idx == 0:
            main_winners = w

        _pay_split(stacks, w, int(pot.amount))

    cleared = _clear_commits(state)
    return replace(cleared, stacks=tuple(stacks)), main_winners


def _clear_commits(state: GameState) -> GameState:
    return replace(
        state,
        pot=Chips(0),
        street_commit=tuple(Chips(0) for _ in range(state.n)),
        total_commit=tuple(Chips(0) for _ in range(state.n)),
    )
