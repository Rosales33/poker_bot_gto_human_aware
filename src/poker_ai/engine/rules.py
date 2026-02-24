from __future__ import annotations

from dataclasses import replace

from poker_ai.engine.history import ActionRecord
from poker_ai.engine.state import GameState
from poker_ai.engine.types import Action, ActionType, Chips, Street


def _next_active_player(state: GameState, start_from: int) -> int:
    n = state.n
    i = (start_from + 1) % n
    for _ in range(n):
        if state.active[i]:
            return i
        i = (i + 1) % n
    return start_from


def _num_not_folded(state: GameState) -> int:
    return sum(1 for a in state.active if a)


def _everyone_matched_or_allin(state: GameState) -> bool:
    for i in range(state.n):
        if not state.active[i]:
            continue
        if int(state.stacks[i]) == 0:
            continue
        if int(state.current_bet) == 0:
            continue
        if int(state.street_commit[i]) != int(state.current_bet):
            return False
    return True


def _betting_round_complete(state: GameState, next_to_act: int) -> bool:
    """
    If no one has bet/raised this street (last_aggressor is None):
      - round completes when action returns to first_to_act (everyone checked).
    If there was aggression:
      - round completes when action returns to last_aggressor AND everyone has matched/all-in.
    """
    if _num_not_folded(state) <= 1:
        return True

    if state.last_aggressor is None:
        return next_to_act == state.first_to_act
    return next_to_act == state.last_aggressor and _everyone_matched_or_allin(state)


def _advance_street(state: GameState) -> GameState:
    pot = Chips(int(state.pot) + sum(int(x) for x in state.street_commit))
    reset_commit = tuple(Chips(0) for _ in range(state.n))

    if state.street == Street.PREFLOP:
        next_street = Street.FLOP
    elif state.street == Street.FLOP:
        next_street = Street.TURN
    elif state.street == Street.TURN:
        next_street = Street.RIVER
    elif state.street == Street.RIVER:
        next_street = Street.SHOWDOWN
    else:
        next_street = Street.SHOWDOWN

    if next_street == Street.FLOP:
        revealed = 3
    elif next_street == Street.TURN:
        revealed = 4
    elif next_street == Street.RIVER:
        revealed = 5
    else:
        revealed = state.revealed_board_cards

    # Postflop first to act:
    # - HU: BB (non-dealer) acts first postflop
    # - 3+ players: player left of dealer acts first
    if state.n == 2:
        first = (state.dealer_pos + 1) % 2
    else:
        first = (state.dealer_pos + 1) % state.n

    # ensure first is active (skip folds)
    first_to_act = first
    if not state.active[first_to_act]:
        first_to_act = _next_active_player(state, first_to_act)

    return replace(
        state,
        street=next_street,
        to_act=first_to_act,
        pot=pot,
        street_commit=reset_commit,
        current_bet=Chips(0),
        last_raise=Chips(0),
        first_to_act=first_to_act,
        last_aggressor=None,
        revealed_board_cards=revealed,
    )


def legal_actions(state: GameState) -> list[Action]:
    if state.is_terminal():
        return []

    i = state.to_act
    if not state.active[i]:
        return []

    stack = int(state.stacks[i])
    committed = int(state.street_commit[i])
    to_call = max(0, int(state.current_bet) - committed)

    actions: list[Action] = []

    if to_call > 0:
        actions.append(Action(ActionType.FOLD, Chips(0)))

    if to_call == 0:
        actions.append(Action(ActionType.CHECK, Chips(0)))
    else:
        actions.append(Action(ActionType.CALL, Chips(min(to_call, stack))))

    # simple sizing v0 (we’ll replace with abstraction module later)
    if stack > to_call:
        if int(state.current_bet) == 0:
            pot_now = int(state.pot) + sum(int(x) for x in state.street_commit)
            half = max(1, pot_now // 2)
            potbet = max(1, pot_now)
            for amt in {half, potbet}:
                actions.append(Action(ActionType.BET, Chips(min(amt, stack))))
        else:
            min_raise_inc = max(1, int(state.last_raise))
            min_total_to = int(state.current_bet) + min_raise_inc
            min_raise_amt = max(0, min_total_to - committed)
            if min_raise_amt <= stack:
                actions.append(Action(ActionType.RAISE, Chips(min_raise_amt)))
            actions.append(Action(ActionType.RAISE, Chips(stack)))  # all-in

    uniq: dict[tuple[ActionType, int], Action] = {}
    for a in actions:
        uniq[(a.type, int(a.amount))] = a
    return list(uniq.values())


def apply_action(state: GameState, action: Action) -> GameState:
    if action not in legal_actions(state):
        raise ValueError(f"Illegal action: {action}")

    if state.is_terminal():
        return state

    i = state.to_act
    stack = int(state.stacks[i])
    committed = int(state.street_commit[i])

    active = list(state.active)
    stacks = list(state.stacks)
    street_commit = list(state.street_commit)

    current_bet = int(state.current_bet)
    last_raise = int(state.last_raise)
    last_aggressor = state.last_aggressor

    def pay(amt: int) -> None:
        nonlocal stack, committed
        pay_amt = min(amt, stack)
        stack -= pay_amt
        committed += pay_amt
        stacks[i] = Chips(stack)
        street_commit[i] = Chips(committed)

    if action.type == ActionType.FOLD:
        active[i] = False

    elif action.type == ActionType.CHECK:
        pass

    elif action.type == ActionType.CALL:
        pay(int(action.amount))

    elif action.type == ActionType.BET:
        bet_amt = int(action.amount)
        pay(bet_amt)
        current_bet = committed
        last_raise = bet_amt
        last_aggressor = i

    elif action.type == ActionType.RAISE:
        raise_amt = int(action.amount)
        old_bet = current_bet
        pay(raise_amt)
        new_bet = committed
        raise_size = new_bet - old_bet
        if raise_size > 0:
            last_raise = raise_size
        current_bet = max(current_bet, new_bet)
        if raise_size > 0:
            last_aggressor = i

    else:
        raise ValueError("Unknown action type")

    record = ActionRecord(
        street=state.street,
        player=i,
        action_type=action.type,
        amount=action.amount,
    )
    s2 = replace(
        state,
        active=tuple(active),
        stacks=tuple(stacks),
        street_commit=tuple(street_commit),
        current_bet=Chips(current_bet),
        last_raise=Chips(last_raise),
        last_aggressor=last_aggressor,
        history=state.history + (record,),
    )

    if _num_not_folded(s2) <= 1:
        return s2

    next_to_act = _next_active_player(s2, s2.to_act)

    if _betting_round_complete(s2, next_to_act):
        return _advance_street(s2)

    return replace(s2, to_act=next_to_act)
