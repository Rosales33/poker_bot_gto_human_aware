from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from poker_ai.engine.rules import legal_actions
from poker_ai.engine.state import GameState
from poker_ai.engine.types import Action, ActionType, Chips


class AbstractActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class AbstractAction:
    type: AbstractActionType
    # For BET/RAISE: parameter is either a pot fraction (bet) or a raise multiplier (raise)
    # bet: fraction of pot (e.g., 0.33, 0.75)
    # raise: multiplier of the current bet to call (e.g., 2.5, 4.0)
    param: float | None = None

    def __str__(self) -> str:
        if self.param is None:
            return self.type.value
        return f"{self.type.value}({self.param})"


@dataclass(frozen=True, slots=True)
class ActionAbstractionConfig:
    bet_fractions: tuple[float, ...] = (0.33, 0.75)
    raise_multipliers: tuple[float, ...] = (2.5, 4.0)
    include_all_in: bool = True


def _pot_now(state: GameState) -> int:
    # Pot visible to betting includes committed money too
    return (
        int(state.pot)
        + sum(int(x) for x in state.total_commit)
        + sum(int(x) for x in state.street_commit)
    )


def abstract_to_action(state: GameState, abs_action: AbstractAction) -> Action:
    i = state.to_act
    committed = int(state.street_commit[i])
    to_call = max(0, int(state.current_bet) - committed)
    stack = int(state.stacks[i])

    if abs_action.type == AbstractActionType.FOLD:
        return Action(ActionType.FOLD, Chips(0))
    if abs_action.type == AbstractActionType.CHECK:
        return Action(ActionType.CHECK, Chips(0))
    if abs_action.type == AbstractActionType.CALL:
        return Action(ActionType.CALL, Chips(min(to_call, stack)))

    if abs_action.type == AbstractActionType.BET:
        if abs_action.param is None:
            raise ValueError("BET requires param")
        if int(state.current_bet) != 0:
            raise ValueError("BET not valid when facing a bet; use RAISE")
        pot_now = _pot_now(state)
        amt = max(1, int(round(abs_action.param * pot_now)))
        amt = min(amt, stack)
        return Action(ActionType.BET, Chips(amt))

    if abs_action.type == AbstractActionType.RAISE:
        if abs_action.param is None:
            raise ValueError("RAISE requires param")
        if int(state.current_bet) == 0:
            raise ValueError("RAISE not valid when no bet; use BET")
        # Simple NL mapping: raise_to = current_bet * multiplier
        raise_to = int(round(int(state.current_bet) * abs_action.param))
        raise_to = min(raise_to, committed + stack)  # cannot exceed all-in
        inc = max(0, raise_to - committed)
        return Action(ActionType.RAISE, Chips(inc))

    raise ValueError("Unknown abstract action type")


def get_abstract_actions(state: GameState, cfg: ActionAbstractionConfig) -> list[AbstractAction]:
    """
    Returns abstract actions that map to currently legal engine actions.
    """
    legals = legal_actions(state)
    legal_set = {(a.type, int(a.amount)) for a in legals}

    abs_actions: list[AbstractAction] = []

    # always consider fold/check/call (they'll be filtered by legality)
    candidates = [
        AbstractAction(AbstractActionType.FOLD),
        AbstractAction(AbstractActionType.CHECK),
        AbstractAction(AbstractActionType.CALL),
    ]

    if int(state.current_bet) == 0:
        for f in cfg.bet_fractions:
            candidates.append(AbstractAction(AbstractActionType.BET, f))
        if cfg.include_all_in:
            candidates.append(AbstractAction(AbstractActionType.BET, 10_000.0))  # will cap to stack
    else:
        for m in cfg.raise_multipliers:
            candidates.append(AbstractAction(AbstractActionType.RAISE, m))
        if cfg.include_all_in:
            candidates.append(AbstractAction(AbstractActionType.RAISE, 10_000.0))

    # filter by legal mapping
    for cand in candidates:
        try:
            eng_action = abstract_to_action(state, cand)
        except ValueError:
            continue
        if (eng_action.type, int(eng_action.amount)) in legal_set:
            abs_actions.append(cand)

    # de-duplicate
    uniq: dict[tuple[AbstractActionType, float | None], AbstractAction] = {}
    for aa in abs_actions:
        uniq[(aa.type, aa.param)] = aa
    return list(uniq.values())
