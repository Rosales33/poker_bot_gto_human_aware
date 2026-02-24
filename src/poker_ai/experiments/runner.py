from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from poker_ai.agent.interfaces import Agent, AgentContext
from poker_ai.engine.dealing import deal_nlhe
from poker_ai.engine.observe import observe
from poker_ai.engine.payout import settle_terminal_hand
from poker_ai.engine.rules import apply_action
from poker_ai.engine.rules import legal_actions as legal_actions_fn
from poker_ai.engine.showdown import winners
from poker_ai.engine.state import GameConfig, GameState, initial_state
from poker_ai.engine.types import Chips, Street


@dataclass(frozen=True, slots=True)
class HandResult:
    final_state: GameState
    num_actions: int
    winners: tuple[int, ...]
    pot_paid: Chips


def _pot_total(state: GameState) -> int:
    return int(state.pot) + sum(int(x) for x in state.street_commit)


def _active_seats(state: GameState) -> list[int]:
    return [i for i, a in enumerate(state.active) if a]


def _pay_winners(state: GameState, winner_seats: Sequence[int]) -> GameState:
    """
    Winner-take-all / split pot equally.
    v0.8: ignores side pots. This is OK for now; we’ll add side pots in 0.9.
    """
    pot = _pot_total(state)
    if pot < 0:
        raise RuntimeError("Negative pot total.")
    if not winner_seats:
        return state

    stacks = list(state.stacks)
    share = pot // len(winner_seats)
    remainder = pot - share * len(winner_seats)

    # Pay shares
    for w in winner_seats:
        stacks[w] = Chips(int(stacks[w]) + share)

    # Assign remainder deterministically to lowest seat indexes
    for w in sorted(winner_seats)[:remainder]:
        stacks[w] = Chips(int(stacks[w]) + 1)

    # After paying, pot + street commits are considered cleared.
    return replace(
        state,
        stacks=tuple(stacks),
        pot=Chips(0),
        street_commit=tuple(Chips(0) for _ in range(state.n)),
    )


def play_hand(
    cfg: GameConfig,
    agents: Sequence[Agent],
    dealer_pos: int,
    hand_id: int,
    base_seed: int = 0,
) -> HandResult:
    if len(agents) != cfg.num_players:
        raise ValueError("Need one agent per seat.")

    deal = deal_nlhe(cfg.num_players, hand_id=hand_id, base_seed=base_seed)
    state = initial_state(cfg, deal=deal, dealer_pos=dealer_pos)
    state = replace(state, hand_id=hand_id)

    while not state.is_terminal():
        seat = state.to_act
        acts = legal_actions_fn(state)
        obs = observe(state, seat)
        action = agents[seat].choose_action(obs, acts, AgentContext(seat=seat))
        state = apply_action(state, action)

        if len(state.history) > 10_000:
            raise RuntimeError("Too many actions in one hand; possible rules bug.")

    # Determine winners
    active = _active_seats(state)
    if len(active) == 1:
        win = tuple(active)
        state_paid = _pay_winners(state, win)
        return HandResult(
            final_state=state_paid,
            num_actions=len(state.history),
            winners=win,
            pot_paid=Chips(_pot_total(state)),
        )

    # Showdown (street == SHOWDOWN)
    if state.street != Street.SHOWDOWN:
        # Terminal by fold would have returned earlier; if we get here, it’s a logic bug.
        raise RuntimeError("Terminal state without single winner or showdown.")

    win_list = winners(state.hole_cards, state.board, state.active)
    win = tuple(win_list)
    pot_paid = Chips(
        sum(int(x) for x in state.total_commit) + sum(int(x) for x in state.street_commit)
    )
    state_paid, main_winners = settle_terminal_hand(state)
    return HandResult(
        final_state=state_paid,
        num_actions=len(state.history),
        winners=tuple(main_winners),
        pot_paid=pot_paid,
    )


def play_match(
    cfg: GameConfig,
    agents: Sequence[Agent],
    num_hands: int,
    dealer_start: int = 0,
    base_seed: int = 0,
) -> list[HandResult]:
    results: list[HandResult] = []
    dealer = dealer_start
    for hand_id in range(num_hands):
        results.append(
            play_hand(cfg, agents, dealer_pos=dealer, hand_id=hand_id, base_seed=base_seed)
        )
        dealer = (dealer + 1) % cfg.num_players
    return results
