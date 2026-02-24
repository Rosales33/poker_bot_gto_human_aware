from __future__ import annotations

from poker_ai.engine.dealing import deal_nlhe
from poker_ai.engine.history import ActionRecord
from poker_ai.engine.rules import apply_action, legal_actions
from poker_ai.engine.state import GameConfig, initial_state
from poker_ai.engine.types import Action, ActionType, Chips, Street


def test_initial_state_blinds_and_turn() -> None:
    cfg = GameConfig(
        num_players=6, small_blind=Chips(1), big_blind=Chips(2), starting_stack=Chips(200)
    )
    deal = deal_nlhe(cfg.num_players, hand_id=0, base_seed=0)
    s = initial_state(cfg, deal=deal, dealer_pos=0)

    # dealer=0 => SB=1, BB=2, UTG=3
    assert s.to_act == 3
    assert s.street == Street.PREFLOP
    assert int(s.street_commit[1]) == 1
    assert int(s.street_commit[2]) == 2
    assert int(s.stacks[1]) == 199
    assert int(s.stacks[2]) == 198


def test_legal_actions_include_call_and_raise_preflop() -> None:
    cfg = GameConfig()
    deal = deal_nlhe(cfg.num_players, hand_id=0, base_seed=0)
    s = initial_state(cfg, deal=deal, dealer_pos=0)
    acts = legal_actions(s)
    types = {a.type for a in acts}
    assert ActionType.CALL in types
    assert ActionType.RAISE in types
    # facing BB, fold should be allowed preflop
    assert ActionType.FOLD in types


def test_fold_ends_when_one_left() -> None:
    cfg = GameConfig(
        num_players=2, small_blind=Chips(1), big_blind=Chips(2), starting_stack=Chips(200)
    )
    deal = deal_nlhe(cfg.num_players, hand_id=0, base_seed=0)
    s = initial_state(cfg, deal=deal, dealer_pos=0)
    # HU: dealer=0 => SB=0 (dealer), BB=1, first to act preflop = SB = 0
    assert s.to_act == 0
    s2 = apply_action(s, Action(ActionType.FOLD))
    assert s2.is_terminal()
    assert sum(1 for a in s2.active if a) == 1


def test_check_check_advances_street_postflop() -> None:
    cfg = GameConfig(
        num_players=2, small_blind=Chips(1), big_blind=Chips(2), starting_stack=Chips(200)
    )
    deal = deal_nlhe(cfg.num_players, hand_id=0, base_seed=0)
    s = initial_state(cfg, deal=deal, dealer_pos=0)

    # Preflop: dealer=0 => SB=1, BB=0, first to act=1 (SB)
    # SB calls, BB checks -> should go to flop
    s = apply_action(s, Action(ActionType.CALL, Chips(1)))
    # BB can check (to_call=0)
    s = apply_action(s, Action(ActionType.CHECK, Chips(0)))
    assert s.street == Street.FLOP

    # flop: check-check advances to turn
    s = apply_action(s, Action(ActionType.CHECK, Chips(0)))
    s = apply_action(s, Action(ActionType.CHECK, Chips(0)))
    assert s.street == Street.TURN


def test_history_records_actions() -> None:
    cfg = GameConfig(
        num_players=2, small_blind=Chips(1), big_blind=Chips(2), starting_stack=Chips(200)
    )
    deal = deal_nlhe(cfg.num_players, hand_id=0, base_seed=0)
    s = initial_state(cfg, deal=deal, dealer_pos=0)
    assert s.history == tuple()

    s2 = apply_action(s, Action(ActionType.CALL, Chips(1)))
    assert len(s2.history) == 1
    rec = s2.history[0]
    assert isinstance(rec, ActionRecord)
    assert rec.street == Street.PREFLOP
    assert rec.player == 0
    assert rec.action_type == ActionType.CALL
    assert int(rec.amount) == 1
