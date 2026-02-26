from __future__ import annotations

from dataclasses import dataclass

from poker_ai.abstraction.bucketing import bucket_hand
from poker_ai.engine.observation import Observation
from poker_ai.engine.types import Street


def _bucket_log2_bb(x_chips: int, bb: int) -> int:
    """
    Coarse log2 bucket in big blinds.
    """
    if bb <= 0:
        return 0
    bb_amt = max(0, x_chips // bb)
    # log2 bucketing: 0,1,2,4,8,16,32,...
    b = 0
    v = bb_amt
    while v > 0:
        v >>= 1
        b += 1
    return b  # 0 for 0bb, 1 for 1bb, 2 for 2-3bb, ...


def _history_abbrev(obs: Observation, last_n: int = 6) -> int:
    """
    Very compact history encoding based on last N action types.
    Encodes into base-5 integer (FOLD/CHECK/CALL/BET/RAISE).
    """
    mapping = {
        "fold": 0,
        "check": 1,
        "call": 2,
        "bet": 3,
        "raise": 4,
    }
    tail = obs.history[-last_n:]
    code = 0
    for r in tail:
        code = code * 5 + mapping[r.action_type.value]
    return code


@dataclass(frozen=True, slots=True)
class InfoSetKey:
    street: Street
    pos: int
    hand_bucket: int
    pot_bucket: int
    current_bet_bucket: int
    stack_bucket: int
    history_code: int
    facing_to_call_bucket: int


def make_infoset_key(obs: Observation, bb: int, history_len: int = 8) -> InfoSetKey:
    hb = bucket_hand(obs)
    pos = (obs.seat - obs.dealer_pos) % obs.num_players

    pot_bucket = _bucket_log2_bb(int(obs.pot), bb)
    bet_bucket = _bucket_log2_bb(int(obs.current_bet), bb)
    stack_bucket = _bucket_log2_bb(int(obs.stacks[obs.seat]), bb)

    to_call = max(0, int(obs.current_bet) - int(obs.street_commit[obs.seat]))
    facing_bucket = _bucket_log2_bb(to_call, bb)

    hist = _history_abbrev(obs, last_n=history_len)

    return InfoSetKey(
        street=obs.street,
        pos=pos,
        hand_bucket=hb.bucket_id,
        pot_bucket=pot_bucket,
        current_bet_bucket=bet_bucket,
        facing_to_call_bucket=facing_bucket,
        stack_bucket=stack_bucket,
        history_code=hist,
    )
