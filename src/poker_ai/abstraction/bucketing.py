from __future__ import annotations

import itertools
from dataclasses import dataclass

from poker_ai.engine.cards import Card, Suit
from poker_ai.engine.observation import Observation
from poker_ai.engine.types import Street


def _preflop_class(hole: tuple[Card, Card]) -> int:
    """
    169-class encoding.
    Returns an id in [0, 168].
    We map (hi, lo, suited) into a stable index.
    """
    c1, c2 = hole
    r1, r2 = int(c1.rank), int(c2.rank)
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    suited = c1.suit == c2.suit

    # ranks 14..2 => map to 12..0
    def ridx(r: int) -> int:
        return 14 - r  # A->0, K->1, ..., 2->12

    i_hi = ridx(hi)
    i_lo = ridx(lo)

    # Build 13x13 grid: diagonal pairs, above diagonal suited, below offsuit
    # index = row*13 + col, where row is hi index, col is lo index
    # Convention: row <= col for suited? We'll use:
    # - pairs: row==col
    # - suited: row < col (hi stronger => smaller index means stronger rank)
    # - offsuit: row > col
    if hi == lo:
        row = i_hi
        col = i_hi
    else:
        if suited:
            row = min(i_hi, i_lo)
            col = max(i_hi, i_lo)
        else:
            row = max(i_hi, i_lo)
            col = min(i_hi, i_lo)

    return row * 13 + col


def _rank_5(cards: tuple[Card, ...]) -> int:
    """Return category 0..8 (same order as showdown: 0 high card ... 8 straight flush)."""
    from poker_ai.engine.showdown import rank_5

    cat, _k = rank_5(cards)
    return cat


def _best_cat(cards: tuple[Card, ...]) -> int:
    """
    Best category from N cards (N=5..7) by choosing best 5-card hand.
    Returns 0..8.
    """
    if len(cards) < 5:
        return 0
    best = 0
    for combo in itertools.combinations(cards, 5):
        best = max(best, _rank_5(combo))
        if best == 8:
            break
    return best


def _flush_draw(cards: tuple[Card, ...]) -> bool:
    suits = [c.suit for c in cards]
    for s in Suit:
        if sum(1 for x in suits if x == s) == 4:
            return True
    return False


def _straight_draw(cards: tuple[Card, ...]) -> int:
    """
    0: none, 1: gutshot-ish, 2: open-ender-ish (rough heuristic)
    Based on unique ranks in the available cards (Ace high and wheel considered roughly).
    """
    ranks = sorted({int(c.rank) for c in cards})
    if 14 in ranks:
        ranks.append(1)  # Ace as low for wheel patterns
    ranks = sorted(set(ranks))

    best = 0
    # check windows of 5 ranks with missing <= 1 => draw
    for start in range(1, 15):  # 1..14
        window = [start + i for i in range(5)]
        present = sum(1 for r in window if r in ranks)
        if present == 4:
            # missing one rank: could be gutshot or OESD depending on missing position
            missing = [r for r in window if r not in ranks][0]
            if missing == window[0] or missing == window[-1]:
                best = max(best, 2)  # open-ender-ish
            else:
                best = max(best, 1)  # gutshot-ish
    return best


@dataclass(frozen=True, slots=True)
class HandBucket:
    bucket_id: int


def bucket_hand(obs: Observation) -> HandBucket:
    """
    Returns a compact bucket id for the player's private hand given the public board (revealed).
    v1: very lightweight; no equity MonteCarlo.

    Best practice for thesis:
    - bucket_id should represent ONLY hand strength / texture features
    - position/seat should be represented separately in the InfoSetKey
    """
    if obs.street == Street.PREFLOP:
        cls = _preflop_class(obs.hole_cards)  # 0..168
        return HandBucket(bucket_id=cls)

    public_cards = tuple(obs.board)
    hole = obs.hole_cards
    cards = (hole[0], hole[1], *public_cards)

    cat = _best_cat(cards)  # 0..8
    fd = 1 if _flush_draw(cards) else 0
    sd = _straight_draw(cards)  # 0..2

    # Combine into bucket id
    # cat: 0..8, fd:0..1, sd:0..2 => 9*2*3 = 54 buckets
    bucket = (cat * 2 + fd) * 3 + sd
    return HandBucket(bucket_id=bucket)
