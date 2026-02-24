from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence

from poker_ai.engine.cards import Card


def _is_straight(ranks: list[int]) -> int | None:
    """Return high card of straight, handling wheel A-5."""
    uniq = sorted(set(ranks), reverse=True)
    if len(uniq) < 5:
        return None
    # wheel
    if set([14, 5, 4, 3, 2]).issubset(set(uniq)):
        return 5
    for i in range(len(uniq) - 4):
        window = uniq[i : i + 5]
        if window[0] - window[4] == 4:
            return window[0]
    return None


def rank_5(cards: tuple[Card, ...]) -> tuple[int, list[int]]:
    ranks = sorted((int(c.rank) for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    flush = len(set(suits)) == 1
    straight_high = _is_straight(ranks)

    cnt = Counter(ranks)
    # sort by (count desc, rank desc)
    counts = sorted(cnt.values(), reverse=True)

    if flush and straight_high is not None:
        return (8, [straight_high])  # straight flush
    if counts == [4, 1]:
        four_rank = max(r for r, c in cnt.items() if c == 4)
        kicker = max(r for r, c in cnt.items() if c == 1)
        return (7, [four_rank, kicker])
    if counts == [3, 2]:
        trips = max(r for r, c in cnt.items() if c == 3)
        pair = max(r for r, c in cnt.items() if c == 2)
        return (6, [trips, pair])
    if flush:
        return (5, ranks)
    if straight_high is not None:
        return (4, [straight_high])
    if counts == [3, 1, 1]:
        trips = max(r for r, c in cnt.items() if c == 3)
        kickers = sorted((r for r, c in cnt.items() if c == 1), reverse=True)
        return (3, [trips] + kickers)
    if counts == [2, 2, 1]:
        pairs = sorted((r for r, c in cnt.items() if c == 2), reverse=True)
        kicker = max(r for r, c in cnt.items() if c == 1)
        return (2, pairs + [kicker])
    if counts == [2, 1, 1, 1]:
        pair = max(r for r, c in cnt.items() if c == 2)
        kickers = sorted((r for r, c in cnt.items() if c == 1), reverse=True)
        return (1, [pair] + kickers)
    return (0, ranks)


def best_of_7(cards7: tuple[Card, ...]) -> tuple[int, list[int]]:
    best: tuple[int, list[int]] | None = None
    for combo in itertools.combinations(cards7, 5):
        r = rank_5(combo)
        if best is None or r > best:
            best = r
    assert best is not None
    return best


def winners(
    hole_cards: tuple[tuple[Card, Card], ...],
    board: tuple[Card, ...],
    active: tuple[bool, ...],
) -> list[int]:
    ranks: list[tuple[tuple[int, list[int]], int]] = []
    for i, is_in in enumerate(active):
        if not is_in:
            continue
        cards7 = (hole_cards[i][0], hole_cards[i][1], *board)
        ranks.append((best_of_7(cards7), i))

    if not ranks:
        return []

    best_rank = max(r[0] for r in ranks)
    return [i for r, i in ranks if r == best_rank]


def winners_among(
    seats: Sequence[int],
    hole_cards: tuple[tuple[Card, Card], ...],
    board: tuple[Card, ...],
) -> list[int]:
    ranks: list[tuple[tuple[int, list[int]], int]] = []
    for i in seats:
        cards7 = (hole_cards[i][0], hole_cards[i][1], *board)
        ranks.append((best_of_7(cards7), i))

    if not ranks:
        return []

    best_rank = max(r[0] for r in ranks)
    return [i for r, i in ranks if r == best_rank]
