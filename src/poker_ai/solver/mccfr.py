from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from poker_ai.abstraction.action_abstraction import (
    AbstractAction,
    ActionAbstractionConfig,
    abstract_to_action,
    get_abstract_actions,
)
from poker_ai.abstraction.infoset import InfoSetKey, make_infoset_key
from poker_ai.engine.dealing import deal_nlhe
from poker_ai.engine.observe import observe
from poker_ai.engine.payout import settle_terminal_hand
from poker_ai.engine.rules import apply_action
from poker_ai.engine.state import GameConfig, GameState, initial_state


@dataclass(frozen=True, slots=True)
class MCCFRConfig:
    iterations: int = 10_000
    seed: int = 0
    base_deal_seed: int = 0
    action_abs: ActionAbstractionConfig = ActionAbstractionConfig()
    # discounting can help stability; keep 1.0 for now
    regret_discount: float = 1.0


def _regret_matching(regrets: NDArray[np.float64]) -> NDArray[np.float64]:
    pos: NDArray[np.float64] = np.maximum(regrets, 0.0)
    s = float(pos.sum())
    if s > 1e-12:
        out: NDArray[np.float64] = pos / s
        return out
    n = int(regrets.size)
    out2: NDArray[np.float64] = np.full((n,), 1.0 / n, dtype=np.float64)
    return out2


class MCCFRTrainer:
    """
    External sampling MCCFR (practical variant).
    Stores regrets and average strategy per InfoSetKey.
    """

    def __init__(self, cfg: GameConfig, mccfr_cfg: MCCFRConfig) -> None:
        self.game_cfg = cfg
        self.cfg = mccfr_cfg
        self.rng = random.Random(mccfr_cfg.seed)

        # per infoset: list of abstract actions (stable ordering)
        self.action_lists: dict[InfoSetKey, list[AbstractAction]] = {}

        # regrets and average strategy sums aligned with action_lists
        self.regret_sum: dict[InfoSetKey, np.ndarray] = {}
        self.strategy_sum: dict[InfoSetKey, np.ndarray] = {}

    def _bb(self) -> int:
        return int(self.game_cfg.big_blind)

    def _get_actions_for_key(self, key: InfoSetKey, state: GameState) -> list[AbstractAction]:
        legal_abs = get_abstract_actions(state, self.cfg.action_abs)

        if key not in self.action_lists:
            self.action_lists[key] = list(legal_abs)
            n = len(self.action_lists[key])
            self.regret_sum[key] = np.zeros(n, dtype=np.float64)
            self.strategy_sum[key] = np.zeros(n, dtype=np.float64)
            return self.action_lists[key]

        existing = self.action_lists[key]
        existing_set = {(a.type, a.param) for a in existing}
        changed = False
        for a in legal_abs:
            if (a.type, a.param) not in existing_set:
                existing.append(a)
                existing_set.add((a.type, a.param))
                changed = True

        if changed:
            old_r = self.regret_sum[key]
            old_s = self.strategy_sum[key]
            n_new = len(existing)
            self.regret_sum[key] = np.pad(old_r, (0, n_new - old_r.size), constant_values=0.0)
            self.strategy_sum[key] = np.pad(old_s, (0, n_new - old_s.size), constant_values=0.0)

        return existing

    def _strategy(self, key: InfoSetKey) -> NDArray[np.float64]:
        return _regret_matching(self.regret_sum[key])

    def _sample_index(self, probs: np.ndarray) -> int:
        x = self.rng.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += float(p)
            if x <= c:
                return i
        return int(probs.size - 1)

    def _utility_chips(self, final_state: GameState, traverser: int) -> float:
        """
        Utility in chips relative to starting stack.
        Assumes no rake.
        """
        start = int(self.game_cfg.starting_stack)
        end = int(final_state.stacks[traverser])
        return float(end - start)

    def _traverse(
        self,
        state: GameState,
        traverser: int,
        reach_others: float,
        dealer_pos: int,
    ) -> float:
        """
        Returns utility for traverser from this state.
        External sampling:
        - At traverser nodes: enumerate actions and update regrets.
        - At opponent nodes: sample one action according to their current strategy.
        """
        if state.is_terminal():
            settled, _main = settle_terminal_hand(state)
            return self._utility_chips(settled, traverser)

        seat = state.to_act

        obs = observe(state, seat)
        key = make_infoset_key(obs, bb=self._bb())

        # terminal safety: some states may have no legal abstract actions due to mapping; fall back to engine legal?
        # In practice, our get_abstract_actions filters by engine legality already.
        action_list = self._get_actions_for_key(key, state)
        strat = self._strategy(key)

        # Determine which abstract actions are currently legal (subset of action_list)
        current_legals = get_abstract_actions(state, self.cfg.action_abs)
        legal_set = {(a.type, a.param) for a in current_legals}

        legal_idx: list[int] = [
            i for i, a in enumerate(action_list) if (a.type, a.param) in legal_set
        ]
        if not legal_idx:
            # Should not happen; if it does, treat as check if possible by forcing engine.
            raise RuntimeError(
                "No legal abstract actions for this state/key. Check action abstraction."
            )

        # Normalize strategy over legal actions only
        probs = np.array([strat[i] for i in legal_idx], dtype=np.float64)
        s = float(probs.sum())
        if s <= 1e-12:
            probs = np.full(len(legal_idx), 1.0 / len(legal_idx), dtype=np.float64)
        else:
            probs /= s

        if seat == traverser:
            # Update average strategy for this infoset for traverser
            for p, idx in zip(probs, legal_idx, strict=True):
                self.strategy_sum[key][idx] += reach_others * float(p)

            # Enumerate traverser actions
            action_utils = np.zeros(len(legal_idx), dtype=np.float64)

            for j, idx in enumerate(legal_idx):
                aa = action_list[idx]
                eng_action = abstract_to_action(state, aa)
                next_state = apply_action(state, eng_action)
                action_utils[j] = self._traverse(
                    next_state,
                    traverser=traverser,
                    reach_others=reach_others,
                    dealer_pos=dealer_pos,
                )

            node_util = float(np.dot(probs, action_utils))

            # Regret update (aligned to full action list)
            for j, idx in enumerate(legal_idx):
                regret = float(action_utils[j] - node_util)
                self.regret_sum[key][idx] = (
                    self.cfg.regret_discount * self.regret_sum[key][idx] + regret
                )

            return node_util

        # Opponent node: sample one action
        sampled_j = self._sample_index(probs)
        sampled_idx = legal_idx[sampled_j]
        sampled_aa = action_list[sampled_idx]
        eng_action = abstract_to_action(state, sampled_aa)
        next_state = apply_action(state, eng_action)

        # Update reach prob of "others" (opponents) for traverser’s updates
        new_reach_others = reach_others * float(probs[sampled_j])

        return self._traverse(
            next_state, traverser=traverser, reach_others=new_reach_others, dealer_pos=dealer_pos
        )

    def run(self, iterations: int | None = None) -> None:
        iters = iterations if iterations is not None else self.cfg.iterations

        dealer = 0
        for t in range(iters):
            traverser = t % self.game_cfg.num_players

            # sample chance (deal)
            deal = deal_nlhe(
                self.game_cfg.num_players, hand_id=t, base_seed=self.cfg.base_deal_seed
            )

            state = initial_state(self.game_cfg, deal=deal, dealer_pos=dealer)
            # important: hand_id exists in state in your code; if not, remove this line
            # state = replace(state, hand_id=t)

            _ = self._traverse(state, traverser=traverser, reach_others=1.0, dealer_pos=dealer)

            dealer = (dealer + 1) % self.game_cfg.num_players

    def average_strategy(self, key: InfoSetKey) -> np.ndarray:
        s = self.strategy_sum[key]
        total = float(s.sum())
        if total <= 1e-12:
            n = s.size
            return np.full(n, 1.0 / n, dtype=np.float64)
        return s / total
