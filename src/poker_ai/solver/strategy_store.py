from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poker_ai.abstraction.action_abstraction import AbstractAction, ActionAbstractionConfig
from poker_ai.abstraction.infoset import InfoSetKey
from poker_ai.engine.state import GameConfig


@dataclass(frozen=True, slots=True)
class StoredStrategy:
    game_cfg: GameConfig
    action_abs_cfg: ActionAbstractionConfig
    action_lists: dict[InfoSetKey, list[AbstractAction]]
    strategy_sum: dict[InfoSetKey, np.ndarray]


def save_strategy(path: str | Path, stored: StoredStrategy) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(stored, f)


def load_strategy(path: str | Path) -> StoredStrategy:
    p = Path(path)
    with p.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, StoredStrategy):
        raise TypeError("Invalid strategy file.")
    return obj
