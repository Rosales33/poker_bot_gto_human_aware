from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RNG:
    seed: int

    def make(self) -> random.Random:
        return random.Random(self.seed)
