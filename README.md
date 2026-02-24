# Poker AI (GTO + Human-Aware) — Master Thesis Project

This repository contains a research-oriented Poker AI system for **No-Limit Texas Hold’em (6-max)**.
The long-term goal is to build an agent with a **GTO-inspired baseline** (robust, hard to exploit) and a
**human-aware adaptive layer** (opponent modeling + controlled exploitation).

The project is intentionally modular so each component can be evaluated and discussed in a thesis.

---

## Current Status

✅ **Phase 0 completed**: a deterministic, tested NLHE environment (engine + dealing + showdown + runner) with no information leakage to agents.

Next: **Phase 1** (abstractions + infosets + MCCFR blueprint).

---

## Tech Stack

- Python 3.11+
- `uv` for dependency management and virtual environments
- Strict typing (`mypy --strict`)
- Linting/formatting via `ruff`
- Tests with `pytest`
- Pre-commit hooks for auto-formatting and linting

---

## Setup

### 1) Install `uv`
If you don't have it yet, install `uv` following Astral's instructions.

### 2) Create environment and install dependencies
From the project root:

```bash
uv venv
uv sync
```
## Run Tests / Type Checks / Lint

```bash
uv run pytest
uv run mypy src
uv run ruff check .
uv run ruff format .
```

## Repository Structure

src/poker_ai/
  engine/            # Core NLHE rules + state transitions
  abstraction/       # (Phase 1+) Action & information abstractions
  solver/            # (Phase 1+) MCCFR/CFR implementations
  opponent_model/    # (Phase 2+) Human-aware opponent modeling
  response/          # (Phase 2+) Exploit / restricted-response logic
  agent/             # Agents/policies
  experiments/       # Match runners, benchmarks, evaluation
  utils/             # Reusable utilities

tests/               # Unit tests (engine, dealing, showdown, runner, observations)
data/                # Local outputs (logs, models, blueprints) — usually gitignored
configs/             # Experiment configs
