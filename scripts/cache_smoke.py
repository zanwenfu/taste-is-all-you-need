"""Prompt-cache smoke test — the Wave-0 gate before any paid sweep.

Makes two identical tiny calls with a consolidated static system block that
crosses the cache minimum (1024 tokens on Sonnet-class models) and asserts the
second call reads from cache. Exits non-zero on failure.

    python scripts/cache_smoke.py        # needs ANTHROPIC_API_KEY in .env

Cost: well under $0.05.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taste.cores import MONITOR_SYSTEM, PLANNER_SYSTEM, WORKER_SYSTEM
from taste.llm import LLM, MODEL_WORKER, static_system


def main() -> int:
    # Consolidated static prefix, padded well past the 1024-token minimum the
    # same way real runs cross it (system prompts + agent spec + tool schemas).
    filler = (
        "Reference notes: the kernel checkpoints every step as a git commit, "
        "rolls back with git reset --hard on monitor failure, and pages context "
        "on demand with git show. "
    ) * 60
    system = [static_system(PLANNER_SYSTEM, WORKER_SYSTEM, MONITOR_SYSTEM, filler)]
    messages = [{"role": "user", "content": "Reply with the single word: ok"}]

    llm = LLM(budget_usd=0.25)
    for i in (1, 2):
        llm.call(
            model=MODEL_WORKER,
            system=system,
            messages=messages,
            max_tokens=8,
            temperature=0.0,
            role=f"smoke-{i}",
        )

    totals = llm.stats.totals
    second = llm.stats.per_role_model[("smoke-2", MODEL_WORKER)]
    print(
        f"calls={totals.calls} input={totals.input_tokens} "
        f"cache_write={totals.cache_creation_tokens} cache_read={totals.cache_read_tokens} "
        f"cost=${llm.stats.total_cost_usd:.4f} hit_rate={llm.stats.cache_hit_rate:.1%}"
    )
    if second.cache_read_tokens <= 0:
        print("FAIL: second call read 0 cache tokens — the static prefix is not caching.")
        return 1
    print(f"PASS: second call read {second.cache_read_tokens} tokens from cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
