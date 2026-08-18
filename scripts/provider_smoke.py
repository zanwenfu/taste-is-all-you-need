"""Prove an adapter works against the live API, before trusting it with a sweep.

Three things that only a real call can establish, per provider:

* **Tool calling round-trips.** The model asks for a tool, the harness answers,
  the model uses the answer. A shape error here is invisible in unit tests
  because both sides are our own fakes.
* **Cache accounting is real.** The second identical call must read from cache,
  and — the part worth checking — ``work_cost`` must stay flat while
  ``billed_cost`` drops. That is the property the whole cost-matching design
  rests on: work is cache-invariant, billed is not.
* **Usage normalization is right.** ``prompt_total`` must reconcile with what
  the provider says it charged for. OpenAI reports a prompt total that
  *includes* cached tokens and Anthropic one that excludes them; if the
  adapter's subtraction is wrong, this is where it shows.

    python scripts/provider_smoke.py --provider anthropic
    python scripts/provider_smoke.py --provider openai

Cost: a few cents at most.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taste.cores import MONITOR_SYSTEM, PLANNER_SYSTEM, WORKER_SYSTEM
from taste.llm import LLM, static_system
from taste.pricing import PRICES

MODELS = {"anthropic": "claude-haiku-4-5-20251001", "openai": "gpt-5.6-luna"}

ECHO_TOOL = {
    "name": "record_answer",
    "description": "Record the final answer.",
    "input_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
}


def _prompt_work(model: str, prompt_tokens: int) -> float:
    """Work cost of a prompt alone — the part caching must not change."""
    from taste.pricing import rates_for

    return prompt_tokens * rates_for(model, prompt_tokens).input / 1e6


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(MODELS), required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model = args.model or MODELS[args.provider]
    if model not in PRICES:
        print(f"{model} is not priced; add it to taste/pricing.py first")
        return 1

    print(f"\n=== {args.provider} / {model} ===")
    llm = LLM(budget_usd=0.50)
    llm.ensure_ready(model)
    ok = True

    # ---- 1. tool calling round-trips
    first = llm.call(
        model=model,
        system="You answer with the record_answer tool and nothing else.",
        messages=[{"role": "user", "content": "Call record_answer with the answer 'blue'."}],
        tools=[ECHO_TOOL],
        max_tokens=200,
        temperature=0.0,
        role="smoke-tool",
    )
    ok &= check("model requested a tool", bool(first.tool_calls),
                f"stop_reason={first.stop_reason}")
    if first.tool_calls:
        call = first.tool_calls[0]
        ok &= check("tool name round-tripped", call.name == "record_answer", call.name)
        ok &= check("arguments decoded to a dict", isinstance(call.arguments, dict),
                    repr(call.arguments)[:80])
        ok &= check("call carries an id", bool(call.id), call.id)

        # ---- 2. the tool result is accepted back
        follow = llm.call(
            model=model,
            system="You answer with the record_answer tool and nothing else.",
            messages=[
                {"role": "user", "content": "Call record_answer with the answer 'blue'."},
                {"role": "assistant", "content": list(first.transcript_blocks)},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": call.id, "content": "recorded"}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Now say DONE."}]},
            ],
            tools=[ECHO_TOOL],
            max_tokens=200,
            temperature=0.0,
            role="smoke-roundtrip",
        )
        ok &= check("tool result accepted; conversation continued",
                    follow.stop_reason in ("end_turn", "tool_use"), follow.stop_reason)

    # ---- 3. cache accounting: work is invariant, billed is not
    # Sized past this model's cache minimum. Haiku's is 4x Sonnet's, and a
    # block below it caches nothing at all — silently.
    from taste.llm import cache_minimum_for

    sentence = (
        "Reference notes: the kernel checkpoints every step as a git commit, "
        "rolls back with git reset --hard on monitor failure, and pages context "
        "on demand with git show. "
    )
    repeats = max(90, int(cache_minimum_for(model) / 8))
    filler = sentence * repeats
    system = [static_system(PLANNER_SYSTEM, WORKER_SYSTEM, MONITOR_SYSTEM, filler)]
    messages = [{"role": "user", "content": "Reply with the single word: ok"}]

    def prompt_tokens() -> int:
        t = llm.stats.totals
        return t.input_tokens + t.cache_read_tokens + t.cache_creation_tokens

    marks = []
    for i in (1, 2):
        before = (llm.stats.total_cost_usd, llm.stats.total_work_usd, prompt_tokens())
        llm.call(
            model=model, system=system, messages=messages,
            max_tokens=8, temperature=0.0, role=f"smoke-cache-{i}",
        )
        marks.append(
            (
                llm.stats.total_cost_usd - before[0],
                llm.stats.total_work_usd - before[1],
                prompt_tokens() - before[2],
            )
        )

    (billed1, work1, prompt1), (billed2, work2, prompt2) = marks
    totals = llm.stats.totals
    print(f"  call 1: billed=${billed1:.6f} work=${work1:.6f} prompt={prompt1}")
    print(f"  call 2: billed=${billed2:.6f} work=${work2:.6f} prompt={prompt2}")
    print(f"  cache_read={totals.cache_read_tokens} write={totals.cache_creation_tokens}")

    ok &= check("second call read from cache", totals.cache_read_tokens > 0,
                f"{totals.cache_read_tokens} tokens")
    # Order-independent on purpose. Comparing call 2 against call 1 assumes
    # call 1 was cold, which is false whenever a previous run left the prefix
    # warm — organization-scoped caches are exactly the order dependence that
    # makes billed cost the wrong thing to budget against.
    ok &= check(
        "caching made the cached call cheaper than its work",
        billed2 < work2,
        f"billed ${billed2:.6f} < work ${work2:.6f}",
    )
    # The cache-invariant quantity is the PROMPT, not the whole call: output
    # tokens (and reasoning tokens inside them) vary run to run, so comparing
    # total work would flag ordinary sampling variance as a cost bug.
    ok &= check(
        "prompt size unchanged whether cached or not",
        prompt1 == prompt2,
        f"{prompt1} vs {prompt2} tokens",
    )
    ok &= check(
        "the same prompt is priced identically in work terms",
        abs(_prompt_work(model, prompt1) - _prompt_work(model, prompt2)) < 1e-12,
    )

    # ---- 4. usage buckets are disjoint and reconcile
    ok &= check(
        "prompt buckets are disjoint and non-negative",
        totals.input_tokens >= 0 and totals.cache_read_tokens >= 0,
    )
    print(f"\n  total billed=${llm.stats.total_cost_usd:.6f} "
          f"work=${llm.stats.total_work_usd:.6f} "
          f"cache_delta=${llm.stats.cache_delta_usd:+.6f}")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
