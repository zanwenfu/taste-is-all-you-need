"""Record arms on the rule-loss benchmark: the published summary plus what a record keeps.

Each arm takes the summary a baseline run already made for an instance (its
``compacted`` field), adds a block kept from the record, and probes again with the
package's own probe, on the same model. The two full-history probes do not depend on
the context, so they are taken from the baseline row: the comparison is paired, the
summary is identical, and only the added block differs.

Arms:
  user_record     every user turn, verbatim, in order. The plainest reading of Taste: the
                  user's own words are the request record, never summarized.
  recent_control  the newest text of the history, cut to the same token count as the
                  user record of that instance: the same extra budget, without the record.

The added block sits in the same single assistant message as the summary, where the
package's compactors put their output. Verdicts are left for ``conpact_bridge
--judge-later``, which reads results.jsonl and judge_inputs.jsonl written here.

    python -m studies.context_rot.record_arms --arm user_record --baseline runs/conpact/hermes-pi_mono \
        --probe-url http://127.0.0.1:8000/v1 --tokenizer <gpt-oss folder> --out runs/conpact/hermes-pi_mono-user
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HEADERS = {"user_record": "The user's own messages in this session, verbatim, from the record:",
           "recent_control": "The most recent part of the session, verbatim:"}


def _render(message: dict) -> str:
    return f"[{message['role']}]: {message['content']}"


def user_record(messages: list[dict]) -> str:
    """Every user turn, verbatim and in order."""
    return "\n".join(_render(m) for m in messages if m["role"] == "user")


def recent_block(messages: list[dict], budget: int, count) -> str:
    """The newest messages that fit ``budget`` tokens; the oldest one kept is cut from its start."""
    kept: list[str] = []
    used = 0
    for message in reversed([m for m in messages if m["role"] != "system"]):
        text = _render(message)
        size = count(text)
        if used + size <= budget:
            kept.append(text)
            used += size
            continue
        words = text.split()
        while words and count(" ".join(words)) > budget - used:
            words = words[max(1, len(words) // 10):]
        if words:
            kept.append(" ".join(words))
        break
    return "\n".join(reversed(kept))


def arm_context(summary: list[dict], block: str, header: str) -> list[dict]:
    """One assistant message: the summary as the baseline returned it, then the kept block."""
    text = "\n\n".join(m["content"] for m in summary)
    return [{"role": "assistant", "content": f"{text}\n\n{header}\n{block}"}]


def score_paired(instances, baseline: dict, contexts: dict, probe, out: Path, workers: int = 8) -> list[dict]:
    """Probe each instance on its new context; take the full-history probes from its baseline row."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_inputs.jsonl").write_text(
        "".join(json.dumps({"id": row.id, "sc_text": row.sc_text}) + "\n" for row in instances))
    done = set()
    if (out / "results.jsonl").exists():
        done = {json.loads(line)["id"] for line in (out / "results.jsonl").read_text().splitlines()}
    lock = threading.Lock()

    def one(inst) -> dict:
        context, base = contexts[inst.id], baseline[inst.id]
        row = {key: getattr(inst, key) for key in ("id", "source", "haystack_id", "target_length", "token_length",
                                                   "position", "repeat", "strict", "direct", "sc_id", "sc_type",
                                                   "compliant_letter")}
        row.update(compacted=context, retention=None)
        for condition in ("full_with_sc", "full_without_sc"):
            row[f"{condition}_output"] = base[f"{condition}_output"]
            row[f"{condition}_compliant"] = base[f"{condition}_compliant"]
        for condition, user_turn in (("compaction", inst.probe_prompt), ("upper_bound", inst.post_sc_probe_prompt)):
            row[f"{condition}_output"], row[f"{condition}_compliant"] = probe(inst, context, user_turn)
        with lock, (out / "results.jsonl").open("a") as written:
            written.write(json.dumps(row) + "\n")
        return row

    todo = [inst for inst in instances if inst.id not in done]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    return [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=tuple(HEADERS))
    parser.add_argument("--baseline", required=True, help="a conpact_bridge output folder whose summaries to reuse")
    parser.add_argument("--probe-url", required=True)
    parser.add_argument("--probe-model", default="openai/gpt-oss-120b")
    parser.add_argument("--tokenizer", required=True, help="gpt-oss folder, to count block sizes as the paper does")
    parser.add_argument("--source", default="hermes")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)

    from conpact import Endpoint, load_static
    from conpact.evaluate import _probe
    from transformers import AutoTokenizer

    from studies.context_rot.conpact_bridge import DATA_REVISION, OUTPUT_CAP

    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    rows = [json.loads(line) for line in (Path(arguments.baseline) / "results.jsonl").read_text().splitlines()]
    baseline = {row["id"]: row for row in rows}
    instances = [inst for inst in load_static(arguments.source, revision=DATA_REVISION) if inst.id in baseline]
    contexts, sizes = {}, {}
    for inst in instances:
        record = user_record(inst.messages)
        block = record if arguments.arm == "user_record" else recent_block(inst.messages, count(record), count)
        contexts[inst.id] = arm_context(baseline[inst.id]["compacted"], block, HEADERS[arguments.arm])
        sizes[inst.id] = {"block_tokens": count(block), "context_tokens": count(contexts[inst.id][0]["content"]),
                          "summary_tokens": count("\n\n".join(m["content"] for m in baseline[inst.id]["compacted"]))}
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sizes.json").write_text(json.dumps(sizes, indent=1))
    endpoint = Endpoint(arguments.probe_model, arguments.probe_url, "EMPTY",
                        {"reasoning_effort": "low", "max_tokens": OUTPUT_CAP})
    client = endpoint.client()
    results = score_paired(instances, baseline, contexts,
                           lambda inst, context, turn: _probe(client, endpoint, inst, context, turn),
                           out, workers=arguments.workers)
    print(json.dumps({"arm": arguments.arm, "instances": len(results),
                      "block_tokens_median": sorted(s["block_tokens"] for s in sizes.values())[len(sizes) // 2]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
