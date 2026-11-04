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
  criteria        the standing instructions a reader wrote down, verbatim, as each user message
                  arrived (criteria.py); one model call per distinct user message.
  criteria_control  the newest text, cut to the token count of the criteria block.

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

from studies.context_rot.compare import median

HEADERS = {"user_record": "The user's own messages in this session, verbatim, from the record:",
           "recent_control": "The most recent part of the session, verbatim:",
           "criteria": "Standing instructions the user gave in this session, verbatim, from the record:",
           "criteria_control": "The most recent part of the session, verbatim:"}


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


def stored_records(path: Path) -> dict:
    """The per-case records an earlier criteria arm wrote, so arm and control share one reading."""
    return json.loads(Path(path).read_text())["records"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=tuple(HEADERS))
    parser.add_argument("--baseline", required=True, help="a conpact_bridge output folder whose summaries to reuse")
    parser.add_argument("--probe-url", required=True)
    parser.add_argument("--probe-model", default="openai/gpt-oss-120b")
    parser.add_argument("--tokenizer", required=True, help="gpt-oss folder, to count block sizes as the paper does")
    parser.add_argument("--source", default="hermes")
    parser.add_argument("--build-length", type=int, help="instances built from the long haystacks (as the baseline)")
    parser.add_argument("--no-cue", action="store_true")
    parser.add_argument("--drop-cue", action="store_true", help="the published cases without the session cue")
    parser.add_argument("--records-from", help="records.json of an earlier criteria arm: reuse its reading of the "
                                                "user messages (the records do not depend on the summary prompt)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)

    from conpact.evaluate import _probe
    from transformers import AutoTokenizer

    from studies.context_rot.conpact_bridge import DATA_REVISION, OUTPUT_CAP, load_instances

    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    rows = [json.loads(line) for line in (Path(arguments.baseline) / "results.jsonl").read_text().splitlines()]
    baseline = {row["id"]: row for row in rows}
    instances = [inst for inst in load_instances(arguments.source, DATA_REVISION, arguments.build_length,
                                                 not arguments.no_cue, arguments.drop_cue) if inst.id in baseline]
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    if arguments.arm.startswith("criteria"):
        from openai import OpenAI

        from studies.context_rot.criteria import records_block, records_for, served_reader

        if arguments.records_from:
            records = stored_records(arguments.records_from)
        else:
            log: list = []
            reader = served_reader(OpenAI(base_url=arguments.probe_url, api_key="EMPTY"), arguments.probe_model, log)
            records = records_for(instances, reader, workers=arguments.workers)
            (out / "records.json").write_text(json.dumps({"records": records, "reads": log}, indent=1))
    contexts, sizes = {}, {}
    for inst in instances:
        if arguments.arm.startswith("criteria"):
            record = records_block(records[inst.id])
        else:
            record = user_record(inst.messages)
        block = record if arguments.arm in ("user_record", "criteria") else recent_block(inst.messages, count(record),
                                                                                         count)
        contexts[inst.id] = arm_context(baseline[inst.id]["compacted"], block, HEADERS[arguments.arm])
        sizes[inst.id] = {"block_tokens": count(block), "context_tokens": count(contexts[inst.id][0]["content"]),
                          "summary_tokens": count("\n\n".join(m["content"] for m in baseline[inst.id]["compacted"]))}
    (out / "sizes.json").write_text(json.dumps(sizes, indent=1))
    from studies.context_rot.conpact_bridge import guarded_endpoint

    endpoint = guarded_endpoint(arguments.probe_model, arguments.probe_url,
                                {"reasoning_effort": "low", "max_tokens": OUTPUT_CAP}, out / "not_served.jsonl")
    client = endpoint.client()
    results = score_paired(instances, baseline, contexts,
                           lambda inst, context, turn: _probe(client, endpoint, inst, context, turn),
                           out, workers=arguments.workers)
    print(json.dumps({"arm": arguments.arm, "instances": len(results),
                      "block_tokens_median": median(s["block_tokens"] for s in sizes.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
