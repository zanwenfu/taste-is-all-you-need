"""Serving-path check for the rule-loss reproduction.

The paper ran gpt-oss-120b through vLLM's offline engine (its VLLMRuntime: LLM.chat with the
model's chat template, reasoning effort low, Harmony stop tokens, its own answer parser); the
released package, used for our reproduction, calls vLLM's OpenAI server. On the same cases this
asks the same probes through the paper's own VLLMRuntime:

  1. on the summaries our server run made (both prompts), and
  2. for the Anthropic prompt, on summaries made again through the same offline engine,

so a gap between our numbers and the paper's can be put on the summaries, on the probe, or on
neither. Answers are graded by the package's rule (last non-empty line) and the paper runner's
(the whole reply is one letter).

    python -m studies.context_rot.engine_check --paper-src <compaction-integrity/src> --model <gpt-oss folder> \\
        --full <runs/conpact/full> --histories 0:10 --out <folder>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from studies.context_rot.compare import median


def probe_messages(inst, context: list[dict], turn: str) -> list[dict]:
    """The package's probe: system, developer, the context, then the user turn (conpact.evaluate._probe)."""
    return [{"role": "system", "content": inst.system_prompt},
            {"role": "developer", "content": inst.developer_prompt},
            *context, {"role": "user", "content": turn}]


def grade(text: str, letter: str) -> tuple[bool | None, bool | None]:
    """(package rule, paper runner's rule), after the engine's "[final]" or "[commentary]" label."""
    body = text or ""
    for label in ("[final]\n", "[commentary]\n"):
        if body.startswith(label):
            body = body[len(label):]
    lines = [line for line in body.strip().splitlines() if line.strip()]
    last = lines[-1].strip().upper() if lines else ""
    whole = body.strip().upper()
    return ((last == letter) if last in ("A", "B") else None, (whole == letter) if whole in ("A", "B") else None)


def _rate(values):
    clear = [v for v in values if v is not None]
    return {"pct": round(100 * sum(clear) / len(clear), 1) if clear else None, "n": len(clear),
            "unreadable": len(values) - len(clear)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-src", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--full", required=True, help="our full-run folder: hermes-<prompt>/hSS-EE/results.jsonl")
    parser.add_argument("--histories", default="0:10")
    parser.add_argument("--cache", required=True, help="HF and vLLM cache folder")
    parser.add_argument("--arms-only", action="store_true",
                        help="probe only the record arms' contexts (user_record, recent_control) of the full run")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    sys.path.insert(0, arguments.paper_src)

    from compaction_integrity.runtime.vllm_runtime import VLLMRuntime
    from conpact import load_static
    from conpact.prompts import summarization_prompt

    from studies.context_rot.conpact_bridge import DATA_REVISION, OUTPUT_CAP, history_range

    start, end = (int(part) for part in arguments.histories.split(":"))
    instances = history_range(load_static("hermes", revision=DATA_REVISION), start, end)
    ids = {inst.id for inst in instances}
    ours = {}
    for prompt in ("pi_mono", "anthropic"):
        rows = {}
        for part in sorted(Path(arguments.full, f"hermes-{prompt}").glob("h??-??")):
            for line in (part / "results.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row["id"] in ids:
                    rows[row["id"]] = row
        ours[prompt] = rows
    runtime = VLLMRuntime({"model": arguments.model, "max_tokens": OUTPUT_CAP, "use_tqdm": True,
                           "gpu_memory_utilization": 0.95,
                           "env": {"hf_home": arguments.cache, "hf_token_path": f"{arguments.cache}/token",
                                   "vllm_cache_root": arguments.cache}})
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    def ask(batch):
        return [response.text for response in runtime.batch_generate(batch)]

    report, records = {"cases": len(instances)}, []
    if arguments.arms_only:
        for prompt in ("pi_mono", "anthropic"):
            for arm in ("user_record", "recent_control"):
                contexts = {}
                for part in sorted(Path(arguments.full, f"hermes-{prompt}").glob(f"h??-??-{arm}")):
                    for line in (part / "results.jsonl").read_text().splitlines():
                        row = json.loads(line)
                        if row["id"] in ids:
                            contexts[row["id"]] = row
                batch, keys = [], []
                for inst in instances:
                    for condition, turn in (("compaction", inst.probe_prompt),
                                            ("upper_bound", inst.post_sc_probe_prompt)):
                        batch.append(probe_messages(inst, contexts[inst.id]["compacted"], turn))
                        keys.append((inst, condition))
                for (inst, condition), text in zip(keys, ask(batch)):
                    package, paper = grade(text, inst.compliant_letter)
                    records.append({"id": inst.id, "step": f"arm_{arm}", "prompt": prompt, "condition": condition,
                                    "output": text, "package": package, "paper": paper,
                                    "server_package": contexts[inst.id][f"{condition}_compliant"]})
                for condition in ("compaction", "upper_bound"):
                    chosen = [r for r in records if r["step"] == f"arm_{arm}" and r["prompt"] == prompt
                              and r["condition"] == condition]
                    report[f"arm_{arm}|{prompt}|{condition}"] = {
                        "engine_package_rule": _rate([r["package"] for r in chosen]),
                        "engine_paper_rule": _rate([r["paper"] for r in chosen]),
                        "server_package_rule": _rate([r["server_package"] for r in chosen])}
        (out / "arms_records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        (out / "arms_report.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return 0
    # 1. our server-made summaries, probed through the paper's engine
    for prompt in ("pi_mono", "anthropic"):
        batch, keys = [], []
        for inst in instances:
            context = ours[prompt][inst.id]["compacted"]
            for condition, turn in (("compaction", inst.probe_prompt), ("upper_bound", inst.post_sc_probe_prompt)):
                batch.append(probe_messages(inst, context, turn))
                keys.append((inst, condition))
        texts = ask(batch)
        for (inst, condition), text in zip(keys, texts):
            package, paper = grade(text, inst.compliant_letter)
            records.append({"id": inst.id, "step": "our_summary", "prompt": prompt, "condition": condition,
                            "output": text, "package": package, "paper": paper,
                            "server_package": ours[prompt][inst.id][f"{condition}_compliant"]})
    # 2. Anthropic summaries made again through the paper's engine, then probed through it
    summaries = ask([summarization_prompt("anthropic", inst.messages) for inst in instances])
    (out / "engine_summaries.jsonl").write_text("".join(json.dumps({"id": inst.id, "summary": text}) + "\n"
                                                        for inst, text in zip(instances, summaries)))
    batch, keys = [], []
    for inst, text in zip(instances, summaries):
        context = [{"role": "assistant", "content": text}]
        for condition, turn in (("compaction", inst.probe_prompt), ("upper_bound", inst.post_sc_probe_prompt)):
            batch.append(probe_messages(inst, context, turn))
            keys.append((inst, condition))
    for (inst, condition), text in zip(keys, ask(batch)):
        package, paper = grade(text, inst.compliant_letter)
        records.append({"id": inst.id, "step": "engine_summary", "prompt": "anthropic", "condition": condition,
                        "output": text, "package": package, "paper": paper})
    (out / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    for step, prompt in (("our_summary", "pi_mono"), ("our_summary", "anthropic"), ("engine_summary", "anthropic")):
        for condition in ("compaction", "upper_bound"):
            chosen = [r for r in records if r["step"] == step and r["prompt"] == prompt and r["condition"] == condition]
            entry = {"engine_package_rule": _rate([r["package"] for r in chosen]),
                     "engine_paper_rule": _rate([r["paper"] for r in chosen])}
            if step == "our_summary":
                entry["server_package_rule"] = _rate([r["server_package"] for r in chosen])
            report[f"{step}|{prompt}|{condition}"] = entry
    lengths = sorted(len(text) for text in summaries)
    report["engine_summary_chars"] = {"min": lengths[0], "median": median(lengths), "max": lengths[-1],
                                      "empty": sum(not text.strip() for text in summaries)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
