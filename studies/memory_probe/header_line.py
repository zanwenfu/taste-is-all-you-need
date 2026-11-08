"""Whether a header list delivered the rule, read line by line.

The retention judge was built and calibrated for summaries. Given a header list whole, a long list of one-line headers, it
misses the rule turn's own line (the middle set: that line delivered in 135 cases, the judge's yes in 25). So a header-list
row is read here one line at a time. The rule reached the model through the list when the line of a turn that holds the rule
was delivered and that one line, read by the judge alone, states the rule; or through a fetch of such a turn, whose words hold
the rule whole (the row's own ``found``). The line is the header file's, in the index's own form; a delivered line that
differs from it is counted, and the delivered one is what the judge reads.

    python -m studies.memory_probe.header_line --results <header-list row>/results.jsonl --cases inputs/cases.jsonl \\
        --headers <matrix>/turn-headers.jsonl --url URL --model NAME

One row per case in header-line.served.jsonl beside the results. A cases or headers file other than the row's own is refused
before any call. Run again, it asks only what has no verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

OUT_NAME = "header-line.served.jsonl"


def turn_key(turn: dict) -> str:
    """headers.turn_key, written out here: that module needs the git library through Taste's store, which the judge's own environment does not hold. A test holds the two equal."""
    text = json.dumps({"role": turn["role"], "content": turn["content"]}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index_line(turn: int, header: dict | None) -> str:
    """A turn's line as the index writes it (cases._index_lines): its place counted from 1, what its header says, its keywords."""
    what, words = (header["what"], header["keywords"]) if header and header.get("ok") else ("", [])
    return f"{turn + 1} {what}" + (f" [{', '.join(words)}]" if words else "")


def _delivered_line(block: str, turn: int, wanted: str) -> str | None:
    """The line the model was given for this turn: the header file's line if the block holds it whole, else the block's own line with the turn's number."""
    if f"\n{wanted}\n" in f"\n{block}\n":
        return wanted
    return next((line for line in block.split("\n") if line.startswith(f"{turn + 1} ")), None)


def check_header_lines(results: Path, cases: Path, headers: Path, verdict: Callable[[str, list[dict]], bool | None], *, workers: int) -> dict[str, int]:
    results = Path(results)
    settings = json.loads(results.with_name("results.settings.json").read_text())
    for name, path in (("cases", cases), ("headers", headers)):
        if _sha256(path) != settings[f"{name}_sha256"]:
            raise ValueError(f"{path} is not the {name} file this row was run with")
    by_id = {case["id"]: case for case in map(json.loads, Path(cases).read_text().splitlines())}
    stored = {row["key"]: row for row in map(json.loads, Path(headers).read_text().splitlines())}
    out = results.with_name(OUT_NAME)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    lock = threading.Lock()

    def one(row: dict) -> None:
        case = by_id[row["id"]]
        block = row.get("delivered_text") or ""
        checked: list[dict[str, Any]] = []
        for turn in row["rule_turns"]:
            header = stored.get(turn_key(case["messages"][turn]))
            wanted = index_line(turn, header)
            line = _delivered_line(block, turn, wanted)
            missing = not (header and header.get("ok"))
            states = None if line is not None and not missing else False
            checked.append({"turn": turn, "line": line if line is not None else wanted, "delivered": line is not None, "differs": line is not None and line != wanted,
                            "missing_header": missing, "states": states})
        asked = 0
        for item in checked:
            if item["delivered"] and not item["missing_header"]:
                item["states"] = verdict(case["sc_text"], [{"role": "assistant", "content": item["line"]}])
                asked += 1
                if item["states"] is True:
                    break
        delivered = [item for item in checked if item["delivered"]]
        states = True if any(item["states"] is True for item in delivered) else (None if any(item["states"] is None for item in delivered) else False)
        # The row's own found: the list holding the rule whole, or a fetch bringing it back whole. The list never holds it whole in these rows (rule_in_index).
        fetched = bool(row.get("found")) and not row.get("rule_in_index")
        held = True if row.get("found") or states is True else (None if states is None else False)
        first = (delivered or checked or [{"line": None}])[0]
        written = {"id": row["id"], "rule_turns": row["rule_turns"], "line_delivered": bool(delivered), "line": first["line"], "states_rule": states,
                   "fetched": fetched, "held": held, "differs": any(item["differs"] for item in checked),
                   "missing_header": any(item["missing_header"] for item in checked), "asked": asked}
        with lock, out.open("a") as handle:
            handle.write(json.dumps(written) + "\n")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, [row for row in rows if row["id"] not in done]))
    found = [json.loads(line) for line in out.read_text().splitlines()]
    return {"cases": len(found), "line_delivered": sum(row["line_delivered"] for row in found), "states_rule": sum(row["states_rule"] is True for row in found),
            "fetched": sum(row["fetched"] for row in found), "held": sum(row["held"] is True for row in found),
            "unclear": sum(row["held"] is None for row in found), "differs": sum(row["differs"] for row in found),
            "missing_header": sum(row["missing_header"] for row in found), "no_rule_turn": sum(not row["rule_turns"] for row in found),
            "asked": sum(row["asked"] for row in found)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("results", "cases", "headers", "url", "model"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args(argv)

    from conpact import Endpoint
    from conpact.prompts import judge_prompts

    from studies.context_rot.conpact_bridge import RetentionJudge
    from studies.context_rot.repeated import ServedChat

    judge = RetentionJudge(ServedChat(Endpoint(arguments.model, arguments.url, "EMPTY", {}).client(), arguments.model))

    def verdict(rule: str, delivered: list[dict]) -> bool | None:
        return judge.verdict(*judge_prompts(rule, delivered))

    print(json.dumps(check_header_lines(Path(arguments.results), Path(arguments.cases), Path(arguments.headers), verdict, workers=arguments.workers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
