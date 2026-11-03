"""The package's retention judge asked of the served model, on summaries a run already made.

Used to measure how far the free judge (gpt-oss-120b on the cluster) agrees with the paper's
GPT-5.4 on the same single-compaction summaries, before it judges the repeated rounds alone.
Writes verdicts.served.jsonl (id, retention) into each folder; resumes by id.

    python -m studies.context_rot.served_judge --probe-url http://127.0.0.1:8000/v1 <run folder> [...]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def judge_folder(folder: Path, verdict, workers: int = 16) -> int:
    """Ask ``verdict(sc_text, compacted)`` for every row not yet in verdicts.served.jsonl."""
    folder = Path(folder)
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    texts = {item["id"]: item["sc_text"]
             for item in map(json.loads, (folder / "judge_inputs.jsonl").read_text().splitlines())}
    ledger = folder / "verdicts.served.jsonl"
    done = set()
    if ledger.exists():
        done = {json.loads(line)["id"] for line in ledger.read_text().splitlines()}
    lock = threading.Lock()

    def one(row: dict) -> None:
        answer = verdict(texts[row["id"]], row["compacted"])
        with lock, ledger.open("a") as written:
            written.write(json.dumps({"id": row["id"], "retention": answer}) + "\n")

    todo = [row for row in rows if row["id"] not in done]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    return len(todo)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--probe-url", required=True)
    parser.add_argument("--probe-model", default="openai/gpt-oss-120b")
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args(argv)

    from conpact import Endpoint
    from conpact.prompts import judge_prompts

    from studies.context_rot.conpact_bridge import RetentionJudge
    from studies.context_rot.repeated import ServedChat

    client = Endpoint(arguments.probe_model, arguments.probe_url, "EMPTY", {}).client()
    judge = RetentionJudge(ServedChat(client, arguments.probe_model))
    asked = {folder: judge_folder(Path(folder), lambda sc, compacted: judge.verdict(*judge_prompts(sc, compacted)),
                                  arguments.workers) for folder in arguments.folders}
    print(json.dumps({"asked": asked, "calls": judge.calls, "unclear": judge.unparseable,
                      "input_tokens": judge.input_tokens, "output_tokens": judge.output_tokens}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
