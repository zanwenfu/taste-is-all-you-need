"""MemPalace 3.11.0 (commit 602d796c), run in its own environment on the shared ingest file. Two arms, reported apart.

* **Benchmark arm** (what MemPalace's published numbers mean): its own benchmark functions, imported from the
  repository at the release commit. LongMemEval: ``build_palace_and_retrieve`` (raw) or
  ``build_palace_and_retrieve_hybrid_v4``, with ``n_results`` 50 as published; they index user turns only and rank
  sessions, and the reader then gets each ranked session whole. LoCoMo: its helpers rebuilt per conversation, as
  its runner does, at session or dialog level.
* **Product arm** (what a user of MemPalace gets): each session written as one Claude Code JSONL file named by its
  neutral id, mined with ``mine_convos`` and searched with ``search_memories``; the reader gets the drawers found.

Facts behind each choice, with file and line at the release commit, are in the study's notes. Nothing here calls a
model; the only network use is the embedding model's first download.

    python -m studies.memory_bench.adapters.mempalace_arms --views views.jsonl --repo <mempalace clone> \\
        --arm bench_raw --out rankings.jsonl
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from studies.memory_bench.adapters.common import run_adapter
from studies.memory_bench.core import Item
from studies.memory_bench.lme import rounds

COMMIT = "602d796c699cabba2971982cd9dccbbf46abfe79"
ARMS = ("bench_raw", "bench_v4", "product")


def _all_rounds(session: dict) -> tuple[int, ...]:
    return tuple(range(len(rounds(session))))


def lme_entry(sessions: list[dict], query: str, date: str | None) -> dict[str, Any]:
    """A LongMemEval question as MemPalace's benchmark functions read it, from what a memory may see."""
    return {"question": query, "question_date": date or "", "haystack_session_ids": [s["sid"] for s in sessions],
            "haystack_dates": [s["date"] for s in sessions],
            "haystack_sessions": [[{"role": t["role"], "content": t["content"]} for t in s["turns"]] for s in sessions]}


def iso_date(date: str) -> str:
    """LongMemEval's "2023/05/02 (Tue) 09:00" or LoCoMo's "1:56 pm on 8 May, 2023", as ISO time."""
    match = re.fullmatch(r"(\d{4})/(\d{2})/(\d{2}) \(\w+\) (\d{2}):(\d{2})", date.strip())
    if match:
        year, month, day, hour, minute = map(int, match.groups())
        return datetime(year, month, day, hour, minute).isoformat()
    return datetime.strptime(date.strip(), "%I:%M %p on %d %B, %Y").isoformat()


def claude_code_jsonl(session: dict, speakers: tuple[str, str] | None = None) -> str:
    """One session in the Claude Code shape MemPalace's miner reads with both roles and the session's date.
    A LoCoMo session has two people: the first speaker is the user, the second the assistant, names in the text."""
    lines = []
    for number, turn in enumerate(session["turns"]):
        if speakers is None:
            role, text = turn["role"], turn["content"]
        else:
            role, text = ("user" if turn["role"] == speakers[0] else "assistant"), f"{turn['role']}: {turn['content']}"
        lines.append(json.dumps({"type": role, "message": {"role": role, "content": text},
                                 "timestamp": iso_date(session["date"]), "uuid": f"{session['sid']}-{number}",
                                 "sessionId": session["sid"]}))
    return "".join(line + "\n" for line in lines)


def load_bench(repo: Path, name: str) -> Any:
    path = Path(repo) / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"mempalace_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchArm:
    """LongMemEval, MemPalace's own retrieval functions; the reader gets each ranked session whole."""

    def __init__(self, bench: Any, *, mode: str) -> None:
        self.bench, self.mode = bench, mode

    def ingest(self, sessions: list[dict]) -> None:
        self.sessions = sessions
        self.by_sid = {s["sid"]: s for s in sessions}

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        entry = lme_entry(self.sessions, query, date)
        if self.mode == "raw":
            ranked, _, ids, _ = self.bench.build_palace_and_retrieve(entry, granularity="session", n_results=50)
        else:
            ranked, _, ids, _ = self.bench.build_palace_and_retrieve_hybrid_v4(entry, granularity="session",
                                                                               n_results=50, hybrid_weight=0.30)
        order = list(dict.fromkeys(ids[i] for i in ranked))
        return [Item(sid, self.by_sid[sid]["date"], tuple(self.by_sid[sid]["turns"]), _all_rounds(self.by_sid[sid]),
                     f"mempalace:{self.mode}") for sid in order]

    def close(self) -> None:
        pass


class ProductArm:
    """MemPalace as a user runs it: mine the sessions, search the palace, hand the reader the drawers found."""

    def __init__(self, mine: Any, search: Any, workdir: Path, *, n_results: int = 200,
                 speakers: tuple[str, str] | None = None) -> None:
        self.mine, self.search, self.n_results, self.speakers = mine, search, n_results, speakers
        self.folder = Path(tempfile.mkdtemp(prefix="palace-", dir=workdir))

    def ingest(self, sessions: list[dict]) -> None:
        self.by_sid = {s["sid"]: s for s in sessions}
        conversations = self.folder / "conversations"
        conversations.mkdir()
        for session in sessions:
            (conversations / f"{session['sid']}.jsonl").write_text(claude_code_jsonl(session, self.speakers))
        self.mine(convo_dir=str(conversations), palace_path=str(self.folder / "palace"), wing="eval", agent="adapter")

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        found = self.search(query, palace_path=str(self.folder / "palace"), n_results=self.n_results)
        if "error" in found:
            raise RuntimeError(f"MemPalace search failed: {found['error']}")
        items = []
        for hit in found["results"]:
            sid = Path(hit["source_path"]).stem
            session = self.by_sid[sid]
            items.append(Item(sid, session["date"], ({"role": "memory", "content": hit["text"]},), (),
                              f"mempalace:product:{hit.get('drawer_id', '')}"))
        return items

    def close(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("views", "repo", "out", "workdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--n-results", type=int, default=200, help="product arm: drawers asked for")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args(argv)
    Path(arguments.workdir).mkdir(parents=True, exist_ok=True)

    paths = list(sys.path)  # importing mempalace drops PYTHONPATH entries; this study's code must stay importable
    import chromadb
    import mempalace

    sys.path[:] = list(dict.fromkeys([*paths, *sys.path]))
    versions = {"mempalace": mempalace.__version__, "commit": COMMIT, "chromadb": chromadb.__version__}
    if arguments.arm == "product":
        from mempalace.convo_miner import mine_convos
        from mempalace.searcher import search_memories

        def make() -> ProductArm:
            return ProductArm(mine_convos, search_memories, Path(arguments.workdir), n_results=arguments.n_results)
    else:
        bench = load_bench(Path(arguments.repo), "longmemeval_bench")

        def make() -> BenchArm:
            return BenchArm(bench, mode="raw" if arguments.arm == "bench_raw" else "v4")
    counts = run_adapter(Path(arguments.views), Path(arguments.out), make, system=f"mempalace_{arguments.arm}",
                         code=f"studies/memory_bench/adapters/mempalace_arms.py:{arguments.arm}", versions=versions,
                         limit=arguments.limit)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
