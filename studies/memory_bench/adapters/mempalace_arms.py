"""MemPalace 3.11.0 (commit 602d796c), run in its own environment on the shared ingest file. Two arms, reported apart.

* **Benchmark arm** (what MemPalace's published numbers mean): its own benchmark functions, imported from the
  repository at the release commit. LongMemEval: ``build_palace_and_retrieve`` (raw) or
  ``build_palace_and_retrieve_hybrid_v4``, with ``n_results`` 50 as published; they index user turns only and rank
  sessions, and the reader then gets each ranked session whole. LoCoMo: ``LocomoArm``, its helpers rebuilt per
  conversation, as its runner does, at session or dialog level.
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

from studies.memory_bench.adapters.common import QueryFree, run_adapter
from studies.memory_bench.core import Item
from studies.memory_bench.lme import rounds

COMMIT = "602d796c699cabba2971982cd9dccbbf46abfe79"
LOCOMO_ARMS = {"locomo_raw_session": ("session", "raw"), "locomo_raw_dialog": ("dialog", "raw"),
               "locomo_hybrid_session": ("session", "hybrid")}
ARMS = ("bench_raw", "bench_v4", "product", *LOCOMO_ARMS)


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
    A LoCoMo session has two people: speaker_a is the user, speaker_b the assistant, names in the text, as Mem0's arm has it."""
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


class LocomoArm:
    """LoCoMo, MemPalace's own helpers rebuilt per conversation as its runner does, with no model. Its runner asks for
    the top ten (three times that for hybrid, then its own re-scoring). Those are kept exactly as published, and the
    rest of the ranking follows them, so a budget larger than ten sessions still has something to fill."""

    TOP = 10

    def __init__(self, bench: Any, chroma: Any, workdir: Path, *, granularity: str, mode: str) -> None:
        self.bench, self.chroma, self.granularity, self.mode = bench, chroma, granularity, mode
        self.folder = Path(tempfile.mkdtemp(prefix="palace-", dir=workdir))

    def ingest(self, sessions: list[dict]) -> None:
        self.sessions = sessions
        conversation: dict[str, Any] = {}
        for number, session in enumerate(sessions, 1):
            conversation[f"session_{number}"] = [
                {"speaker": turn["role"], "text": turn["content"], "dia_id": f"D{number}:{index}"}
                for index, turn in enumerate(session["turns"], 1)]
            conversation[f"session_{number}_date_time"] = session["date"]
        built = self.bench.load_conversation_sessions(conversation, {})
        corpus, ids, stamps = self.bench.build_corpus_from_sessions(built, granularity=self.granularity)
        client = self.chroma.PersistentClient(path=str(self.folder / "palace"))
        self.collection = client.create_collection("mempal_drawers")
        self.collection.add(documents=corpus, ids=[f"doc_{i}" for i in range(len(corpus))],
                            metadatas=[{"corpus_id": c, "timestamp": t, "room": "general"}
                                       for c, t in zip(ids, stamps, strict=True)])
        self.size = len(corpus)

    def _fused(self, query: str, found: dict) -> list[str]:
        """Its hybrid score, as its runner computes it."""
        bench = self.bench
        names = bench._person_names(query)
        lowered = {name.lower() for name in names}
        predicate = [word for word in bench._kw(query) if word not in lowered]
        quoted = bench._quoted_phrases(query)
        scored = []
        for meta, distance, document in zip(found["metadatas"][0], found["distances"][0], found["documents"][0], strict=True):
            fused = distance * (1.0 - 0.50 * bench._kw_overlap(predicate, document))
            boost = bench._quoted_boost(quoted, document)
            if boost > 0:
                fused *= 1.0 - 0.60 * boost
            boost = bench._name_boost(names, document)
            if boost > 0:
                fused *= 1.0 - 0.20 * boost
            scored.append((fused, meta["corpus_id"]))
        scored.sort(key=lambda row: row[0])
        return [corpus_id for _, corpus_id in scored]

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        pool = min(self.TOP * 3 if self.mode == "hybrid" else self.TOP, self.size)
        found = self.bench._query(self.collection, query, pool, "default")
        ids = self._fused(query, found) if self.mode == "hybrid" else [m["corpus_id"] for m in found["metadatas"][0]]
        if pool < self.size:
            whole = self.bench._query(self.collection, query, self.size, "default")
            seen = set(ids)
            ids += [m["corpus_id"] for m in whole["metadatas"][0] if m["corpus_id"] not in seen]
        return [self._item(corpus_id) for corpus_id in ids]

    def _item(self, corpus_id: str) -> Item:
        source = f"mempalace:locomo:{corpus_id}"
        whole = re.fullmatch(r"session_(\d+)", corpus_id)
        if whole:
            session = self.sessions[int(whole.group(1)) - 1]
            return Item(session["sid"], session["date"], tuple(session["turns"]), _all_rounds(session), source)
        number, index = re.fullmatch(r"D(\d+):(\d+)", corpus_id).groups()
        session = self.sessions[int(number) - 1]
        return Item(session["sid"], session["date"], (session["turns"][int(index) - 1],), (int(index) - 1,), source)

    def close(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)


class ProductArm:
    """MemPalace as a user runs it: mine the sessions, search the palace, hand the reader the drawers found. A drawer
    MemPalace folded from several sessions (identical text) is delivered once, and its source names the others."""

    def __init__(self, mine: Any, search: Any, workdir: Path, *, n_results: int = 1000, stack: Any = None) -> None:
        self.mine, self.search, self.n_results, self.stack = mine, search, n_results, stack
        self.folder = Path(tempfile.mkdtemp(prefix="palace-", dir=workdir))

    def ingest(self, sessions: list[dict]) -> None:
        self.by_sid = {s["sid"]: s for s in sessions}
        conversations = self.folder / "conversations"
        conversations.mkdir()
        for session in sessions:
            (conversations / f"{session['sid']}.jsonl").write_text(claude_code_jsonl(session, session.get("speakers")))
        self.mine(convo_dir=str(conversations), palace_path=str(self.folder / "palace"), wing="eval", agent="adapter")

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        found = self.search(query, palace_path=str(self.folder / "palace"), n_results=self.n_results)
        if "error" in found:
            raise RuntimeError(f"MemPalace search failed: {found['error']}")
        items = []
        for hit in found["results"]:
            sid = Path(hit["source_path"]).stem
            session = self.by_sid[sid]
            others = "".join(f"|also:{Path(copy['source_path']).stem}" for copy in hit.get("also_in") or [])
            items.append(Item(sid, session["date"], ({"role": "memory", "content": hit["text"]},), (),
                              f"mempalace:product:{hit.get('drawer_id', '')}{others}"))
        return items

    def memories(self) -> list[str]:
        """Its wake-up text, for the rule-loss cases, which give no question to search with."""
        stack = self.stack(palace_path=str(self.folder / "palace"), identity_path=str(self.folder / "identity.txt"))
        return [stack.wake_up(wing="eval")]

    def close(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("views", "repo", "out", "workdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--n-results", type=int, default=1000,
                        help="product arm: drawers asked for, more than any budget here can hold: at 200, LoCoMo ran short at 16,000")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th unit from the i-th")
    parser.add_argument("--query-free", action="store_true", help="product arm: its wake-up text, for the rule-loss cases")
    arguments = parser.parse_args(argv)
    shard = tuple(int(part) for part in arguments.shard.split("/"))
    Path(arguments.workdir).mkdir(parents=True, exist_ok=True)

    paths = list(sys.path)  # importing mempalace drops PYTHONPATH entries; this study's code must stay importable
    import chromadb
    import mempalace

    sys.path[:] = list(dict.fromkeys([*paths, *sys.path]))
    versions = {"mempalace": mempalace.__version__, "commit": COMMIT, "chromadb": chromadb.__version__}
    if arguments.arm == "product":
        from mempalace.convo_miner import mine_convos
        from mempalace.layers import MemoryStack
        from mempalace.searcher import search_memories

        def make() -> Any:
            arm = ProductArm(mine_convos, search_memories, Path(arguments.workdir), n_results=arguments.n_results,
                             stack=MemoryStack)
            return QueryFree(arm) if arguments.query_free else arm
    elif arguments.arm in LOCOMO_ARMS:
        bench = load_bench(Path(arguments.repo), "locomo_bench")
        granularity, mode = LOCOMO_ARMS[arguments.arm]

        def make() -> LocomoArm:
            return LocomoArm(bench, chromadb, Path(arguments.workdir), granularity=granularity, mode=mode)
    else:
        bench = load_bench(Path(arguments.repo), "longmemeval_bench")

        def make() -> BenchArm:
            return BenchArm(bench, mode="raw" if arguments.arm == "bench_raw" else "v4")
    label = f"mempalace_{arguments.arm}{'_free' if arguments.query_free else ''}"
    counts = run_adapter(Path(arguments.views), Path(arguments.out), make, system=label,
                         code=f"studies/memory_bench/adapters/mempalace_arms.py:{arguments.arm}", versions=versions,
                         limit=arguments.limit, shard=shard)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
