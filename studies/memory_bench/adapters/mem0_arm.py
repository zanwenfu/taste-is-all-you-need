"""Mem0 open source 2.2.1, run in its own environment on the shared ingest file, with no route to a paid service.

One ``Memory`` per process, reset for every unit. Each round (a user turn and its replies; one turn in LoCoMo,
whose two speakers become user and assistant with their names in the text, as Mem0's own harness does) is one
``add`` with ``infer=True``: exactly one call to the served model, which writes the memories. A system line gives
the session's date, and the session's id, round and date go in the metadata, which every hit carries back. Search
asks for the whole ranked list (``threshold`` 0) of this unit's memories, and each memory reaches the reader as
written, tagged with its session and round.

The served model is asked for JSON; how often its reply parses is counted on every call, because a reply that
does not parse leaves nothing stored and the brief stops an arm whose parse rate is under 95%.

    MEM0_TELEMETRY=False python -m studies.memory_bench.adapters.mem0_arm --views views.jsonl --url URL \\
        --model NAME --embedder sentence-transformers/all-MiniLM-L6-v2 --dims 384 --out rankings.jsonl
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from studies.memory_bench.adapters.common import run_adapter
from studies.memory_bench.core import Item
from studies.memory_bench.lme import rounds

USER = "eval"  # one store per unit, reset in between, so one user id serves every unit


def mem0_config(url: str, model: str, embedder: str, dims: int, device: str) -> dict[str, Any]:
    return {
        "llm": {"provider": "vllm", "config": {"model": model, "vllm_base_url": url, "api_key": "EMPTY",
                                               "temperature": 0.0, "max_tokens": 2000, "top_p": 1.0}},
        "embedder": {"provider": "huggingface", "config": {"model": embedder, "embedding_dims": dims,
                                                           "model_kwargs": {"device": device}}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": "mem0_eval", "embedding_model_dims": dims,
                                                          "path": ":memory:", "on_disk": False}},
        "history_db_path": ":memory:",
    }


def _parses(text: str) -> bool:
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return isinstance(json.loads(body), dict)
    except json.JSONDecodeError:
        return False


class Counted:
    """Wraps the model behind Mem0 to count its calls and the replies that do not parse as one JSON object."""

    def __init__(self, llm: Any) -> None:
        self.llm, self.calls, self.unparsed = llm, 0, 0

    def generate_response(self, *args: Any, **kwargs: Any) -> Any:
        reply = self.llm.generate_response(*args, **kwargs)
        self.calls += 1
        self.unparsed += not _parses(reply if isinstance(reply, str) else json.dumps(reply))
        return reply

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)


class Mem0Arm:
    def __init__(self, memory: Any, *, top_k: int = 200) -> None:
        self.memory, self.top_k = memory, top_k

    def ingest(self, sessions: list[dict]) -> None:
        self.memory.reset()
        self.by_sid = {session["sid"]: session for session in sessions}
        people = list(dict.fromkeys(t["role"] for s in sessions for t in s["turns"] if t["role"] not in ("user", "assistant")))
        for session in sessions:
            for number, part in enumerate(rounds(session)):
                messages = [{"role": "system", "content": f"Session date: {session['date']}"}]
                for turn in part:
                    if turn["role"] in ("user", "assistant"):
                        messages.append({"role": turn["role"], "content": turn["content"]})
                    else:
                        role = "user" if people and turn["role"] == people[0] else "assistant"
                        messages.append({"role": role, "content": f"{turn['role']}: {turn['content']}"})
                self.memory.add(messages, user_id=USER, infer=True,
                                metadata={"session_id": session["sid"], "round": str(number),
                                          "session_date": session["date"]})

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        found = self.memory.search(query, filters={"user_id": USER}, top_k=self.top_k, threshold=0.0)
        items = []
        for hit in found["results"]:
            meta = hit.get("metadata") or {}
            sid = meta.get("session_id")
            if sid not in self.by_sid:
                raise ValueError(f"a memory came back with no session of this unit: {sid!r}")
            items.append(Item(sid, self.by_sid[sid]["date"], ({"role": "memory", "content": hit["memory"]},),
                              (int(meta["round"]),), f"mem0:{hit.get('id', '')}"))
        return items

    def close(self) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("views", "url", "model", "embedder", "out", "workdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th unit from the i-th")
    arguments = parser.parse_args(argv)
    shard = tuple(int(part) for part in arguments.shard.split("/"))

    if os.environ.get("MEM0_TELEMETRY") != "False":
        raise SystemExit("set MEM0_TELEMETRY=False before starting: Mem0 reads it once, at import")
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        os.environ.pop(key, None)
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"  # any accidental OpenAI client fails here, locally
    os.environ.setdefault("MEM0_DIR", str(Path(arguments.workdir) / "mem0"))

    import mem0
    from mem0 import Memory

    memory = Memory.from_config(mem0_config(arguments.url, arguments.model, arguments.embedder, arguments.dims,
                                            arguments.device))
    if type(memory.llm).__name__ != "VllmLLM" or type(memory.embedding_model).__name__ != "HuggingFaceEmbedding":
        raise SystemExit("Mem0 is not using the served model and the local embedder")
    if not getattr(memory.vector_store, "_has_bm25_slot", False):
        raise SystemExit("Mem0's keyword search is off; install fastembed")
    counted = Counted(memory.llm)
    memory.llm = counted
    versions = {"mem0ai": mem0.__version__, "embedder": arguments.embedder, "model": arguments.model}
    counts = run_adapter(Path(arguments.views), Path(arguments.out), lambda: Mem0Arm(memory, top_k=arguments.top_k),
                         system="mem0", code="studies/memory_bench/adapters/mem0_arm.py", versions=versions,
                         limit=arguments.limit, shard=shard)
    counts |= {"llm_calls": counted.calls, "llm_unparsed": counted.unparsed}
    Path(arguments.out).with_suffix(".counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
