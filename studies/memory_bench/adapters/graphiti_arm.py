"""Graphiti (graphiti-core 0.31.2, the open-source engine behind Zep), run in its own environment on the shared ingest
file, with no route to a paid service and no database server.

Each unit gets a graph of its own: a Kuzu database (an in-process library, so no server, Docker or root) in a folder
of its own under ``--workdir``, deleted when the unit is done. Kuzu 0.11.3 carries its full-text search inside the
wheel, so nothing is fetched at run time. Each round (a user turn and its replies; one turn in LoCoMo) is one
``add_episode``, oldest first, as a message episode whose body is the round's "role: content" lines and whose reference
time is the session's date, which Graphiti reads as when the episode happened and dates its facts from. Search is
Graphiti's own hybrid search over facts (keyword and embedding lists merged by reciprocal rank, the recipe its
``search`` uses) with no reranking model. Each fact reaches the reader with its dates, as Zep shows facts, tagged with
the session and round of the first episode it came from.

Kept from Graphiti: every prompt, the whole ingest (entity extraction and dedupe, fact extraction, fact dedupe and
invalidation, entity summaries), its temperature, its JSON schema output mode, its retries and its search recipe.
Changed, and why:

- the model is the served one at ``--url`` for every step, through Graphiti's generic OpenAI-compatible client;
  Graphiti's small and medium sizes are one model here;
- every call says the reasoning effort the rest of the study uses, and its output cap is cut to ``--max-tokens``:
  Graphiti asks for up to 16,384 tokens on some steps, and a reasoning model can spend all of them thinking;
- the embedder is sentence-transformers all-MiniLM-L6-v2 on the local machine, as for every other arm, not OpenAI's;
- there is no reranking model; Graphiti's default is OpenAI's, a paid call per search, and its fact search does not
  use one;
- the search asks for ``--top-k`` facts with the embedding floor at ``--min-score`` (-1, so every fact is ranked).
  Graphiti's floor of 0.6 was set for OpenAI's embedder, and the list must hold more than any budget here can, as the
  Mem0 arm asks at threshold 0;
- the search runs on a copy of Graphiti's recipe: its ``search`` writes the limit into the shared recipe that its own
  ingest uses to find duplicate facts, so asking for 1,000 there would put up to 1,000 candidates into every later
  dedupe prompt;
- the full-text indexes are made by Kuzu's own maintenance step, since Graphiti's call to make them does nothing on
  Kuzu in this release, and both halves of the search are tried on a scratch graph before any model call.

Every model call is logged whole with its finish reason, whether it parsed and the step that asked for it, because a
reply that does not parse costs Graphiti its retries (up to four, 5 to 120 seconds apart) and then the episode.

    # Python 3.11 or later; the embedder must already be in the Hugging Face cache, since the run is offline
    pip install "graphiti-core[kuzu,sentence-transformers]==0.31.2" "kuzu==0.11.3" "python-dotenv>=1.2.0"
    python -m studies.memory_bench.adapters.graphiti_arm --views views.jsonl --url URL --model NAME \\
        --out rankings.jsonl --workdir /scratch/graphiti
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from studies.memory_bench.adapters.common import run_adapter
from studies.memory_bench.adapters.mempalace_arms import iso_date
from studies.memory_bench.core import Item
from studies.memory_bench.lme import rounds, turns_text

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
KEYS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
        "GROQ_API_KEY", "VOYAGE_API_KEY", "JEV_API_KEY")
PROMPT: contextvars.ContextVar[str | None] = contextvars.ContextVar("graphiti_prompt", default=None)


def reference_time(date: str) -> datetime:
    """A session's date as Graphiti's reference time. The benchmarks give no time zone, so it is read as UTC, which
    is what Graphiti assumes for a time without one. A date in any other form raises: Graphiti would otherwise date
    every fact by today."""
    return datetime.fromisoformat(iso_date(date)).replace(tzinfo=UTC)


def episode_body(part: list[dict]) -> str:
    """A round as Graphiti's message episodes are written: one "actor: content" line per turn."""
    return turns_text(part).rstrip("\n")


def _day(value: Any) -> str:
    return value.date().isoformat() if isinstance(value, datetime) else str(value)[:10]


def fact_text(fact: str, valid_at: Any, invalid_at: Any) -> str:
    """A fact with the dates Graphiti gives it: when it became true and, once contradicted, when it stopped."""
    if valid_at and invalid_at:
        return f"{fact} (from {_day(valid_at)} to {_day(invalid_at)})"
    if valid_at:
        return f"{fact} (from {_day(valid_at)})"
    if invalid_at:
        return f"{fact} (until {_day(invalid_at)})"
    return fact


def _parses(text: str) -> bool:
    body = re.sub(r"^```[\w-]*\s*|\s*```$", "", (text or "").strip())
    try:
        return isinstance(json.loads(body), dict)
    except json.JSONDecodeError:
        return False


class Record:
    """What a run's model calls and episodes came to, across every unit, with each reply and each lost episode
    written whole when given a file to write them to."""

    def __init__(self, replies: Path | None = None, failures: Path | None = None) -> None:
        self.replies, self.failures = replies, failures
        self.calls = self.unparsed = self.truncated = self.errors = self.episodes = self.episodes_failed = 0

    def reply(self, row: dict[str, Any]) -> None:
        self.calls += 1
        if "error" in row:
            self.errors += 1
        else:
            self.unparsed += not row["parsed"]
            self.truncated += row["finish_reason"] == "length"
        self._write(self.replies, row)

    def failure(self, row: dict[str, Any]) -> None:
        self.episodes_failed += 1
        self._write(self.failures, row)

    def counts(self) -> dict[str, int]:
        return {"llm_calls": self.calls, "llm_unparsed": self.unparsed, "llm_truncated": self.truncated,
                "llm_errors": self.errors, "episodes": self.episodes, "episodes_failed": self.episodes_failed}

    @staticmethod
    def _write(path: Path | None, row: dict[str, Any]) -> None:
        if path is not None:
            with path.open("a") as written:
                written.write(json.dumps(row, default=str) + "\n")


class Logged:
    """Stands where Graphiti's generic client expects an ``AsyncOpenAI``, of which it calls only
    ``chat.completions.create``. Every call also says ``effort`` and asks for at most ``cap`` output tokens, and every
    reply is logged with its finish reason, so a reply cut off while the model was still thinking can be told from
    one that came back malformed."""

    def __init__(self, client: Any, record: Record, *, effort: str | None, cap: int) -> None:
        self.client, self.record, self.effort, self.cap = client, record, effort, cap
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs: Any) -> Any:
        asked = kwargs.get("max_tokens")
        kwargs["max_tokens"] = min(asked, self.cap) if asked else self.cap
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        row = {"prompt": PROMPT.get(), "format": (kwargs.get("response_format") or {}).get("type"),
               "asked_max_tokens": asked, "max_tokens": kwargs["max_tokens"]}
        started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as error:
            self.record.reply({**row, "seconds": time.perf_counter() - started,
                               "error": f"{type(error).__name__}: {error}"})
            raise
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        self.record.reply({**row, "seconds": time.perf_counter() - started, "finish_reason": choice.finish_reason,
                           "parsed": _parses(text), "reply": text,
                           "usage": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens}
                           if usage else None})
        return response

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            await close()


def named(client_class: type) -> type:
    """Graphiti's client, with the name of the step that asks for each call kept for the log."""

    class Named(client_class):
        async def generate_response(self, *args: Any, **kwargs: Any) -> Any:
            token = PROMPT.set(kwargs.get("prompt_name"))
            try:
                return await super().generate_response(*args, **kwargs)
            finally:
                PROMPT.reset(token)

    return Named


class LocalEmbedder:
    """Graphiti's embedder interface over a local model: ``create`` embeds the first text it is given, as Graphiti's
    own embedders do, and ``create_batch`` embeds them all."""

    def __init__(self, encode: Callable[[list[str]], list[list[float]]]) -> None:
        self.encode = encode

    async def create(self, input_data: str | Iterable[str]) -> list[float]:
        texts = [input_data] if isinstance(input_data, str) else list(input_data)
        if not texts or not all(isinstance(text, str) for text in texts):
            raise TypeError("this embedder takes text only")
        return self.encode(texts[:1])[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return self.encode(list(input_data_list)) if input_data_list else []


class NoReranker:
    """Graphiti asks for a reranking model even when its search uses none; this one fails loudly if ever asked."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        raise RuntimeError("the Graphiti arm uses no reranking model")


def search_config(recipe: Any, limit: int, min_score: float) -> Any:
    """A copy of Graphiti's hybrid fact recipe, asking for ``limit`` facts with the embedding floor at ``min_score``.
    The shared recipe is left as it is, because Graphiti's ingest uses it too."""
    config = recipe.model_copy(deep=True)
    config.limit = limit
    config.edge_config.sim_min_score = min_score
    return config


class GraphitiGraph:
    """One unit's graph: Graphiti over a Kuzu database at ``path``, with model clients made for this unit's event
    loop, since an async HTTP client cannot be carried from one loop to the next."""

    def __init__(self, path: Path, *, url: str, model: str, effort: str | None, cap: int, output_mode: str,
                 encode: Callable[[list[str]], list[list[float]]], record: Record, top_k: int,
                 min_score: float) -> None:
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.client import CrossEncoderClient
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.client import EmbedderClient
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
        from openai import AsyncOpenAI

        self.transport = Logged(AsyncOpenAI(api_key="EMPTY", base_url=url), record, effort=effort, cap=cap)
        llm = named(OpenAIGenericClient)(LLMConfig(api_key="EMPTY", model=model, base_url=url), client=self.transport,
                                         max_tokens=cap, structured_output_mode=output_mode)
        embedder = type("LocalEmbedderClient", (LocalEmbedder, EmbedderClient), {})(encode)
        reranker = type("NoRerankerClient", (NoReranker, CrossEncoderClient), {})()
        self.driver = KuzuDriver(db=str(path))
        self.graphiti = Graphiti(graph_driver=self.driver, llm_client=llm, embedder=embedder, cross_encoder=reranker)
        self.config = search_config(EDGE_HYBRID_SEARCH_RRF, top_k, min_score)

    async def setup(self) -> None:
        """Graphiti's ``build_indices_and_constraints`` goes to the Kuzu driver's, which does nothing in 0.31.2, so
        the full-text indexes its keyword search reads would never be made (getzep/graphiti#1258). Kuzu's own
        maintenance step makes them."""
        await self.driver.graph_ops.build_indices_and_constraints(self.driver)

    async def probe(self) -> None:
        """Each half of the search, alone, must find a fact stored after the indexes were made. No model is called."""
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EntityNode
        from graphiti_core.search.search_config import EdgeSearchMethod

        when = datetime(2023, 5, 20, 2, 21, tzinfo=UTC)
        user, car = EntityNode(name="user", group_id=""), EntityNode(name="car", group_id="")
        for node in (user, car):
            await node.generate_name_embedding(self.graphiti.embedder)
            await node.save(self.driver)
        fact = EntityEdge(group_id="", source_node_uuid=user.uuid, target_node_uuid=car.uuid, created_at=when,
                          name="PARKED", fact="The user parked the car on level three of the garage", episodes=[],
                          valid_at=when)
        await fact.generate_embedding(self.graphiti.embedder)
        await fact.save(self.driver)
        for method in (EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity):
            config = self.config.model_copy(deep=True)
            config.edge_config.search_methods = [method]
            found = (await self.graphiti.search_("where did I park the car", config=config)).edges
            if [edge.uuid for edge in found] != [fact.uuid]:
                raise RuntimeError(f"Graphiti's {method.value} search on Kuzu does not find a stored fact")

    async def add(self, name: str, body: str, description: str, when: datetime) -> str:
        from graphiti_core.nodes import EpisodeType

        added = await self.graphiti.add_episode(name=name, episode_body=body, source_description=description,
                                                reference_time=when, source=EpisodeType.message)
        return added.episode.uuid

    async def search(self, query: str) -> list[Any]:
        return (await self.graphiti.search_(query, config=self.config)).edges

    async def close(self) -> None:
        try:
            await self.graphiti.close()
            await self.transport.close()
        finally:
            for part in (self.driver.client, self.driver.db):  # Graphiti leaves Kuzu's to the garbage collector
                close = getattr(part, "close", None)
                if close is not None:
                    close()


class GraphitiArm:
    """``open_graph(path)`` gives a fresh graph stored at ``path``. Each unit runs in an event loop of its own, opened
    at ``ingest`` and closed at ``close``. An episode Graphiti cannot add is logged and skipped, as Mem0's arm loses a
    round whose reply does not parse, until more than ``max_failed`` of the unit's rounds are lost; then the unit
    fails, so a broken setup stops at its first unit."""

    def __init__(self, open_graph: Callable[[Path], Any], workdir: Path, *, record: Record | None = None,
                 max_failed: float = 0.05) -> None:
        self.open_graph, self.workdir, self.max_failed = open_graph, Path(workdir), max_failed
        self.record = record or Record()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.graph: Any = None
        self.folder: Path | None = None

    def ingest(self, sessions: list[dict]) -> None:
        self.by_sid = {session["sid"]: session for session in sessions}
        self.episodes: dict[str, tuple[str, int]] = {}
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.folder = Path(tempfile.mkdtemp(prefix="graphiti-", dir=self.workdir))

        async def opened() -> Any:  # made inside the loop, so whatever binds to a loop binds to this one
            return self.open_graph(self.folder / "graph.kuzu")

        self.graph = self.loop.run_until_complete(opened())
        self.loop.run_until_complete(self.graph.setup())
        total, lost = sum(len(rounds(session)) for session in sessions), 0
        for session in sorted(sessions, key=lambda session: reference_time(session["date"])):
            sid, when = session["sid"], reference_time(session["date"])
            for number, part in enumerate(rounds(session)):
                try:
                    uuid = self.loop.run_until_complete(
                        self.graph.add(f"{sid} round {number}", episode_body(part), f"chat session {sid}", when))
                except Exception as error:
                    lost += 1
                    self.record.failure({"session": sid, "round": number, "error": f"{type(error).__name__}: {error}"})
                    if lost > self.max_failed * total:
                        raise RuntimeError(f"Graphiti lost {lost} of this unit's {total} rounds") from error
                    continue
                self.episodes[uuid] = (sid, number)
                self.record.episodes += 1

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        items = []
        for edge in self.loop.run_until_complete(self.graph.search(query)):
            source = next((self.episodes[uuid] for uuid in edge.episodes if uuid in self.episodes), None)
            if source is None:
                raise ValueError(f"a fact came back from no episode of this unit: {edge.uuid}")
            sid, number = source
            items.append(Item(sid, self.by_sid[sid]["date"],
                              ({"role": "memory", "content": fact_text(edge.fact, edge.valid_at, edge.invalid_at)},),
                              (number,), f"graphiti:{edge.uuid}"))
        return items

    def close(self) -> None:
        try:
            if self.graph is not None:
                self.loop.run_until_complete(self.graph.close())
        finally:
            if self.loop is not None:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.run_until_complete(self.loop.shutdown_default_executor())
                self.loop.close()
                asyncio.set_event_loop(None)
            if self.folder is not None:
                shutil.rmtree(self.folder)


def check_search(open_graph: Callable[[Path], Any], workdir: Path) -> None:
    """A half of the search that finds nothing fails quietly, so each is tried once on a scratch graph before any
    model call, as Mem0's arm checks its search parts."""
    folder = Path(tempfile.mkdtemp(prefix="graphiti-probe-", dir=workdir))
    loop = asyncio.new_event_loop()

    async def probe() -> None:
        graph = open_graph(folder / "graph.kuzu")
        try:
            await graph.setup()
            await graph.probe()
        finally:
            await graph.close()

    try:
        loop.run_until_complete(probe())
    finally:
        loop.close()
        shutil.rmtree(folder)


def _scrub() -> None:
    for key in KEYS:
        os.environ.pop(key, None)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("views", "url", "model", "out", "workdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--embedder", default=EMBEDDER)
    parser.add_argument("--dims", type=int, default=384)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=1000, help="more facts than any budget here can hold")
    parser.add_argument("--min-score", type=float, default=-1.0,
                        help="the embedding floor of the search; Graphiti's default is 0.6, -1 ranks every fact")
    parser.add_argument("--max-tokens", type=int, default=4096, help="the output cap on every call")
    parser.add_argument("--effort", default="low", help="the model's reasoning effort, on every call")
    parser.add_argument("--output-mode", choices=("json_schema", "json_object"), default="json_schema",
                        help="Graphiti's default asks the server to enforce the schema; json_object puts it in the prompt")
    parser.add_argument("--max-failed", type=float, default=0.05, help="the share of a unit's rounds it may lose")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th unit from the i-th")
    parser.add_argument("--probe-only", action="store_true",
                        help="build a scratch graph, check both halves of the search, and stop: no model call, no views read")
    arguments = parser.parse_args(argv)
    shard = tuple(int(part) for part in arguments.shard.split("/"))

    os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"  # Graphiti loads a .env on import; python-dotenv 1.2 honours this
    os.environ["EMBEDDING_DIM"] = str(arguments.dims)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the embedder must already be on disk
    _scrub()
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"  # any accidental OpenAI client fails here, locally

    import warnings
    from importlib.metadata import version

    warnings.filterwarnings("ignore", message="The Kuzu backend is deprecated")
    import graphiti_core  # noqa: F401
    import kuzu
    from sentence_transformers import SentenceTransformer

    _scrub()  # in case an older python-dotenv loaded a .env all the same

    embedder = SentenceTransformer(arguments.embedder, device=arguments.device)
    if embedder.get_sentence_embedding_dimension() != arguments.dims:
        raise SystemExit(f"{arguments.embedder} does not give {arguments.dims} dimensions")

    def encode(texts: list[str]) -> list[list[float]]:
        return embedder.encode(texts, convert_to_numpy=True).tolist()

    out, workdir = Path(arguments.out), Path(arguments.workdir)
    out.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    record = Record(out.with_suffix(".replies.jsonl"), out.with_suffix(".failed.jsonl"))
    versions = {"graphiti_core": version("graphiti-core"), "kuzu": kuzu.__version__,
                "sentence_transformers": version("sentence-transformers"), "embedder": arguments.embedder,
                "model": arguments.model, "max_tokens": arguments.max_tokens, "effort": arguments.effort,
                "output_mode": arguments.output_mode, "top_k": arguments.top_k, "min_score": arguments.min_score,
                "search": "EDGE_HYBRID_SEARCH_RRF", "max_failed": arguments.max_failed}

    def open_graph(path: Path) -> GraphitiGraph:
        return GraphitiGraph(path, url=arguments.url, model=arguments.model, effort=arguments.effort,
                             cap=arguments.max_tokens, output_mode=arguments.output_mode, encode=encode,
                             record=record, top_k=arguments.top_k, min_score=arguments.min_score)

    def make() -> GraphitiArm:
        return GraphitiArm(open_graph, workdir, record=record, max_failed=arguments.max_failed)

    check_search(open_graph, workdir)
    if arguments.probe_only:
        print(json.dumps({"probe": "ok", **versions}))
        return 0
    counts = run_adapter(Path(arguments.views), out, make, system="graphiti",
                         code="studies/memory_bench/adapters/graphiti_arm.py", versions=versions,
                         limit=arguments.limit, shard=shard)
    counts |= record.counts()
    out.with_suffix(".counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
