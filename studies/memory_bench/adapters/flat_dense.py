"""The shared meaning-based baseline: the same units as the keyword list, ranked by cosine similarity under one
local embedder, and like the keyword list it holds the first copy of identical text only. It runs on the shared ingest file in the
environment that holds the embedder.

    python -m studies.memory_bench.adapters.flat_dense --views views.jsonl \\
        --embedder sentence-transformers/all-MiniLM-L6-v2 --out rankings.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from studies.memory_bench.adapters.common import run_adapter
from studies.memory_bench.core import Item
from studies.memory_bench.lme import units_of


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


class FlatDenseArm:
    def __init__(self, embed: Callable[[list[str]], Sequence[Sequence[float]]], unit: str = "round") -> None:
        self.embed, self.unit = embed, unit

    def ingest(self, sessions: list[dict]) -> None:
        first: dict[str, Any] = {}
        for session in sessions:
            for unit in units_of(session, self.unit):
                first.setdefault(unit.text, unit)
        self.units = list(first.values())
        self.vectors = self.embed([unit.text for unit in self.units])

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        wanted = self.embed([query])[0]
        scored = [(_cosine(wanted, vector), number) for number, vector in enumerate(self.vectors)]
        scored.sort(key=lambda pair: (-pair[0], -pair[1]))  # a tie goes to the newest, as in Taste's search
        return [self.units[number].item() for _, number in scored]

    def close(self) -> None:
        pass


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("views", "embedder", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--unit", default="round", choices=("round", "session"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th unit from the i-th")
    arguments = parser.parse_args(argv)
    shard = tuple(int(part) for part in arguments.shard.split("/"))

    import sentence_transformers

    model = sentence_transformers.SentenceTransformer(arguments.embedder, device=arguments.device)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, batch_size=arguments.batch_size, normalize_embeddings=True,
                            show_progress_bar=False).tolist()

    versions = {"sentence_transformers": sentence_transformers.__version__, "embedder": arguments.embedder,
                "max_seq_length": model.max_seq_length, "unit": arguments.unit}
    counts = run_adapter(Path(arguments.views), Path(arguments.out), lambda: FlatDenseArm(embed, arguments.unit),
                         system="flat_dense", code="studies/memory_bench/adapters/flat_dense.py", versions=versions,
                         limit=arguments.limit, shard=shard)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
