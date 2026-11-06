"""The claim ledger: which code each reported number really ran, checked by machine.

On Nov 3-4 2026 a test with no Taste code in it, no git and no search was reported as the memory under
study. Every case record of these papers now carries one of these records, and ``check`` refuses any
that could repeat that. Three names only, in every report, table and file name:

  real Taste run              planner, workers, monitor and memory store, running on Bedrock
  Taste memory, study-driven  code under taste/ for the store, search and fetch, driven by study code
  baseline, or stand-in       anything with no taste/ code in the measured path

This module imports nothing from taste, or asking whether a process imported the memory would always
say yes.

    python -m studies.memory_ledger results.jsonl [...]    every line needs a "ledger" field
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REAL_RUN = "real Taste run"
STUDY_DRIVEN = "Taste memory, study-driven"
BASELINE = "baseline"
STAND_IN = "stand-in"
NAMES = (REAL_RUN, STUDY_DRIVEN, BASELINE, STAND_IN)
TASTE_NAMES = (REAL_RUN, STUDY_DRIVEN)

# How the memory reached the model: the agent chose to call the tools, the harness asked with the
# question as the query, or stored text was put in front of the agent at every call.
MODES = ("agent_tools", "single_query", "delivered")


class ClaimRefused(ValueError):
    """A number that cannot be reported under the name it claims."""


def code_version() -> tuple[str, bool]:
    """(commit, whether taste/ or studies/ differ from it). A job with no git sets TASTE_COMMIT."""
    root = Path(__file__).resolve().parent.parent
    if os.environ.get("TASTE_COMMIT"):
        return os.environ["TASTE_COMMIT"], False

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", "taste", "studies"))


def _in_code_digest(name: str) -> bool:
    """Every Python file under taste/ and studies/, the job scripts, and the split files: what a run stands on."""
    parts = name.split("/")
    return ((parts[0] in ("taste", "studies") and name.endswith(".py"))
            or (parts[0] == "studies" and "jobs" in parts[1:-1])
            or name.startswith("data/splits/"))


def _in_ingest_digest(name: str) -> bool:
    """What decides what the long ingest stages write: the adapters, the header writer and their job scripts. A later fix to reading or judging leaves this alone."""
    return (name.startswith("studies/memory_bench/adapters/") and name.endswith(".py")) or name in (
        "studies/memory_bench/headers.py", "studies/memory_bench/jobs/mem0.sbatch", "studies/memory_bench/jobs/rank.sbatch",
        "studies/memory_bench/jobs/headers.sbatch")


def _digest_of(names: list[str], read: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(read(name))
        digest.update(b"\0")
    return digest.hexdigest()


def _files_on_disk(base: Path) -> list[str]:
    found = []
    for folder in ("taste", "studies", "data/splits"):
        for path in (base / folder).rglob("*"):
            name = path.relative_to(base).as_posix()
            if path.is_file() and "__pycache__" not in name and not name.endswith(".pyc"):
                found.append(name)
    return found


def code_digest(root: Path | None = None, *, wanted: Any = _in_code_digest) -> str:
    """A hash of the code actually on disk. ``TASTE_COMMIT`` only says what a job was told, so this is what shows that all of a run's stages ran one copy of the code,
    and, against ``digest_of_commit`` worked out where the repository is, that the copy is the commit's."""
    base = Path(root) if root else Path(__file__).resolve().parent.parent
    return _digest_of([name for name in _files_on_disk(base) if wanted(name)], lambda name: (base / name).read_bytes())


def ingest_digest(root: Path | None = None) -> str:
    """The same over what the long ingest stages depend on, so that a run's later stages may be on a later commit that left it alone."""
    return code_digest(root, wanted=_in_ingest_digest)


def digest_of_commit(commit: str, root: Path | None = None, *, wanted: Any = _in_code_digest) -> str:
    """``code_digest`` of a commit, from its git objects, with nothing unpacked."""
    base = Path(root) if root else Path(__file__).resolve().parent.parent

    def git(*args: str) -> bytes:
        return subprocess.run(["git", "-C", str(base), *args], capture_output=True, check=True).stdout

    names = git("ls-tree", "-r", "--name-only", commit, "taste", "studies", "data/splits").decode().splitlines()
    return _digest_of([name for name in names if wanted(name)], lambda name: git("cat-file", "blob", f"{commit}:{name}"))


def process_facts() -> dict[str, Any]:
    """What this process has loaded, read at the moment a record is made."""
    module = sys.modules.get("taste.memstore")
    return {"memstore_imported": module is not None, "memstore_file": getattr(module, "__file__", "") or ""}


def taste_record(store: Any, *, mode: str, name: str = STUDY_DRIVEN, commit: str | None = None,
                 dirty: bool | None = None) -> dict[str, Any]:
    """The ledger entry of one case that used a real store. Make it when the case ends."""
    if commit is None:
        commit, dirty = code_version()
    return {
        **process_facts(),
        "name": name,
        "mode": mode,
        "taste_commit": commit,
        "taste_dirty": bool(dirty),
        "store_path": str(store.root),
        "store_head": ",".join(sorted(state.id for state in store.heads().values())),
        "search_calls": store.memory_counts["search"],
        "fetch_calls": store.memory_counts["fetch"],
    }


def stand_in_record(name: str, *, code: str, commit: str | None = None, dirty: bool | None = None) -> dict[str, Any]:
    """The ledger entry of one case with no taste/ code in its measured path."""
    if name in TASTE_NAMES:
        raise ClaimRefused(f"{name!r} cannot be a stand-in")
    if name not in NAMES:
        raise ClaimRefused(f"{name!r} is not one of the allowed names")
    if commit is None:
        commit, dirty = code_version()
    return {**process_facts(), "name": name, "code": code, "taste_commit": commit, "taste_dirty": bool(dirty)}


def _check_one(record: dict[str, Any]) -> None:
    name = record.get("name")
    if name not in NAMES:
        raise ClaimRefused(f"{name!r} is not one of the allowed names")
    if not record.get("taste_commit"):
        raise ClaimRefused("no commit recorded")
    if record.get("taste_dirty"):
        raise ClaimRefused("the code had uncommitted changes")
    if name not in TASTE_NAMES:
        if not record.get("code"):
            raise ClaimRefused("a baseline must name its code; no code path")
        if record.get("search_calls"):
            raise ClaimRefused("a baseline that searched the memory is not a baseline")
        return
    if not record.get("memstore_imported"):
        raise ClaimRefused("this process did not import taste.memstore")
    if not record.get("store_path") or not record.get("store_head"):
        raise ClaimRefused("no store recorded")
    if record.get("mode") not in MODES:
        raise ClaimRefused(f"unknown mode {record.get('mode')!r}")
    for key in ("search_calls", "fetch_calls"):
        if type(record.get(key)) is not int or record[key] < 0:
            raise ClaimRefused(f"{key} is not a count")
    if record["mode"] == "single_query" and record["search_calls"] == 0:
        raise ClaimRefused("a single-query case never searched")


def check(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Refuse the first record that cannot be reported, or say what these records are."""
    for record in records:
        _check_one(record)
    return {"cases": len(records), "names": dict(Counter(r["name"] for r in records)), "looks": looks(records)}


def looks(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How often an agent that could choose did search. Zero searches count here and nowhere else."""
    chose = [r for r in records if r.get("mode") == "agent_tools"]
    looked = sum(1 for r in chose if r["search_calls"] > 0)
    return {"cases": len(chose), "looked": looked, "rate": looked / len(chose) if chose else None}


def measured_path(record: dict[str, Any]) -> str:
    return "taste/memstore" if record["name"] in TASTE_NAMES else record["code"]


def main(argv: list[str] | None = None) -> int:
    records = []
    for path in sys.argv[1:] if argv is None else argv:
        for number, line in enumerate(Path(path).read_text().splitlines(), 1):
            ledger = json.loads(line).get("ledger")
            if not isinstance(ledger, dict):
                print(f"{path}:{number}: no ledger field", file=sys.stderr)
                return 1
            records.append(ledger)
    try:
        print(json.dumps(check(records)))
    except ClaimRefused as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
