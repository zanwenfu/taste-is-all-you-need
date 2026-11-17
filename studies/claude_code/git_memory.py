"""One project memory, three designs, for real Claude Code (agent-os session, Nov 16 2026; the user's direction: structured git
memory, delivered when needed and searchable, against a flat markdown file dumped into every prompt).

A project memory is a list of items {"kind": "rule" | "folder" | "note" | "fact", "scope": folder, "file": path (notes only),
"text": ...}: the house rules that hold for every task, how to run each folder's tests, notes on the code (what each public function
and class is), and project facts that live only in memory (a decision, a limit, a name), written once and given identically to every
design, so the writer cannot explain a difference.

    flat      one CLAUDE.md holding everything; Claude Code reads it on every call (its own design)
    folders   Claude Code's path-scoped rules: CLAUDE.md holds the house rules; each folder's items go to .claude/rules/ with
              paths: frontmatter, which Claude Code loads when Claude reads or edits a matching file
    git       Taste's store (taste/memstore): one commit per item, its header naming what it is (kind, folder, file); the house rules
              go with every message, a folder's instructions and a file's notes when Claude first touches them, and everything is
              searchable with memory_search and memory_fetch

The store needs Python 3.11 and GitPython, and two of the task images have Python 3.9, so the store runs on the machine as a
service; the hooks and the tool server inside the task container reach it through a socket in the mounted memory folder, with the
standard library only on their side.

    python -m studies.claude_code.git_memory build --items items.json --out DIR
    python -m studies.claude_code.git_memory serve --memory DIR/git --socket DIR/git/memory.sock
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

HOUSE_RULES = (
    "Don't touch the tests unless the user asks.",
    "Start each hand-back with a line saying which files you changed.",
    "Keep hand-backs under 150 words.",
    "Use plain text in hand-backs: no headings, no bold, no bullet or numbered lists.",
    "Search the code with your Grep and Glob tools, not grep or find in the shell.",
    "Read files with your Read tool, not cat, head, tail or sed in the shell.",
    "Whenever you run Python, use python -B.",
    "End each hand-back with a line starting 'Tests run:' that lists the test commands you ran, or says none.",
)
SESSION = "project-memory"
BRANCH = "memory"
NOTES_PER_FILE = 25  # notes delivered for one touched file, at most
# Capture as the work happens (the fair test, the user's yes Nov 17 ~12:15 am ET): Claude saves with memory_save (the same writer as
# a CLAUDE.md it keeps); a small reader saves any standing rule Claude did not (source "reader"); every turn is recorded (kind "turn").
SAVE_KINDS = ("rule", "decision", "fact")
STANDING = ("rule", "decision")  # delivered with every message; facts, notes and turns are found by search, or arrive on touch if scoped
SAVE = "memory_save"
SAVE_DEFINITION = {
    "name": SAVE,
    "description": ("Save something to the project memory so it is kept for later tasks and later sessions, as one versioned "
                    "entry: a rule the user wants followed, a decision made, or a fact about the project. Rules and decisions are "
                    "given back to you with every message; facts are found later with memory_search."),
    "inputSchema": {"type": "object", "properties": {
        "kind": {"type": "string", "enum": list(SAVE_KINDS), "description": "rule, decision or fact"},
        "text": {"type": "string", "description": "What to keep, in one or two sentences; for a rule, the user's own words where possible."},
        "scope": {"type": "string", "description": "Optional: the folder it applies to, e.g. openlibrary/solr; empty for the whole project."}},
        "required": ["kind", "text"], "additionalProperties": False},
}
SIMILAR = 0.6  # a reader's rule this close (word overlap) to one already kept is not saved again


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def similar(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, len(wa | wb))


def with_rules(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The house rules first, then the given folder items and notes."""
    return [{"kind": "rule", "scope": "", "text": text} for text in HOUSE_RULES] + [i for i in items if i["kind"] != "rule"]


def _section(title: str, rows: list[str]) -> str:
    return f"## {title}\n\n" + "".join(f"- {row}\n" for row in rows) + "\n" if rows else ""


def flat_markdown(items: list[dict[str, Any]]) -> str:
    """Everything in one file, as a project's CLAUDE.md grows."""
    return ("# Project memory\n\n" + _section("House rules (every task)", [i["text"] for i in items if i["kind"] == "rule"])
            + _section("Running tests by folder", [i["text"] for i in items if i["kind"] == "folder"])
            + _section("Project notes", [i["text"] for i in items if i["kind"] == "fact"])
            + _section("Code notes", [i["text"] for i in items if i["kind"] == "note"]))


def _slug(scope: str) -> str:
    return scope.replace("/", "__") or "root"


def folder_rules(items: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    """Claude Code's path-scoped rules: (CLAUDE.md with the house rules, {".claude/rules/<name>.md": text}). A folder's test
    instruction covers everything under it; a folder's notes cover the files directly in it."""
    claude_md = "# Project memory\n\n" + _section("House rules (every task)", [i["text"] for i in items if i["kind"] == "rule"])
    files: dict[str, str] = {}
    for item in items:
        if item["kind"] == "folder":
            files[f".claude/rules/{_slug(item['scope'])}--tests.md"] = (
                f'---\npaths:\n  - "{item["scope"]}/**"\n---\n\n' + _section("Running tests", [item["text"]]))
    by_folder: dict[str, list[str]] = {}
    for item in items:
        if item["kind"] == "note":
            by_folder.setdefault(item["scope"], []).append(item["text"])
    for scope, rows in by_folder.items():
        files[f".claude/rules/{_slug(scope)}--notes.md"] = f'---\npaths:\n  - "{scope}/*"\n---\n\n' + _section("Code notes", rows)
    facts: dict[str, list[str]] = {}
    for item in items:
        if item["kind"] == "fact":
            facts.setdefault(item["scope"], []).append(item["text"])
    for scope, rows in facts.items():
        files[f".claude/rules/{_slug(scope)}--facts.md"] = f'---\npaths:\n  - "{scope}/**"\n---\n\n' + _section("Project notes", rows)
    return claude_md, files


def build_store(items: list[dict[str, Any]], folder: Path) -> list[dict[str, Any]]:
    """Taste's store with one commit per item, and the index read back from it (written to index.json beside the repository)."""
    from taste.memstore import CommitHeader, Store
    folder.mkdir(parents=True, exist_ok=True)
    store = Store.open(folder / "repo", SESSION)
    try:
        branch = store.branch(BRANCH, producer="claude-code-study")
        index = []
        for number, item in enumerate(items):
            where = item.get("file") or item["scope"] or "project"
            rel = f"memory/{item['kind']}/{where}/{number:05d}.md"
            branch.write(rel, item["text"] + "\n")
            words = [item["kind"], item["scope"]] + ([Path(item["file"]).name] if item.get("file") else [])
            header = CommitHeader(what=item["text"][:200], keywords=tuple(w for w in words if w),
                                  instructions=(item["text"],) if item["kind"] == "rule" else ())
            state = branch.checkpoint(f"project memory: {item['kind']}", header=header)
            index.append({"id": number, "kind": item["kind"], "scope": item["scope"], "file": item.get("file", ""),
                          "tests": item.get("tests", ""), "text": item["text"], "path": rel, "state": state.id})
    finally:
        store.close()
    (folder / "index.json").write_text(json.dumps(index) + "\n")
    return index


class Memory:
    """The git memory as the service answers it: the house rules and the items for touched files come from the index the store
    was built with (checked against the store on start), search and fetch from the store itself."""

    def __init__(self, folder: Path) -> None:
        from taste.memstore import Store
        self.folder = Path(folder)
        self.index = json.loads((self.folder / "index.json").read_text())
        self.store = Store.open(self.folder / "repo", SESSION)
        self.check()
        self.log = self.folder / "service.jsonl"
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()  # saves come one at a time: each is one commit, none is lost
        self.branch = None

    def save(self, kind: str, text: str, *, scope: str = "", source: str = "claude", allowed: tuple[str, ...] = SAVE_KINDS) -> dict[str, Any]:
        """One entry, committed on its own: the text as a file, a header naming its kind, folder and source. A reader's rule close
        to one already kept is not saved again; Claude's saves are always kept."""
        from taste.memstore import CommitHeader
        text, scope = " ".join(str(text or "").split())[:2000], str(scope or "").strip().strip("/")
        if kind not in allowed or not text:
            raise ValueError(f"kind must be one of {', '.join(allowed)} and text must not be empty")
        with self.write_lock:
            if source == "reader":
                close = [r for r in self.index if r["kind"] in STANDING and similar(r["text"], text) >= SIMILAR]
                if close:
                    return {"skipped": "already kept", "id": close[0]["id"]}
            if self.branch is None:
                self.branch = self.store.branch(BRANCH, producer="claude-code-study")
            number = len(self.index)
            rel = f"memory/{kind}/{scope or 'project'}/{number:05d}.md"
            self.branch.write(rel, text + "\n")
            header = CommitHeader(what=text[:200], keywords=tuple(w for w in (kind, scope, source) if w),
                                  instructions=(text,) if kind in STANDING else ())
            state = self.branch.checkpoint(f"project memory: {kind} saved by {source}", header=header)
            row = {"id": number, "kind": kind, "scope": scope, "file": "", "tests": "", "text": text, "path": rel, "state": state.id,
                   "source": source, "saved": time.time()}
            self.index.append(row)
            temp = self.folder / "index.json.tmp"
            temp.write_text(json.dumps(self.index) + "\n")
            temp.replace(self.folder / "index.json")
        return {"saved": number, "kind": kind}

    def check(self) -> None:
        """Every indexed item's text is in the store at its path, so the index cannot drift from the truth."""
        from taste.memstore import tools  # noqa: F401  (imported here so a broken install fails at start, not mid-run)
        for row in self.index[:: max(1, len(self.index) // 50)]:
            text = self.store.state(row["state"]).read(row["path"])
            if (text or "").rstrip("\n") != row["text"]:
                raise RuntimeError(f"index and store disagree at item {row['id']}")

    def rules(self) -> list[str]:
        """What is delivered with every message: the rules and decisions kept, oldest first, each text once."""
        seen, out = set(), []
        for row in self.index:
            if row["kind"] in STANDING and row["text"].lower() not in seen:
                seen.add(row["text"].lower())
                out.append(row["text"])
        return out

    def items_for(self, paths: list[str]) -> list[dict[str, Any]]:
        """The folder instructions whose folder holds a touched path, then the notes of each touched file (at most NOTES_PER_FILE
        each)."""
        found: list[dict[str, Any]] = []
        clean = [p.split("/app/", 1)[-1].lstrip("./") for p in paths if p]
        for row in self.index:  # a folder's instruction and its project facts, like a path-scoped rule covering the folder
            if row["kind"] in ("folder", "fact") and any(p == row["scope"] or p.startswith(row["scope"] + "/") for p in clean):
                found.append(row)
        for path in clean:
            found += [row for row in self.index if row["kind"] == "note" and row["file"] == path][:NOTES_PER_FILE]
        seen, unique = set(), []
        for row in found:
            if row["id"] not in seen:
                seen.add(row["id"])
                unique.append({"id": row["id"], "kind": row["kind"], "text": row["text"]})
        return unique

    def tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        from taste.memstore import tools
        if name == SAVE:
            try:
                extra = set(arguments) - {"kind", "text", "scope"}
                if extra:
                    raise ValueError(f"unknown arguments {sorted(extra)}")
                return json.dumps(self.save(arguments.get("kind", ""), arguments.get("text", ""), scope=arguments.get("scope", ""))), False
            except ValueError as error:
                return json.dumps({"error": str(error)}), True
        return tools.answer(self.store, name, arguments, kinds=("file", "message"))

    def definitions(self, saving: bool = False) -> list[dict[str, Any]]:
        from taste.memstore import tools
        found = [{"name": d["name"], "description": d["description"].replace("a record of this session", "the project memory"),
                  "inputSchema": d["parameters"]} for d in tools.definitions()]
        return found + ([SAVE_DEFINITION] if saving else [])

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "rules":
            reply: dict[str, Any] = {"rules": self.rules()}
        elif op == "items_for":
            reply = {"items": self.items_for(list(request.get("paths") or []))}
        elif op == "tool":
            text, failed = self.tool(str(request.get("name")), dict(request.get("arguments") or {}))
            reply = {"text": text, "failed": failed}
        elif op == "definitions":
            reply = {"tools": self.definitions(saving=bool(request.get("saving")))}
        elif op == "save":  # the hooks: the reader's rules and the record of each turn
            reply = self.save(str(request.get("kind")), str(request.get("text", "")), scope=str(request.get("scope", "")),
                              source=str(request.get("source", "hook")), allowed=(*SAVE_KINDS, "turn"))
        else:
            reply = {"error": f"unknown op {op!r}"}
        with self.lock, self.log.open("a") as log:
            log.write(json.dumps({"time": time.time(), "op": op, "name": request.get("name"), "reply_chars": len(json.dumps(reply))}) + "\n")
        return reply


def serve(memory: Memory, socket_path: Path) -> None:
    socket_path = Path(socket_path)
    if socket_path.exists():
        socket_path.unlink()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline()
            try:
                reply = memory.handle(json.loads(line))
            except Exception as error:  # a bad request is answered, never fatal to the service
                reply = {"error": f"{type(error).__name__}: {error}"[:500]}
            self.wfile.write((json.dumps(reply) + "\n").encode())

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    with Server(str(socket_path), Handler) as server:
        os.chmod(socket_path, 0o777)  # the container's user (root) and the runner's user both connect
        server.serve_forever()


def build(items_path: Path, out: Path, *, house_rules: bool = True) -> dict[str, Any]:
    """The three designs from one item list; ``house_rules`` False starts with none (the fair test: rules come during the work)."""
    loaded = json.loads(Path(items_path).read_text())
    items = with_rules(loaded) if house_rules else [i for i in loaded if i["kind"] != "rule"]
    out = Path(out)
    (out / "flat").mkdir(parents=True, exist_ok=True)
    (out / "flat" / "CLAUDE.md").write_text(flat_markdown(items))
    claude_md, files = folder_rules(items)
    (out / "folders").mkdir(parents=True, exist_ok=True)
    (out / "folders" / "CLAUDE.md").write_text(claude_md)
    for rel, text in files.items():
        (out / "folders" / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / "folders" / rel).write_text(text)
    index = build_store(items, out / "git")
    return {"items": len(items), "flat_chars": len(flat_markdown(items)), "folder_files": len(files),
            "largest_folder_file": max((len(t) for t in files.values()), default=0), "commits": len(index)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--items", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--no-house-rules", action="store_true", help="start with no rules (the fair test)")
    s = sub.add_parser("serve")
    s.add_argument("--memory", type=Path, required=True)
    s.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(args.items, args.out, house_rules=not args.no_house_rules)))
    else:
        serve(Memory(args.memory), args.socket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
