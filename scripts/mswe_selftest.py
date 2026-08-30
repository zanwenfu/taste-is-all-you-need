"""Would the agent's own testing have caught what it broke?

    python scripts/mswe_selftest.py /root/mswe40_sonnet2_s1 /root/mswe40_sonnet2_s2 ...

The undercount says the final state hides what happened on the timeline. The
obvious objection is that it does not matter: a competent agent runs the
tests, sees the breakage, and fixes it -- the instrument is measuring a
transient the agent itself would have caught. That objection is checkable,
because mini-swe-agent's trajectory records every command it ran and every
output it saw, in order, and the shadow timeline says which command opened
each episode.

For every contamination episode this prints, from the onset command to the
end of the run:

  tested        the agent ran some test command while the test was broken
  covered       one of those commands would have executed *this* test
                (its output names the test, or it targeted the test's file
                or module -- or it ran the whole suite)
  seen          the covering command's output shows the failure, so the
                agent was told and continued anyway

A breakage that is never covered was not "caught and fixed"; it was invisible
to the agent for the rest of the run, and the final state -- the artifact the
benchmark grades -- is the only place it could have shown up.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

#: Command heads that mean "this ran tests". Deliberately generous: a false
#: positive here weakens our own claim, a false negative inflates it.
TEST_RUNNERS = (
    "pytest", "py.test", "runtests.py", "unittest", "django test", "manage.py test",
    "tox", "nosetests", "trial ", "make test", "bin/test", "./test", "run_tests",
)

FAILURE_MARKERS = ("FAILED", "FAIL:", "ERROR:", "assert", "Traceback", "=== FAILURES", "failed,")


def parse_probe(pid: str) -> tuple[str, list[str]]:
    """(test name, target tokens a command could name to run it).

    Handles both id styles SWE-bench carries: pytest node ids
    (``a/b/test_x.py::Klass::test_y``) and unittest ids as django emits them
    (``test_y (a.b.Klass)`` or ``a.b.Klass.test_y``).
    """
    pid = pid.strip()
    tokens: list[str] = []
    name = pid
    if "::" in pid:
        path, _, rest = pid.partition("::")
        name = rest.split("::")[-1]
        tokens += [path, Path(path).stem, Path(path).parent.as_posix()]
    elif (m := re.match(r"^(\S+)\s+\(([^)]+)\)$", pid)):
        name, dotted = m.group(1), m.group(2)
        parts = dotted.split(".")
        tokens += [dotted, ".".join(parts[:-1]), parts[0]]
    else:
        parts = pid.split(".")
        if len(parts) > 1:
            name = parts[-1]
            tokens += [".".join(parts[:-1]), parts[0], "/".join(parts[:-1])]
    tokens.append(name)
    return name, [t for t in dict.fromkeys(tokens) if t and t not in (".", "/")]


def is_test_command(command: str) -> bool:
    return any(r in command for r in TEST_RUNNERS)


def whole_suite(command: str) -> bool:
    """A test command with no positional target runs everything.

    Tokenise the pipeline segment that holds the runner, drop everything up
    to and including the runner token, then drop options (``-x``,
    ``--settings=...``) and shell operators. Anything left is a target, so
    the command is not a whole-suite run.
    """
    if not is_test_command(command):
        return False
    segment = re.split(r"\||>|;", command)[0]
    tokens = segment.split()
    for i, token in enumerate(tokens):
        if any(r.split()[0] in token for r in TEST_RUNNERS if r.split()):
            rest = tokens[i + 1 :]
            # `python -m pytest`, `manage.py test`: the module name itself is
            # not a target, so skip a bare runner word that follows.
            if rest and rest[0] in ("test", "pytest", "unittest"):
                rest = rest[1:]
            return not [t for t in rest if not t.startswith("-") and t not in ("&&", "||")]
    return False


def commands_of(traj: dict) -> list[str]:
    """The commands in the order ``TasteEnvironment`` numbered them (cmd-NNN)."""
    out: list[str] = []
    for message in traj.get("messages", []):
        for action in (message.get("extra", {}) or {}).get("actions", []) or []:
            out.append(action.get("command", ""))
    return out


def outputs_of(traj: dict) -> list[str]:
    out: list[str] = []
    for message in traj.get("messages", []):
        extra = message.get("extra", {}) or {}
        if "returncode" in extra:
            out.append(extra.get("raw_output") or "")
    return out


def seq_to_command(shadow_path: Path) -> dict[int, int]:
    """observation seq -> 1-based command number (from ``cmd-NNN`` step ids)."""
    mapping: dict[int, int] = {}
    for line in shadow_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        m = re.match(r"^cmd-(\d+)$", str(row.get("step_id") or ""))
        if m:
            mapping[int(row["seq"])] = int(m.group(1))
    return mapping


def analyse_cell(root: Path, instance: str, evidence: dict) -> list[dict]:
    cell = root / "runs" / instance / "MSWE" / "t1" / ".git" / "taste"
    traj_path, shadow_path = cell / "miniswe.traj.json", cell / "shadow.jsonl"
    if not (traj_path.exists() and shadow_path.exists()):
        return []
    traj = json.loads(traj_path.read_text())
    commands, outputs = commands_of(traj), outputs_of(traj)
    seqmap = seq_to_command(shadow_path)
    rows = []
    for ep in evidence.get("episodes") or []:
        name, tokens = parse_probe(str(ep.get("probe") or ""))
        onset_cmd = seqmap.get(int(ep.get("onset_seq") or 0), 0)
        rec_seq = ep.get("recovered_seq")
        rec_cmd = seqmap.get(int(rec_seq), len(commands)) if rec_seq else len(commands)
        # cmd-NNN is 1-based; commands is 0-based. Start at the breaking
        # command itself, since one command can both edit and test.
        window = range(max(0, onset_cmd - 1), len(commands))
        tested = covered = seen = False
        cover_kind = ""
        for i in window:
            command = commands[i] if i < len(commands) else ""
            output = outputs[i] if i < len(outputs) else ""
            if not is_test_command(command):
                continue
            tested = True
            if name and name in output:
                covered, cover_kind = True, "output names the test"
            elif any(t in command for t in tokens if len(t) > 3):
                covered, cover_kind = True, "command targets its module/file"
            elif whole_suite(command):
                covered, cover_kind = True, "whole suite"
            if covered:
                if name and name in output and any(k in output for k in FAILURE_MARKERS):
                    seen = True
                break
        rows.append({
            "instance": instance,
            "probe": ep.get("probe"),
            "onset_cmd": onset_cmd,
            "recovered_cmd": rec_cmd if rec_seq else None,
            "commands_broken": (rec_cmd - onset_cmd) if rec_seq else (len(commands) - onset_cmd),
            "persisted_to_end": rec_seq is None,
            "tested_after_onset": tested,
            "covered": covered,
            "cover_kind": cover_kind,
            "failure_shown_to_agent": seen,
            "commands_total": len(commands),
        })
    return rows


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows: list[dict] = []
    for spec in sys.argv[1:]:
        root = Path(spec)
        for f in sorted((root / "ledger" / "evidence").glob("*.json")):
            evidence = json.loads(f.read_text())
            if not evidence.get("episodes"):
                continue
            rows += analyse_cell(root, evidence["instance_id"], evidence)
    if not rows:
        print("no episodes found under", " ".join(sys.argv[1:]))
        return 0
    print(f"{'instance':32s} {'probe':52s} {'onset':>5s} {'broken':>6s} {'tested':>6s} {'covered':>7s} {'shown':>5s}  how")
    for r in rows:
        print(f"{r['instance']:32s} {str(r['probe'])[:52]:52s} {r['onset_cmd']:>5d} "
              f"{r['commands_broken']:>6d} {r['tested_after_onset']!s:>6s} {r['covered']!s:>7s} "
              f"{r['failure_shown_to_agent']!s:>5s}  {r['cover_kind']}")
    n = len(rows)
    print(f"\n{n} episode(s): "
          f"{sum(r['tested_after_onset'] for r in rows)} ran some test afterwards, "
          f"{sum(r['covered'] for r in rows)} ran one that would have executed the broken test, "
          f"{sum(r['failure_shown_to_agent'] for r in rows)} were shown the failure, "
          f"{sum(r['persisted_to_end'] for r in rows)} persisted to the final state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
