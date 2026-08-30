"""Would the agent's own testing have caught what it broke?

    python scripts/mswe_selftest.py /root/mswe40_sonnet2_s1 [more roots ...]
    python scripts/mswe_selftest.py --json out.json <roots>

The undercount says the final state hides what happened on the timeline. The
standing objection is that it does not matter: a competent agent runs the
tests, sees the breakage and fixes it, so the instrument is measuring a
transient the agent itself would have caught. mini-swe-agent's trajectory
records every command and every output in order, and the shadow timeline
says which command opened and which closed each episode, so the objection is
checkable rather than arguable.

Within the window the test was actually broken -- from the breaking command
up to, but not including, the command that repaired it -- each episode gets
three labels:

  tested    a test runner ran, and its output proves it ran (a crashed
            runner that printed "No module named pytest" did not test)
  covered   one of those runs would have executed *this* test: its output
            names the test, or it targeted the test's module/file (by
            label boundary, honouring -k/-m/--deselect and explicit node
            ids), or it ran the whole suite
  shown     a covering run displayed *this* test failing, so the agent was
            told and kept working

Three rules earn their keep, each from a real miscount found by auditing
these labels against the raw trajectories by hand:

* the window ends at recovery. Scanning to the end of the run credits test
  runs against an already-fixed tree, which cannot have told the agent
  anything: unbounded, coverage was 16/16 -- a statement with no content.
* coverage is decided on argv, not on substrings of the command text. A run
  of two sibling modules (``runtests.py queries.test_query
  queries.test_qs_combinators``) contains the string "queries" and executes
  60 tests, none of them the probe in ``queries/tests.py``.
* every covering run is inspected, not just the first. The first covering
  command is often a crashed runner; the failure the agent was actually
  shown appears eleven commands later.

An episode that is *not* covered was never executed by the agent's own
testing while it was broken. That is the interesting cell of the table: not
an agent ignoring red output, but a regression its test commands never ran.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

#: Tokens that make a command a test run. Matched on argv words (or as an
#: explicit call form), never as bare substrings: ``py.test`` inside
#: ``numpy.testing`` is not a test run, and ``src/_pytest/python.py`` in a
#: ``sed`` command is not either.
RUNNER_WORDS = ("pytest", "py.test", "unittest", "nosetests", "trial", "tox", "runtests.py", "run_tests.py")
RUNNER_CALLS = ("sympy.test(", "pytest.main(", "unittest.main(", "manage.py test", "django test", "make test")

#: Output that proves a runner actually started. Without one of these the
#: command may have died on an import or a collection error.
RAN_MARKERS = (
    "collected ", "Ran ", "passed", "failed", "error", "PASSED", "FAILED", "ERROR",
    "test session starts", "OK", "no tests ran", "deselected", "s [", "% ]",
)
#: Output that proves it did not.
DEAD_MARKERS = (
    "No module named", "command not found", "Error while finding module specification",
    "can't open file", "ImportError while loading conftest", "No such file or directory",
)
FAILURE_MARKERS = ("FAILED", "FAIL:", "ERROR:", "ERROR ", "AssertionError", "Traceback")


def parse_probe(pid: str) -> tuple[str, list[str]]:
    """(test name, targets that would select it).

    Handles the id styles SWE-bench carries: pytest node ids
    (``a/b/test_x.py::Klass::test_y``), unittest ids as django prints them
    (``test_y (a.b.Klass)``), dotted ids (``a.b.Klass.test_y``) and bare
    function names (sympy). A bare name yields no target, which is reported
    rather than guessed -- see ``locatable``.

    Exactly one target is returned, the most specific one, and matching is
    by label boundary in both directions (see ``_label_match``). Emitting the
    probe's package as well would re-create the miscount this replaced: a run
    of ``queries.test_query`` contains the label ``queries`` and executes
    sixty tests, none of them the probe in ``queries/tests.py``.
    """
    pid = pid.strip()
    targets: list[str] = []
    name = pid
    if "::" in pid:
        path, _, rest = pid.partition("::")
        name = rest.split("::")[-1]
        targets.append(path)
    elif (m := re.match(r"^(\S+)\s+\(([^)]+)\)$", pid)):
        name, dotted = m.group(1), m.group(2)
        targets.append(dotted)
    elif "." in pid:
        parts = pid.split(".")
        name = parts[-1]
        targets.append(".".join(parts[:-1]))
    return name, [t for t in targets if t and t not in (".", "/")]


def locatable(pid: str) -> bool:
    """Can a command target this test by anything other than naming it?"""
    return bool(parse_probe(pid)[1])


def _words(command: str) -> list[str]:
    """argv-ish words of the command, quoting errors tolerated."""
    try:
        return shlex.split(command, comments=False, posix=True)
    except ValueError:
        return command.split()


def is_test_command(command: str) -> bool:
    if any(call in command for call in RUNNER_CALLS):
        return True
    # An argv word whose basename is a runner: `pytest`, `python -m pytest`,
    # `python tests/runtests.py`. Never a bare substring -- `py.test` inside
    # `numpy.testing`, or `src/_pytest/python.py` in a sed, is not a test run.
    return any(word.rsplit("/", 1)[-1] in RUNNER_WORDS for word in _words(command))


def _runner_index(words: list[str]) -> int:
    for i, word in enumerate(words):
        if word.rsplit("/", 1)[-1] in RUNNER_WORDS:
            return i
    return -1


def targets_of(command: str) -> list[str]:
    """Positional targets after the runner: what it was told to run.

    An empty list means the runner was given no target, i.e. the whole
    suite. Options and their values are dropped; so are the shell operators
    that follow the runner's own pipeline segment.
    """
    segment = re.split(r"\||;|&&|>", command)
    words: list[str] = []
    for part in segment:
        if is_test_command(part):
            words = _words(part)
            break
    if not words:
        words = _words(command)
    i = _runner_index(words)
    if i < 0:
        return []
    rest = words[i + 1 :]
    if rest and rest[0] in ("test", "pytest", "unittest"):  # `manage.py test`, `python -m unittest`
        rest = rest[1:]
    out: list[str] = []
    skip = False
    for word in rest:
        if skip:
            skip = False
            continue
        if word.startswith("-"):
            # options that take a separate value
            if word in ("-k", "-m", "-p", "--deselect", "--ignore", "-n", "--settings", "-v"):
                skip = word not in ("-v",)
            continue
        out.append(word)
    return out


def selectors_of(command: str) -> dict[str, list[str]]:
    """``-k``/``-m`` expressions and explicit ``::`` node ids.

    Read only from the words *after* the runner: in ``python -m pytest`` the
    ``-m`` belongs to the interpreter, and reading it as a marker filter
    made every whole-suite run look like a filtered one.
    """
    words = _words(command)
    runner = _runner_index(words)
    words = words[runner + 1 :] if runner >= 0 else words
    out: dict[str, list[str]] = {"k": [], "m": [], "nodes": []}
    for i, word in enumerate(words):
        if word in ("-k", "-m") and i + 1 < len(words):
            out["k" if word == "-k" else "m"].append(words[i + 1])
        elif word.startswith("-k") and len(word) > 2:
            out["k"].append(word[2:])
        elif "::" in word:
            out["nodes"].append(word)
    return out


def _selector_admits(name: str, selectors: dict[str, list[str]]) -> bool:
    """Would the run's own filters have let this test through?

    Conservative: an expression we cannot read is treated as excluding, so
    an unreadable filter never inflates coverage.
    """
    if selectors["nodes"]:
        return any(name and name in node for node in selectors["nodes"])
    for expression in selectors["k"] + selectors["m"]:
        terms = [t for t in re.split(r"\s+|\(|\)", expression) if t and t not in ("and", "or", "not")]
        if not any(name and (t in name or name in t) for t in terms):
            return False
    return True


def _label_match(target: str, probe_target: str) -> bool:
    """Does a run target select the probe, on a label boundary?

    Both directions are allowed and both are meaningful: the target may be
    an ancestor of the probe (``queries`` selects ``queries.tests.UnionTests``)
    or a descendant of it (``queries.tests.UnionTests.test_x`` selects the
    probe's own test). A *sibling* matches neither, which is the whole point.
    """
    target, probe_target = target.strip("./"), probe_target.strip("./")
    if not target or not probe_target:
        return False
    if target == probe_target:
        return True
    for sep in (".", "/"):
        if target.startswith(probe_target + sep) or probe_target.startswith(target + sep):
            return True
    # A directory target against a dotted module, e.g. `tests/delete/` for
    # `delete.tests.FastDeleteTests`: compare in one alphabet.
    flat_target, flat_probe = target.replace("/", "."), probe_target.replace("/", ".")
    return flat_target == flat_probe or flat_probe.startswith(flat_target + ".") or flat_target.startswith(flat_probe + ".")


def ran_anything(output: str) -> bool:
    if any(dead in output for dead in DEAD_MARKERS) and not any(m in output for m in ("collected ", "Ran ")):
        return False
    return any(marker in output for marker in RAN_MARKERS)


def names_failure(output: str, name: str) -> bool:
    """Does the output show *this* test failing, on one line?"""
    if not name:
        return False
    for line in output.splitlines():
        if name in line and any(marker in line for marker in FAILURE_MARKERS):
            return True
    return False


#: The run stopped at the first failure (or after a few), so tests ordered
#: after the stop were never reached.
EARLY_STOP = ("-x", "--exitfirst", "--maxfail")
#: The output we can see is not the output the agent saw in full.
TRUNCATORS = ("| tail", "| head", "|tail", "|head", "| grep", "|grep")


def covers(command: str, output: str, name: str, probe_targets: list[str]) -> tuple[str, str]:
    """``yes`` / ``no`` / ``unknown`` -- would this run have executed the probe?

    ``unknown`` is a real answer and the honest one more often than it looks.
    A run with ``-x`` that failed on another test stopped before reaching
    this one; a run piped through ``tail -40`` shows a window, not a result.
    Coding either as "ran it and the agent still was not told" would be a
    strong claim built on absent evidence -- the seaborn cell in the GPT
    sweep is exactly that shape, and it accounted for 33 of 48 apparent
    coverages before this existed.
    """
    if name and name in output:
        return "yes", "output names the test"
    selectors = selectors_of(command)
    if not _selector_admits(name, selectors):
        return "no", ""
    explicit = selectors["k"] or selectors["m"] or selectors["nodes"]
    # Something narrowed the run and we cannot see what (a config-level
    # deselect, a plugin): refuse coverage unless the output names the test.
    if not explicit and re.search(r"\b\d+ deselected", output) and name not in output:
        return "no", ""
    targets = targets_of(command)
    hit = "whole suite" if not targets else next(
        (f"targeted {t}" for t in targets if any(_label_match(t, p) for p in probe_targets)), "")
    if not hit:
        return "no", ""
    words = _words(command)
    stopped_early = any(w in EARLY_STOP or w.startswith("--maxfail") for w in words)
    if stopped_early and re.search(r"\b\d+ failed|\bFAILURES\b|\berror\b", output):
        return "unknown", hit + ", but the run stopped at an earlier failure"
    if any(t in command for t in TRUNCATORS):
        return "unknown", hit + ", but its output was truncated"
    return "yes", hit


def commands_of(traj: dict) -> list[str]:
    return [a.get("command", "") for m in traj.get("messages", [])
            for a in (m.get("extra", {}) or {}).get("actions", []) or []]


def outputs_of(traj: dict) -> list[str]:
    return [(m.get("extra", {}) or {}).get("raw_output") or ""
            for m in traj.get("messages", []) if "returncode" in (m.get("extra", {}) or {})]


def seq_to_command(shadow_path: Path) -> dict[int, int]:
    """observation seq -> 1-based command number (``cmd-NNN`` step ids)."""
    mapping: dict[int, int] = {}
    for line in shadow_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (m := re.match(r"^cmd-(\d+)$", str(row.get("step_id") or ""))):
            mapping[int(row["seq"])] = int(m.group(1))
    return mapping


def analyse_episode(ep: dict, commands: list[str], outputs: list[str], seqmap: dict[int, int]) -> dict:
    probe = str(ep.get("probe") or "")
    name, probe_targets = parse_probe(probe)
    onset_cmd = seqmap.get(int(ep.get("onset_seq") or 0), 0)
    rec_seq = ep.get("recovered_seq")
    rec_cmd = seqmap.get(int(rec_seq), 0) if rec_seq else 0
    # The window the test was actually broken: from the breaking command
    # (one command can both edit and test) up to, not including, the repair.
    start = max(0, onset_cmd - 1)
    end = (rec_cmd - 1) if rec_cmd else len(commands)
    tested, covered, shown = False, "no", "no"
    cover_kind, first_cover, first_shown, runs = "", None, None, 0
    for i in range(start, min(end, len(commands))):
        command = commands[i]
        output = outputs[i] if i < len(outputs) else ""
        if not is_test_command(command) or not ran_anything(output):
            continue
        tested, runs = True, runs + 1
        verdict, kind = covers(command, output, name, probe_targets)
        if verdict == "no":
            continue
        # yes beats unknown beats no, and the first run of the best kind wins.
        if verdict == "yes" and covered != "yes":
            covered, cover_kind, first_cover = "yes", kind, i + 1
        elif verdict == "unknown" and covered == "no":
            covered, cover_kind, first_cover = "unknown", kind, i + 1
        if names_failure(output, name) and first_shown is None:
            shown, first_shown = "yes", i + 1
        elif verdict == "unknown" and shown == "no":
            shown = "unknown"
    if covered == "yes" and shown == "no":
        # The run executed it and named no failure. Either the agent was
        # shown a pass (so our replay and its run disagree) or the output
        # simply did not list it; both are worth seeing, neither is "told".
        shown = "no"
    return {
        "probe": probe,
        "locatable": bool(probe_targets),
        "onset_cmd": onset_cmd,
        "recovered_cmd": rec_cmd or None,
        "commands_broken": (rec_cmd - onset_cmd) if rec_cmd else (len(commands) - onset_cmd),
        "persisted_to_end": rec_seq is None,
        "test_runs_in_window": runs,
        "tested": tested,
        "covered": covered,
        "cover_kind": cover_kind,
        "first_covering_cmd": first_cover,
        "shown": shown,
        "first_showing_cmd": first_shown,
        "commands_total": len(commands),
    }


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
        row = analyse_episode(ep, commands, outputs, seqmap)
        row["instance"] = instance
        row["root"] = root.name
        # Episodes sharing an onset are one incident: a single edit that
        # breaks a code path many tests run through (one matplotlib line
        # produced seven). Both units are reported; only one is a fact
        # about agent behaviour.
        row["incident"] = f"{instance}@{ep.get('onset_sha') or ep.get('onset_seq')}"
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", default="", help="Also write the rows here.")
    args = ap.parse_args()

    rows: list[dict] = []
    for spec in args.roots:
        root = Path(spec)
        for f in sorted((root / "ledger" / "evidence").glob("*.json")):
            evidence = json.loads(f.read_text())
            if evidence.get("episodes"):
                rows += analyse_cell(root, evidence["instance_id"], evidence)
    if not rows:
        print("no episodes found under", " ".join(args.roots))
        return 0

    print(f"{'instance':30s} {'probe':46s} {'onset':>5s} {'brk':>4s} {'runs':>4s} "
          f"{'tested':>6s} {'covered':>7s} {'shown':>7s}  how")
    for r in rows:
        print(f"{r['instance']:30s} {r['probe'][:46]:46s} {r['onset_cmd']:>5d} {r['commands_broken']:>4d} "
              f"{r['test_runs_in_window']:>4d} {r['tested']!s:>6s} {r['covered']:>7s} "
              f"{r['shown']:>7s}  {r['cover_kind']}{'' if r['locatable'] else ' [probe id carries no path]'}")

    incidents: dict[str, list[dict]] = {}
    for r in rows:
        incidents.setdefault(r["incident"], []).append(r)

    def best(values: list[str]) -> str:
        return "yes" if "yes" in values else ("unknown" if "unknown" in values else "no")

    inc = [{"tested": any(x["tested"] for x in g),
            "covered": best([x["covered"] for x in g]), "shown": best([x["shown"] for x in g]),
            "persisted_to_end": any(x["persisted_to_end"] for x in g)}
           for g in incidents.values()]

    def tally(field: str, source: list[dict]) -> str:
        if field == "tested":
            return f"{sum(1 for x in source if x[field])}"
        yes = sum(1 for x in source if x[field] == "yes")
        unk = sum(1 for x in source if x[field] == "unknown")
        return f"{yes}" + (f" (+{unk} unknown)" if unk else "")

    print(f"\nepisodes {len(rows):3d}: tested {tally('tested', rows)}, covered {tally('covered', rows)}, "
          f"shown {tally('shown', rows)}, persisted {sum(1 for r in rows if r['persisted_to_end'])}, "
          f"unlocatable probe ids {sum(1 for r in rows if not r['locatable'])}")
    print(f"incidents {len(inc):3d}: tested {tally('tested', inc)}, covered {tally('covered', inc)}, "
          f"shown {tally('shown', inc)}, persisted {sum(1 for x in inc if x['persisted_to_end'])}")
    print("  (an incident is one breaking edit; episodes sharing an onset are the tests it broke)")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print("rows:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
