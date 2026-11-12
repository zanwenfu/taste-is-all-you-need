"""The checks a rebuilt final state passes before it stands for its episode, and the losses by model and arm.

A rebuild (``taste.brains.branch_replay``) is faithful when every recorded command's output came back as recorded;
that checks outputs, not the files. Three more checks catch a state that is faithful and still not the one recorded
(replay feasibility check, Nov 11, section 3):

- ``grade``: the hidden tests grade the rebuilt state as they graded the recorded one;
- ``changed_files``: the files the rebuilt state changed are those the record lists, where it lists them (a trial's
  own final save, or a save the study rebuilt for a check), compared inside the task's folder, git's own folder left
  out: outside it are caches and temporary folders whose names differ from run to run, and inside ``.git`` object
  names carry commit times;
- ``strip_mode_only``: git's mode lines are not changes (some images ship files whose mode differs from git's
  index, so every diff of them shows mode lines the agent never caused). A file whose only change is its mode is
  dropped from the diff and the paths; the mode lines of a file that also changed are dropped, its text kept.

``losses`` tables what was kept and what was lost, and why, by model and arm, and ``arm_gaps`` puts each model's
arms' lost shares side by side: a loss that falls unevenly between keep going and start over shows there.
"""

from __future__ import annotations

from collections import defaultdict

# A checkpoint's record lists at most this many changed paths (taste.benchmarks.branch_trial.CHANGED_PATHS).
LISTED_PATHS = 1000
DELETED = " (deleted)"
_HEADER = "diff --git a/"
_MODE = ("old mode ", "new mode ")
# How an item ended, as the loss table counts it.
COLUMNS = ("kept", "kept_first_try", "never_faithful", "grade_differs", "files_differ", "short_record", "no_record",
           "other", "running")
NO_RECORD = frozenset({"no_record", "export_failed", "not_rebuildable", "no_steps"})


def _sections(diff):
    """The diff cut at each file's header: (text before the first, [each file's lines])."""
    lines = diff.splitlines(keepends=True)
    head, files = [], []
    for line in lines:
        if line.startswith(_HEADER):
            files.append([line])
        elif files:
            files[-1].append(line)
        else:
            head.append(line)
    return "".join(head), files


def _path(header):
    """The path of ``diff --git a/<path> b/<path>`` (no renames: both sides are the same path)."""
    rest = header.rstrip("\n")[len(_HEADER):]
    half = (len(rest) - 3) // 2
    if half > 0 and rest[half:half + 3] == " b/" and rest[half + 3:] == rest[:half]:
        return rest[:half]
    return rest.split(" b/", 1)[0]


def strip_mode_only(diff):
    """(the diff without mode lines and without the files whose only change was their mode, those files' paths)."""
    if not diff:
        return diff or "", []
    head, files = _sections(diff)
    kept, only = [head], []
    for lines in files:
        body = [line for line in lines[1:] if not line.startswith(_MODE)]
        if len(body) < len(lines) - 1 and not body:
            only.append(_path(lines[0]))
            continue
        kept.append(lines[0] + "".join(body))
    return "".join(kept), only


def change_view(change):
    """A trial's final change (``taste.benchmarks.final_change``) with its mode-only files left out: the diff, the
    paths and the length without them, the length as recorded, and the rest of the record as it was."""
    if not isinstance(change, dict):
        return {"read": False, "reason": "the trial read no final change"}
    if not change.get("read"):
        return {"read": False, "reason": change.get("reason") or "the read failed"}
    diff, only = strip_mode_only(change.get("diff") or "")
    gone = set(only)
    paths = [line for line in change.get("paths") or ()
             if not (line.startswith("M\t") and line[2:] in gone)]
    return {"read": True, "diff": diff, "diff_chars": len(diff), "recorded_diff_chars": change.get("diff_chars"),
            "cut": bool(change.get("cut")), "complete": change.get("complete"), "paths": paths,
            "paths_more": change.get("paths_more"), "mode_only_paths": only, "new_files": change.get("new_files"),
            "base": change.get("base"), "head": change.get("head"), "head_moved": change.get("head_moved"),
            "cap_chars": change.get("cap_chars")}


def grade(recorded, rebuilt):
    """The hidden tests' grade of the recorded state and of the rebuilt one; equal is None when none was recorded."""
    return {"recorded": recorded, "rebuilt": rebuilt,
            "equal": None if recorded is None else (rebuilt is not None and rebuilt == recorded)}


def _inside(paths, workdir):
    """The listed paths inside the task's folder and outside its ``.git``, relative to it; and those outside."""
    root = workdir.rstrip("/") + "/"
    inside, outside = set(), set()
    for item in paths:
        item = str(item)
        deleted = item.endswith(DELETED)
        path = item[:-len(DELETED)] if deleted else item
        if path.startswith(root) and not (path[len(root):] == ".git" or path[len(root):].startswith(".git/")):
            inside.add(path[len(root):] + (DELETED if deleted else ""))
        else:
            outside.add(item)
    return inside, outside


def changed_files(recorded, rebuilt, workdir):
    """Whether the rebuilt state changed the files the record lists: judged inside the task's folder (``workdir``),
    ``.git`` left out, when both lists exist and neither was cut at the record's limit."""
    if recorded is None:
        return {"judged": False, "equal": None, "why": "the record lists no changed files"}
    if rebuilt is None:
        return {"judged": False, "equal": None, "why": "the rebuild saved no list of changed files"}
    if len(recorded) >= LISTED_PATHS or len(rebuilt) >= LISTED_PATHS:
        return {"judged": False, "equal": None, "why": f"a list was cut at {LISTED_PATHS:,} paths"}
    if not workdir or not str(workdir).startswith("/"):
        return {"judged": False, "equal": None, "why": "the task's folder is not known"}
    was, now = _inside(recorded, workdir), _inside(rebuilt, workdir)
    return {"judged": True, "equal": was[0] == now[0], "only_recorded": sorted(was[0] - now[0])[:40],
            "only_rebuilt": sorted(now[0] - was[0])[:40], "inside": {"recorded": len(was[0]), "rebuilt": len(now[0])},
            "outside_differ": len(was[1] ^ now[1])}


def _column(row):
    end = row.get("end")
    if end is None:
        return "running"
    if end in NO_RECORD:
        return "no_record"
    return end if end in COLUMNS else "other"


def losses(rows):
    """{(model, arm): counts}: the items, those kept (and on the first try), those lost by why, those still running;
    ``lost`` and ``lost_share`` count the ended items that were not kept."""
    table = defaultdict(lambda: dict.fromkeys(COLUMNS, 0) | {"items": 0})
    for row in rows:
        counts = table[(row.get("model") or "", row.get("recovery") or "")]
        counts["items"] += 1
        column = _column(row)
        counts[column] += 1
        if column == "kept" and row.get("tries") == 1:
            counts["kept_first_try"] += 1
    found = {}
    for key, counts in sorted(table.items()):
        ended = counts["items"] - counts["running"]
        lost = ended - counts["kept"]
        found[key] = {"items": counts["items"], **{name: counts[name] for name in COLUMNS},
                      "lost": lost, "lost_share": round(lost / ended, 6) if ended else None}
    return found


def arm_gaps(table):
    """Per model, the arms' lost shares (base states left out): the highest, the lowest and the gap between them."""
    by_model = defaultdict(dict)
    for (model, arm), counts in table.items():
        if arm != "base" and counts["lost_share"] is not None:
            by_model[model][arm] = counts["lost_share"]
    found = {}
    for model, shares in sorted(by_model.items()):
        highest = max(shares.items(), key=lambda item: (item[1], item[0]))
        lowest = min(shares.items(), key=lambda item: (item[1], item[0]))
        found[model] = {"shares": shares, "highest": highest, "lowest": lowest, "gap": highest[1] - lowest[1]}
    return found


def loss_text(rows):
    """The loss table as Markdown, and one line per model on the gap between its arms."""
    table = losses(rows)
    lines = ["| Model | Arm | Items | Kept | Kept first try | Never faithful | Grade differs | Files differ | "
             "Short record | No record | Other | Running | Lost (share of ended) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (model, arm), counts in table.items():
        share = "" if counts["lost_share"] is None else f" ({100 * counts['lost_share']:.1f}%)"
        lines.append(f"| {model} | {arm} | {counts['items']} | "
                     + " | ".join(str(counts[name]) for name in COLUMNS) + f" | {counts['lost']}{share} |")
    for model, gap in arm_gaps(table).items():
        (high, high_share), (low, low_share) = gap["highest"], gap["lowest"]
        lines.append(f"\n{model}: lost {100 * high_share:.1f}% ({high}) against {100 * low_share:.1f}% ({low}), "
                     f"a gap of {100 * gap['gap']:.1f} points")
    return "\n".join(lines) + "\n"
