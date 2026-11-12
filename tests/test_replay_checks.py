"""The checks a rebuilt final state passes before it stands for the episode (taste.recovery_study.replay_checks).

A rebuild is "faithful" when every recorded command's output came back; that checks outputs, not file bytes. Three
more checks: the hidden tests grade the rebuilt state as they graded the recorded one; the files it changed are the
files the record lists, where it lists them; git's mode-only lines are not changes. And the losses are tabled by
model and arm, so that an uneven loss shows.
"""

from __future__ import annotations

from taste.recovery_study import replay_checks as checks

MODE_ONLY = ("diff --git a/bin/run b/bin/run\nold mode 100644\nnew mode 100755\n")
MODE_AND_TEXT = ("diff --git a/src/app.py b/src/app.py\nold mode 100644\nnew mode 100755\nindex 1111111..2222222\n"
                 "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n")
TEXT = ("diff --git a/src/my file.py b/src/my file.py\nindex 3333333..4444444 100644\n--- a/src/my file.py\n"
        "+++ b/src/my file.py\n@@ -1 +1 @@\n-a\n+b\n")
NEW = ("diff --git a/notes.txt b/notes.txt\nnew file mode 100644\nindex 0000000..5555555\n--- /dev/null\n"
       "+++ b/notes.txt\n@@ -0,0 +1 @@\n+note\n")


def test_a_file_whose_only_change_is_its_mode_is_no_change_and_mode_lines_go():
    diff, only = checks.strip_mode_only(MODE_ONLY + MODE_AND_TEXT + TEXT + NEW)
    assert only == ["bin/run"]
    assert "bin/run" not in diff and "old mode" not in diff and "new mode 100755" not in diff
    # A file with a mode change and a text change keeps its text change; a new file keeps its "new file mode" line.
    assert "+new" in diff and "diff --git a/src/app.py b/src/app.py\nindex 1111111..2222222" in diff
    assert "new file mode 100644" in diff and "diff --git a/src/my file.py b/src/my file.py" in diff
    assert checks.strip_mode_only("") == ("", [])


def test_the_change_view_drops_mode_only_paths_and_keeps_the_record():
    change = {"read": True, "diff": MODE_ONLY + TEXT, "diff_chars": len(MODE_ONLY + TEXT), "cut": False,
              "complete": True, "paths": ["M\tbin/run", "M\tsrc/my file.py", "?\tnotes.txt"], "paths_more": 0,
              "new_files": 1, "base": "a" * 40, "head": "a" * 40, "head_moved": False, "cap_chars": 60000}
    view = checks.change_view(change)
    assert view["paths"] == ["M\tsrc/my file.py", "?\tnotes.txt"] and view["mode_only_paths"] == ["bin/run"]
    assert view["diff"] == TEXT and view["diff_chars"] == len(TEXT) and view["recorded_diff_chars"] == len(MODE_ONLY + TEXT)
    assert view["read"] and view["complete"] and not view["cut"] and view["head_moved"] is False
    assert checks.change_view(None) == {"read": False, "reason": "the trial read no final change"}
    assert checks.change_view({"read": False, "reason": "not a git repository"})["reason"] == "not a git repository"


def test_the_grade_check_holds_only_when_both_grades_are_known_and_equal():
    assert checks.grade(1.0, 1.0) == {"recorded": 1.0, "rebuilt": 1.0, "equal": True}
    assert checks.grade(1.0, 0.0)["equal"] is False
    assert checks.grade(None, 1.0)["equal"] is None and checks.grade(1.0, None)["equal"] is False


def test_changed_files_are_judged_inside_the_task_folder_only_and_never_inside_git():
    recorded = ["/app/src/app.py", "/app/old.py (deleted)", "/app/.git/index", "/root/.cache/go-build/ab/cd",
                "/tmp/pytest-of-root/pytest-3"]
    rebuilt = ["/app/src/app.py", "/app/old.py (deleted)", "/app/.git/index", "/app/.git/objects/12/34",
               "/root/.cache/go-build/ef/01", "/tmp/pytest-of-root/pytest-7"]
    found = checks.changed_files(recorded, rebuilt, "/app")
    assert found["judged"] and found["equal"] is True
    # Outside the folder (caches, temporary folders, git's objects) names differ: listed as a count, not judged.
    assert found["outside_differ"] == 5 and found["inside"] == {"recorded": 2, "rebuilt": 2}
    other = checks.changed_files(recorded, ["/app/src/app.py", "/app/new.py"], "/app/")
    assert other["equal"] is False
    assert other["only_recorded"] == ["old.py (deleted)"] and other["only_rebuilt"] == ["new.py"]


def test_changed_files_are_not_judged_without_both_lists_or_when_a_list_was_cut():
    assert checks.changed_files(None, ["/app/a"], "/app") == {"judged": False, "equal": None,
                                                            "why": "the record lists no changed files"}
    assert checks.changed_files(["/app/a"], None, "/app")["why"] == "the rebuild saved no list of changed files"
    full = [f"/app/f{number}" for number in range(checks.LISTED_PATHS)]
    assert checks.changed_files(full, full, "/app")["why"] == "a list was cut at 1,000 paths"
    assert checks.changed_files(["/app/a"], ["/app/a"], "")["why"] == "the task's folder is not known"


def rows():
    """Per-item rows as the replay driver's summary gives them."""
    def row(model, recovery, end, tries=1):
        return {"model": model, "recovery": recovery, "end": end, "tries": tries}
    claude, luna = "anthropic/claude-x", "azure/gpt-6-luna"
    return ([row(claude, "continue", "kept")] * 18 + [row(claude, "continue", "kept", 2)] * 1
            + [row(claude, "continue", "never_faithful", 3)]
            + [row(claude, "retry_feedback", "kept")] * 15 + [row(claude, "retry_feedback", "grade_differs", 3)] * 2
            + [row(claude, "retry_feedback", "files_differ", 3), row(claude, "retry_feedback", "short_record"),
               row(claude, "retry_feedback", "no_record", 0), row(claude, "retry_feedback", None, 1)]
            + [row(luna, "continue", "kept")] * 10 + [row(luna, "base", "kept")] * 3)


def test_the_loss_table_counts_each_model_and_arm_and_its_lost_share():
    table = checks.losses(rows())
    claude_continue = table[("anthropic/claude-x", "continue")]
    assert claude_continue == {"items": 20, "kept": 19, "kept_first_try": 18, "never_faithful": 1, "grade_differs": 0,
                               "files_differ": 0, "short_record": 0, "no_record": 0, "other": 0, "running": 0,
                               "lost": 1, "lost_share": 0.05}
    retry = table[("anthropic/claude-x", "retry_feedback")]
    assert (retry["items"], retry["kept"], retry["lost"], retry["running"]) == (21, 15, 5, 1)
    assert retry["lost_share"] == 0.25  # of the 20 ended
    assert table[("azure/gpt-6-luna", "continue")]["lost"] == 0


def test_the_gap_between_arms_shows_an_uneven_loss_per_model():
    gaps = checks.arm_gaps(checks.losses(rows()))
    claude = gaps["anthropic/claude-x"]
    assert claude["highest"] == ("retry_feedback", 0.25) and claude["lowest"] == ("continue", 0.05)
    assert abs(claude["gap"] - 0.20) < 1e-9 and gaps["azure/gpt-6-luna"]["gap"] == 0.0
    text = checks.loss_text(rows())
    assert "| anthropic/claude-x | retry_feedback | 21 | 15 | 15 | 0 | 2 | 1 | 1 | 1 | 0 | 1 | 5 (25.0%) |" in text
    assert "anthropic/claude-x: lost 25.0% (retry_feedback) against 5.0% (continue), a gap of 20.0 points" in text
