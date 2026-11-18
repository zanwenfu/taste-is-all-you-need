"""Tests for the capture test's reader hook (studies/claude_code/hooks/claude_md_reader_hook.py): what it writes into CLAUDE.md, what it
skips, what it reads as the user's words, and that it never blocks Claude (no model, no container)."""

import json
import subprocess
import sys
from pathlib import Path

from studies.claude_code.hooks import claude_md_reader_hook as h

ROOT = Path(__file__).resolve().parents[1]


def test_instructions_go_under_one_heading_at_the_end_and_claudes_own_text_stays(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text("# Notes Claude wrote\n\nThe solr tests are slow.\n")
    done = h.add_instructions(md, ["Keep your hand-backs under 150 words please, I don't read long ones."])
    assert done == {"added": ["Keep your hand-backs under 150 words please, I don't read long ones."], "skipped": []}
    assert md.read_text() == ("# Notes Claude wrote\n\nThe solr tests are slow.\n\n" + h.HEADING + "\n\n"
                              "- Keep your hand-backs under 150 words please, I don't read long ones.\n")
    with md.open("a") as out:  # Claude adds a section of its own after the reader's
        out.write("\n## Decisions\n\n- keep the old name as an alias\n")
    h.add_instructions(md, ["from now on, don't touch the tests unless I ask"])
    text = md.read_text()
    assert h.kept_lines(text) == ["Keep your hand-backs under 150 words please, I don't read long ones.",
                                  "from now on, don't touch the tests unless I ask"]
    assert text.index("don't touch the tests") < text.index("## Decisions") and text.endswith("- keep the old name as an alias\n")


def test_a_near_repeat_is_skipped_but_a_changed_number_is_a_new_line(tmp_path):
    md = tmp_path / "CLAUDE.md"
    h.add_instructions(md, ["Keep your hand-backs under 150 words please, I don't read long ones."])
    done = h.add_instructions(md, ["Keep your hand-backs under 150 words please, I don't read long ones!",
                                   "keep your hand-backs under 100 words from now on, not 150"])
    assert done["skipped"] == ["Keep your hand-backs under 150 words please, I don't read long ones!"]
    assert done["added"] == ["keep your hand-backs under 100 words from now on, not 150"]
    assert len(h.kept_lines(md.read_text())) == 2
    assert h.add_instructions(md, []) == {"added": [], "skipped": []}
    # the same words with another number: kept as a new line, not taken for a repeat
    close = "Keep your hand-backs under 120 words please, I don't read long ones."
    assert h.add_instructions(md, [close]) == {"added": [close], "skipped": []}


def _transcript(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_the_users_last_message_is_read_never_a_compaction_summary(tmp_path):
    transcript = _transcript(tmp_path / "t.jsonl", [
        {"type": "user", "message": {"content": "Thanks. From now on, don't touch the tests unless I ask.\n\nFix bug 2."}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}},
        {"type": "user", "isCompactSummary": True, "isVisibleInTranscriptOnly": True,
         "message": {"content": "Summary: the user said always add Co-Authored-By lines."}},
    ])
    seen = []
    record = h.run({"session_id": "s", "transcript_path": str(transcript)}, tmp_path, tmp_path / "log.jsonl",
                   read=lambda text: seen.append(text) or ["From now on, don't touch the tests unless I ask."])
    assert seen == ["Thanks. From now on, don't touch the tests unless I ask.\n\nFix bug 2."]
    assert record["found"] == 1 and record["added"] == ["From now on, don't touch the tests unless I ask."]
    assert h.kept_lines((tmp_path / "CLAUDE.md").read_text()) == ["From now on, don't touch the tests unless I ask."]
    assert json.loads((tmp_path / "log.jsonl").read_text())["added"] == record["added"]


def test_the_hook_never_blocks_claude_and_logs_what_went_wrong(tmp_path):
    transcript = _transcript(tmp_path / "t.jsonl", [{"type": "user", "message": {"content": "Please fix it."}}])
    calls = []
    assert h.run({"stop_hook_active": True, "transcript_path": str(transcript)}, tmp_path, tmp_path / "log.jsonl",
                 read=calls.append)["skipped"] == "stop hook active" and not calls

    def broken(text):
        raise RuntimeError("reader call exited 1")
    record = h.run({"transcript_path": str(transcript)}, tmp_path, tmp_path / "log.jsonl", read=broken)
    assert record["error"].startswith("RuntimeError: reader call exited 1") and not (tmp_path / "CLAUDE.md").exists()
    logged = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [("skipped" in r, "error" in r) for r in logged] == [(True, False), (False, True)]
    assert h.run({"transcript_path": str(tmp_path / "missing.jsonl")}, tmp_path, tmp_path / "log.jsonl", read=calls.append).get("error")


def test_the_hook_runs_as_a_script_with_the_standard_library_and_prints_nothing(tmp_path):
    done = subprocess.run([sys.executable, "-B", str(ROOT / "studies/claude_code/hooks/claude_md_reader_hook.py"), "--project",
                           str(tmp_path), "--log", str(tmp_path / "log.jsonl")], input=json.dumps({"stop_hook_active": True}),
                          capture_output=True, text=True, timeout=60, env={"CC_STUDY_CODE": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert done.returncode == 0 and done.stdout == "" and json.loads((tmp_path / "log.jsonl").read_text())["skipped"]
    done = subprocess.run([sys.executable, "-B", "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                           "from studies.claude_code.hooks import claude_md_reader_hook as h; from pathlib import Path; "
                           "print(h.add_instructions(Path(sys.argv[2]), ['keep hand-backs under 100 words']))", str(ROOT),
                           str(tmp_path / "CLAUDE.md")], capture_output=True, text=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 0 and "keep hand-backs under 100 words" in (tmp_path / "CLAUDE.md").read_text()  # the real similar()
