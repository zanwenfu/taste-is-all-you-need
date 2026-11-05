"""A small YAML header per commit: what a state holds, readable before reading the state.

It lives in the commit message, between the subject and the trailers, so it is bound to the commit id
and shows in plain ``git log``. Off unless asked for: a checkpoint without one must not change by a
byte, because commit ids depend on the message and older runs must stay reproducible.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from taste.memstore import CommitHeader, Store
from taste.memstore.objects import header_fields


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


def _git(store: Store, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=store.root, capture_output=True, text=True, check=True)
    return result.stdout


def _fields(text: str) -> dict:
    """A reader that shares no code with the store's: the flat form, by hand."""
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.partition(": ")
        assert separator, line
        fields[key] = value
    return fields


def test_without_a_header_the_message_is_what_it_always_was(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "body\n")
    state = a.checkpoint("plain work")

    message = _git(store, "log", "-1", "--format=%B", state.id).strip().split("\n")
    assert message[0] == "plain work" and message[1] == ""
    assert [line.split(":")[0] for line in message[2:]] == [
        "Taste-State", "Taste-Branch", "Taste-Session", "Taste-Created",
    ]
    assert state.header == ""


def test_the_header_sits_between_subject_and_trailers_and_git_still_reads_both(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "body\n")
    state = a.checkpoint("step 3", header=CommitHeader(what="the user set a rule"))

    assert _git(store, "log", "-1", "--format=%s", state.id).strip() == "step 3"
    body = _git(store, "log", "-1", "--format=%b", state.id)
    assert 'what: "the user set a rule"' in body
    # A line of three hyphens would hide the trailers from git, so the header must not use one.
    assert "\n---" not in body
    trailers = _git(store, "log", "-1", "--format=%(trailers:only)", state.id)
    assert "Taste-State: checkpoint" in trailers and "Taste-Session: s1" in trailers


def test_the_header_holds_what_the_caller_said_and_what_the_commit_can_work_out(store: Store) -> None:
    a = store.branch("a")
    a.write("old.md", "before\n")
    earlier = a.checkpoint("first")
    a.write("old.md", "after\n")
    a.write("dir/new.md", "new\n")
    state = a.checkpoint(
        "second",
        header=CommitHeader(what="two files", keywords=("alpha", "beta"), pointers=(earlier.id,)),
    )

    assert header_fields(state.header) == {
        "header": "taste.memstore/1",
        "what": "two files",
        "kind": "checkpoint",
        "branch": "a",
        "paths": ["dir/new.md", "old.md"],
        "paths_total": "2",
        "keywords": ["alpha", "beta"],
        "pointers": [earlier.id],
    }


def test_the_key_set_is_fixed_and_one_per_line_with_quoted_values(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "body\n")
    fields = _fields(a.checkpoint("work", header=CommitHeader()).header)
    assert list(fields) == ["header", "what", "kind", "branch", "paths", "paths_total", "keywords", "pointers"]
    for key in ("header", "what", "kind", "branch", "paths_total"):
        assert fields[key].startswith('"') and fields[key].endswith('"')
    for key in ("paths", "keywords", "pointers"):
        assert fields[key].startswith("[") and fields[key].endswith("]")


def test_the_header_is_real_yaml_whatever_it_is_given() -> None:
    yaml = pytest.importorskip("yaml")
    hard = 'say "no": caf\u00e9 \\ back\nslash #1 - [x] {y} \x7f\x85' + chr(0x2028) + "\ud800 end"
    text = CommitHeader(what=hard, keywords=(hard,), pointers=("abc",)).render(
        kind="checkpoint", branch="a", paths=["a b/c:d.md"]
    )
    assert "\n\n" not in text
    assert yaml.safe_load(text) == header_fields(text)
    assert header_fields(text)["what"].startswith('say "no": café \\ back slash #1')


def test_a_long_header_is_cut_to_stay_small() -> None:
    text = CommitHeader(
        what="w" * 10_000,
        keywords=tuple(f"k{n}" for n in range(1_000)),
        pointers=tuple(f"p{n}" for n in range(1_000)),
    ).render(kind="checkpoint", branch="a", paths=[f"p/{n}.md" for n in range(5_000)])
    fields = header_fields(text)
    assert len(text.encode()) < 8_192
    assert len(fields["what"]) == 400
    assert len(fields["keywords"]) == 32 and len(fields["pointers"]) == 16
    assert len(fields["paths"]) == 32 and fields["paths_total"] == "5000"


def test_headers_never_change_older_commit_ids(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "one\n")
    first = a.checkpoint("one")
    a.write("x.md", "two\n")
    second = a.checkpoint("two")
    a.write("x.md", "three\n")
    third = a.checkpoint("three", header=CommitHeader(what="later"))

    assert [state.id for state in a.history(limit=3)] == [third.id, second.id, first.id]
    assert second.header == "" and first.header == ""
    assert second.read("x.md") == "two\n"
    assert _git(store, "merge-base", "--is-ancestor", first.id, third.id) == ""


def test_a_header_can_be_built_and_published_apart(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "body\n")
    built = a.build("work", header=CommitHeader(what="built apart"))
    state = a.publish_state(built)
    assert header_fields(state.header)["what"] == "built apart"


def test_search_finds_the_words_a_header_gives_and_every_hit_carries_its_commits_header(store: Store) -> None:
    a = store.branch("a")
    a.write("notes/x.md", "the body says nothing useful\n")
    state = a.checkpoint("step 1", header=CommitHeader(what="a rule about herons", keywords=("egret",)))

    (hit,) = store.search_contents("herons", kinds=("message",))
    assert (hit.pointer.state, hit.pointer.kind) == (state.id, "message")
    assert hit.header == state.header
    assert "herons" in hit.snippet and hit.snippet in store.fetch(hit.pointer).text
    (hit,) = store.search_contents("egret", kinds=("message",))
    assert hit.pointer.state == state.id
    (hit,) = store.search_contents("notes", kinds=("message",))  # a changed path is a word too
    assert hit.pointer.state == state.id

    (hit,) = store.search_contents("useful", kinds=("file",))
    assert hit.header == state.header


def test_the_headers_own_labels_are_not_searched(store: Store) -> None:
    """Every header carries these words, so matching them would rank every commit."""
    a = store.branch("a")
    a.write("x.md", "body\n")
    a.checkpoint("step 1", header=CommitHeader(what="nothing here"))
    for word in ("what", "kind", "branch", "paths", "keywords", "pointers", "header", "checkpoint", "memstore"):
        assert store.search_contents(word, kinds=("message",)) == [], word


def test_a_hit_on_a_state_with_no_header_says_so(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "the shrike\n")
    a.checkpoint("work")
    (hit,) = store.search_contents("shrike")
    assert hit.header == ""
