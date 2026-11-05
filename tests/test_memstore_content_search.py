"""Full-text search over everything a session recorded, and fetching what it found.

``Store.search`` reads only what was published, on the newest state. An agent looking for
something it was told needs more, and each of these tests holds the new search to one part of it:
any fact ever recorded can be found by its own words, in any state, and the hit's pointer brings
back exactly the bytes that matched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from taste.memstore import BadPointer, Pointer, Store, Verdict
from taste.memstore.keyword import Bm25, tokens
from taste.memstore.store import NOTES


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


def _refs(store: Store) -> dict[str, str]:
    return dict(store.backend.for_each_ref("refs/"))


# ------------------------------------------------------------------ finding and fetching


def test_a_fact_only_an_old_state_holds_is_found_and_its_pointer_fetches_it(store: Store) -> None:
    a = store.branch("a", producer="brain-a")
    a.write("notes/deploy.md", "intro\nNever deploy on Fridays without a sign-off.\nend\n")
    kept = a.checkpoint("wrote the deploy note")
    a.write("notes/deploy.md", "rewritten, and nothing about that rule\n")
    a.checkpoint("rewrote the note")

    hit = store.search_contents("deploy Fridays sign-off")[0]

    assert (hit.pointer.state, hit.pointer.kind, hit.pointer.where) == (kept.id, "file", "notes/deploy.md")
    assert hit.pointer.line == 2
    assert hit.pointer.session == "s1" and hit.pointer.branch == "a"
    assert hit.pointer.blob == kept.blob("notes/deploy.md")
    assert hit.snippet == "Never deploy on Fridays without a sign-off."
    assert store.fetch(hit.pointer).text == "intro\nNever deploy on Fridays without a sign-off.\nend\n"


def test_a_word_is_found_in_its_other_forms(store: Store) -> None:
    a = store.branch("a")
    a.write("log.md", "We were deploying the connections carefully.\n")
    a.checkpoint("log")
    assert [h.pointer.where for h in store.search_contents("deployed connection")] == ["log.md"]


def test_the_rarer_word_ranks_first(store: Store) -> None:
    a = store.branch("a")
    for number in range(6):
        a.write(f"common{number}.md", "the build is green\n")
    a.write("rare.md", "the zanzibar build\n")
    a.checkpoint("many files")
    assert store.search_contents("zanzibar build")[0].pointer.where == "rare.md"


def test_a_search_that_grew_with_the_store_matches_a_fresh_one(store: Store, tmp_path: Path) -> None:
    a = store.branch("a")
    for number in range(8):
        a.write(f"f{number}.md", f"the wren said {number}\n")
        a.turn(role="user", content=f"the wren heard {number}")
        state = a.checkpoint(f"wren step {number}")
        store.search_contents("wren")
    store.judge(state, Verdict(status="pass", by="monitor", detail="wren looks fine"))

    grown = store.search_contents("wren", limit=None)
    fresh = Store.open(tmp_path / "repo", "s1")
    try:
        again = fresh.search_contents("wren", limit=None)
    finally:
        fresh.close()
    assert len(grown) == 8 + 8 + 8 + 1
    assert sorted((h.pointer.to_text(), h.score, h.snippet) for h in grown) == sorted(
        (h.pointer.to_text(), h.score, h.snippet) for h in again
    )


def test_a_word_nothing_holds_finds_nothing(store: Store) -> None:
    a = store.branch("a")
    a.write("log.md", "something else entirely\n")
    a.checkpoint("log")
    assert store.search_contents("zanzibar") == []
    assert store.search_contents("") == []


def test_the_limit_bounds_the_list(store: Store) -> None:
    a = store.branch("a")
    for number in range(5):
        a.write(f"f{number}.md", f"marker text {number}\n")
    a.checkpoint("files")
    assert len(store.search_contents("marker", limit=2)) == 2


def test_the_same_bytes_in_many_states_are_one_hit_at_the_state_that_first_held_them(store: Store) -> None:
    """Every request.json repeats the world; a hit list of copies would hide everything else."""
    a = store.branch("a")
    a.write("world.md", "the walrus rule\n")
    first = a.checkpoint("one")
    a.write("other.md", "unrelated\n")
    a.checkpoint("two")
    a.write("copy.md", "the walrus rule\n")
    a.checkpoint("three")

    (hit,) = store.search_contents("walrus")
    assert (hit.pointer.state, hit.pointer.where) == (first.id, "world.md")


# ------------------------------------------------------------------ what else is searched


def test_a_commit_message_is_found_and_its_trailers_are_not(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "body\n")
    state = a.checkpoint("chose the pelican plan")

    (hit,) = store.search_contents("pelican", kinds=("message",))
    assert (hit.pointer.state, hit.pointer.kind) == (state.id, "message")
    assert hit.snippet == "chose the pelican plan"
    assert "Taste-Session: s1" in store.fetch(hit.pointer).text
    # Every message carries these words, so matching them would rank every commit.
    assert store.search_contents("Taste-Session", kinds=("message",)) == []
    # Nor did anyone write the store's own structural commits.
    assert store.search_contents("branch from session root", kinds=("message",)) == []


def test_one_turn_of_a_transcript_is_one_hit_and_fetches_that_turn(store: Store) -> None:
    a = store.branch("a")
    a.turn(role="user", content="first thing")
    said = 'Keep the "albatross" rule\nacross lines, café included'
    a.turn(role="user", content=said)
    state = a.checkpoint("two turns")

    (hit,) = store.search_contents("albatross", kinds=("transcript",))
    assert (hit.pointer.state, hit.pointer.where) == (state.id, NOTES["transcript"])
    assert hit.pointer.line == 2
    fetched = store.fetch(hit.pointer).text
    assert said in fetched and hit.snippet in fetched
    assert "first thing" not in fetched


def test_the_names_of_a_turns_fields_are_not_searched(store: Store) -> None:
    a = store.branch("a")
    a.turn(role="user", content="nothing to see")
    a.checkpoint("one turn")
    assert store.search_contents("content", kinds=("transcript",)) == []
    assert store.search_contents("role", kinds=("transcript",)) == []


def test_a_transcript_snapshot_does_not_repeat_its_turns(store: Store, monkeypatch) -> None:
    """Every so often a state stores the whole transcript, not the turns it added."""
    monkeypatch.setattr("taste.memstore.store.TRANSCRIPT_SNAPSHOT_EVERY", 2)
    a = store.branch("a")
    a.turn(role="user", content="the heron rule")
    first = a.checkpoint("turn 0")
    for number in range(1, 6):
        a.turn(role="assistant", content=f"reply {number}")
        a.checkpoint(f"turn {number}")
    (hit,) = store.search_contents("heron", kinds=("transcript",))
    assert hit.pointer.state == first.id


def test_a_verdict_recorded_later_is_found_and_fetched(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "body\n")
    state = a.checkpoint("work")
    assert store.search_contents("flamingo") == []

    store.judge(state, Verdict(status="fail", by="monitor", detail="the flamingo check failed"))

    (hit,) = store.search_contents("flamingo")
    assert (hit.pointer.state, hit.pointer.kind, hit.pointer.where) == (state.id, "verdict", NOTES["verdicts"])
    assert "the flamingo check failed" in store.fetch(hit.pointer).text


def test_states_on_every_branch_are_searched_and_one_branch_can_be_chosen(store: Store) -> None:
    a, b = store.branch("a"), store.branch("b")
    a.write("a.md", "ostrich on a\n")
    a.checkpoint("a")
    b.write("b.md", "ostrich on b\n")
    b.checkpoint("b")
    assert {h.pointer.branch for h in store.search_contents("ostrich")} == {"a", "b"}
    assert [h.pointer.branch for h in store.search_contents("ostrich", branches=["b"])] == ["b"]


def test_a_new_state_is_found_by_the_next_search(store: Store) -> None:
    a = store.branch("a")
    a.write("one.md", "first words\n")
    a.checkpoint("one")
    assert store.search_contents("sparrow") == []
    a.write("two.md", "the sparrow arrived\n")
    a.checkpoint("two")
    assert [h.pointer.where for h in store.search_contents("sparrow")] == ["two.md"]


def test_ordinary_git_history_under_the_session_is_not_a_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for command in (["init", "-q"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", *command], cwd=repo, check=True)
    (repo / "old.md").write_text("gannet in the imported tree\n")
    subprocess.run(["git", "add", "old.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "imported kingfisher commit"], cwd=repo, check=True)
    store = Store.open(repo, "s1")
    try:
        a = store.branch("a")
        a.write("new.md", "fresh\n")
        a.checkpoint("work")
        # The tree the session imported is the session's; the old project's commits are not.
        assert store.search_contents("kingfisher") == []
        (hit,) = store.search_contents("gannet")
        assert store.state(hit.pointer.state).meta.session == "s1"
    finally:
        store.close()


# ------------------------------------------------------------------ which version is current


def test_a_revised_file_puts_the_current_version_first_on_a_tie_and_marks_the_old_one(store: Store) -> None:
    a = store.branch("a")
    a.write("rules.md", "deploy to port 8080 on every release\n")
    old = a.checkpoint("the first rule")
    a.write("rules.md", "deploy to port 9090 on every release\n")
    new = a.checkpoint("the rule changed")

    first, second = store.search_contents("which port do we deploy to on release", kinds=("file",))

    assert first.score == second.score
    assert (first.pointer.state, first.current, first.replaced_by) == (new.id, True, "")
    assert (second.pointer.state, second.current) == (old.id, False)
    assert store.fetch(second.replaced_by).text == "deploy to port 9090 on every release\n"
    assert first.order > second.order


def test_a_file_changed_back_is_current_again(store: Store) -> None:
    """After A, B, A the bytes of A are in the newest state, so they are current, though they
    are found at the state that first held them, and the B between is the one replaced."""
    a = store.branch("a")
    a.write("rules.md", "deploy to port 8080 on every release\n")
    first_a = a.checkpoint("A")
    a.write("rules.md", "deploy to port 9090 on every release\n")
    b = a.checkpoint("B")
    a.write("rules.md", "deploy to port 8080 on every release\n")
    a.checkpoint("A again")

    hits = store.search_contents("deploy port release", kinds=("file",))

    assert [(h.pointer.state, h.current) for h in hits] == [(first_a.id, True), (b.id, False)]
    assert store.fetch(hits[1].replaced_by).text == "deploy to port 8080 on every release\n"


def test_a_withdrawn_rule_is_still_found_and_says_it_was_replaced(store: Store) -> None:
    a = store.branch("a")
    a.write("rules.md", "never touch the staging database\nother rule\n")
    a.write("gone.md", "always ask about the staging database\n")
    a.checkpoint("two rules")
    a.write("rules.md", "other rule\n")
    (a.worktree / "gone.md").unlink()
    a.checkpoint("withdrew them")

    hits = {h.pointer.where: h for h in store.search_contents("staging database", kinds=("file",))}

    assert hits["rules.md"].current is False
    assert store.fetch(hits["rules.md"].replaced_by).text == "other rule\n"
    assert hits["gone.md"].current is False
    assert hits["gone.md"].replaced_by == ""  # no file holds that path any more


def test_a_better_match_in_an_old_version_keeps_its_place_and_its_mark(store: Store) -> None:
    """The current version wins a tie, not the ranking: a question about history must still find history."""
    a = store.branch("a")
    a.write("rules.md", "port port port port port and the staging port\n")
    a.checkpoint("old")
    a.write("rules.md", "one port now, among many other words that dilute it a good deal\n")
    new = a.checkpoint("new")

    first, second = store.search_contents("port", kinds=("file",))
    assert (first.current, second.current) == (False, True)
    assert second.pointer.state == new.id


def test_the_mark_moves_when_a_later_state_replaces_the_bytes(store: Store) -> None:
    a = store.branch("a")
    a.write("rules.md", "the sandpiper rule\n")
    a.checkpoint("one")
    assert store.search_contents("sandpiper", kinds=("file",))[0].current is True
    a.write("rules.md", "something else\n")
    a.checkpoint("two")
    assert store.search_contents("sandpiper", kinds=("file",))[0].current is False


def test_bytes_another_branch_still_holds_are_current(store: Store) -> None:
    a, b = store.branch("a"), store.branch("b")
    a.write("x.md", "the oystercatcher\n")
    a.checkpoint("a holds it")
    b.write("x.md", "the oystercatcher\n")
    b.checkpoint("b holds it too")
    a.write("x.md", "changed\n")
    a.checkpoint("a moved on")

    (hit,) = store.search_contents("oystercatcher", kinds=("file",))
    assert hit.current is True
    (only_a,) = store.search_contents("oystercatcher", kinds=("file",), branches=["a"])
    assert only_a.current is False


def test_every_hit_says_when_it_was_recorded_and_in_what_order(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "the kestrel\n")
    a.turn(role="user", content="the kestrel again")
    first = a.checkpoint("the kestrel in a message")
    a.turn(role="user", content="the kestrel once more")
    second = a.checkpoint("next")

    hits = {(h.pointer.kind, h.pointer.state): h for h in store.search_contents("kestrel", limit=None)}

    assert hits["file", first.id].created_at == first.meta.created_at
    assert hits["transcript", first.id].order == hits["file", first.id].order
    assert hits["transcript", second.id].order == hits["transcript", first.id].order + 1
    # A message or a turn is a record of what happened. Nothing replaces it, so the mark does not apply.
    for kind, state in (("message", first.id), ("transcript", first.id), ("transcript", second.id)):
        assert hits[kind, state].current is None and hits[kind, state].replaced_by == ""


# ------------------------------------------------------------------ what must never happen


def test_search_never_returns_another_sessions_states(tmp_path: Path) -> None:
    one, two = Store.open(tmp_path / "repo", "s1"), Store.open(tmp_path / "repo", "s2")
    try:
        a = one.branch("a")
        a.write("theirs.md", "the cormorant fact\n")
        a.checkpoint("s1 work")
        b = two.branch("b")
        b.write("mine.md", "something else\n")
        b.checkpoint("s2 work")

        assert two.search_contents("cormorant") == []
        (hit,) = one.search_contents("cormorant")
        with pytest.raises(BadPointer):
            two.fetch(hit.pointer)
    finally:
        one.close()
        two.close()


def test_searching_writes_nothing(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "the puffin\n")
    a.checkpoint("work")
    before = _refs(store)
    store.search_contents("puffin")
    store.fetch(store.search_contents("puffin")[0].pointer)
    assert _refs(store) == before


# ------------------------------------------------------------------ pointers and bytes


def test_a_pointer_survives_being_written_out_as_text(store: Store) -> None:
    a = store.branch("a")
    a.write("dir/with:colon.md", "the lapwing\n")
    a.checkpoint("work")
    hit = store.search_contents("lapwing")[0]

    text = hit.pointer.to_text()
    assert isinstance(text, str)
    assert store.fetch(text).text == "the lapwing\n"
    assert store.fetch(Pointer.from_text(text)).text == "the lapwing\n"


def test_bad_pointers_are_refused(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "the curlew\n")
    state = a.checkpoint("work")
    good = store.search_contents("curlew")[0].pointer

    for bad in (
        "not a pointer",
        f"{state.id[:12]}:file:1:x.md",  # an exact id only
        f"{'0' * 40}:file:1:x.md",  # no such state
        f"{state.id}:file:1:missing.md",
        f"{state.id}:nonsense:1:x.md",
        f"{state.id}:transcript:99:{NOTES['transcript']}",
        f"{state.id}:verdict:1:{NOTES['verdicts']}",
    ):
        with pytest.raises(BadPointer):
            store.fetch(bad)
    with pytest.raises(BadPointer):
        store.fetch(Pointer(**{**good.__dict__, "blob": "f" * 40}))


def test_fetch_pages_through_a_document(store: Store) -> None:
    a = store.branch("a")
    body = "the dunlin\n" + "café " * 2000
    a.write("long.md", body)
    a.checkpoint("work")
    pointer = store.search_contents("dunlin")[0].pointer

    pages, offset = [], 0
    while True:
        page = store.fetch(pointer, offset=offset, limit=1000)
        pages.append(page.text)
        assert page.size == len(body)
        offset = page.next_offset
        if page.eof:
            break
    assert "".join(pages) == body
    assert store.fetch(pointer, offset=len(body) + 5).text == ""
    with pytest.raises(ValueError):
        store.fetch(pointer, offset=-1)
    with pytest.raises(ValueError):
        store.fetch(pointer, limit=0)


def test_a_binary_file_is_not_searched_and_other_bytes_round_trip(store: Store) -> None:
    a = store.branch("a")
    a.write("image.bin", b"\x00\x01 the godwit \x02")
    latin = "the plover café\n".encode("latin-1")
    a.write("latin.txt", latin)
    a.checkpoint("work")

    assert store.search_contents("godwit") == []
    (hit,) = store.search_contents("plover")
    assert store.fetch(hit.pointer).text.encode("utf-8", "surrogateescape") == latin


def test_a_large_file_is_left_out_and_counted(store: Store, monkeypatch) -> None:
    monkeypatch.setattr("taste.memstore.search.MAX_DOCUMENT_BYTES", 100)
    a = store.branch("a")
    a.write("big.md", "the avocet " * 50)
    a.checkpoint("work")
    assert store.search_contents("avocet") == []
    assert store.memory_counts["documents_too_large"] == 1


# ------------------------------------------------------------------ properties


def test_each_planted_fact_is_found_by_its_own_words_and_fetches_its_own_bytes(store: Store) -> None:
    a = store.branch("a")
    facts = [f"plantedword{number:02d}x remembers item {number}" for number in range(30)]
    for commit in range(10):
        for fact_number in range(commit * 3, commit * 3 + 3):
            a.write(f"facts/{fact_number}.md", f"before\n{facts[fact_number]}\nafter\n")
        a.checkpoint(f"commit {commit}")

    for number, fact in enumerate(facts):
        hit = store.search_contents(f"plantedword{number:02d}x")[0]
        assert hit.pointer.where == f"facts/{number}.md"
        assert hit.snippet == fact
        assert hit.snippet in store.fetch(hit.pointer).text


def test_the_same_text_ranked_by_the_same_code_outside_git_gives_the_same_order(store: Store) -> None:
    """With files only, git adds nothing to the ranking. A difference here would be a bug, not a finding."""
    a = store.branch("a")
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    texts = [" ".join(words[(number * step) % 10] for step in range(1, 2 + number % 7)) + f" doc{number}"
             for number in range(40)]
    for number, text in enumerate(texts):
        a.write(f"d{number:02d}.md", text + "\n")
        a.checkpoint(f"commit {number}")

    flat = Bm25()
    for text in texts:
        flat.add(tokens(text + "\n"))
    for query in ("alpha gamma", "kappa", "doc7 theta", "beta beta iota"):
        in_git = [h.pointer.where for h in store.search_contents(query, limit=40, kinds=("file",))]
        outside = [f"d{doc:02d}.md" for doc, _ in flat.rank(tokens(query), limit=40)]
        assert in_git == outside, query


# ------------------------------------------------------------------ the claim ledger's counts


def test_search_and_fetch_calls_are_counted(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "the turnstone\n")
    a.checkpoint("work")
    assert store.memory_counts["search"] == 0 and store.memory_counts["fetch"] == 0
    hit = store.search_contents("turnstone")[0]
    store.search_contents("nothing matches this")
    store.fetch(hit.pointer)
    assert store.memory_counts["search"] == 2 and store.memory_counts["fetch"] == 1
