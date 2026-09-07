"""The communicator scenario with the brains stubbed out.

Two branches. B publishes a record. A searches for it, finds it, reads only
that. A checkpoints with its context, fails, rolls back, and the failed
state is still there with its reason and its transcript. Then the process
is "killed" between building a state and publishing it, and the store is
consistent on reopen. Nothing is lost at any point.

Run:  python examples/memstore_demo.py [root]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from taste.memstore import ObjectType, StaleBranch, Store, Transcript


def run(root: Path, *, echo: bool = False) -> dict[str, Any]:
    say = print if echo else (lambda *a, **k: None)
    facts: dict[str, Any] = {}
    store = Store.open(root, "demo")

    # --- B produces something and publishes it ------------------------------
    b = store.branch("worker-b", producer="brain-b")
    b.publish("revenue_summary", "out/revenue.json", type=ObjectType.RECORD,
              description="Q3 revenue by region, in USD")
    b_state = b.checkpoint(
        "computed Q3 revenue",
        records={"out/revenue.json": {"emea": 120, "amer": 340, "apac": 95}},
        transcript=Transcript().append(role="assistant", content="summed the regions"),
    )
    say(f"B published revenue_summary at {b_state.id[:10]}")

    # --- A asks the store: does anyone have revenue? -------------------------
    a = store.branch("worker-a", producer="brain-a")
    hits = store.search("revenue Q3")
    facts["found"] = [(h.branch, h.entry.name) for h in hits]
    say(f"A searched 'revenue Q3' -> {facts['found']}")
    hit = hits[0]
    revenue = hit.state.record(hit.entry.path)  # reads only that artifact
    facts["revenue_total"] = sum(revenue.values())
    say(f"A paged in {hit.entry.path} from {hit.branch}: total {facts['revenue_total']}")

    # --- A works, checkpoints with its context, then fails -------------------
    a.write("report.md", f"# Report\n\nTotal revenue: {facts['revenue_total']}\n")
    a.publish("report", "report.md", description="the quarterly report")
    good = a.checkpoint("drafted report",
                        transcript=Transcript().append(role="assistant", content="used B's numbers"))
    a.write("report.md", "# Report\n\nTotal revenue: 9999999\n")  # a bad edit
    bad = a.checkpoint("revised report (wrong)",
                       transcript=Transcript().append(role="assistant", content="tried a shortcut"))
    say(f"A's bad state {bad.id[:10]} exists")

    # --- rollback: an append; nothing is lost --------------------------------
    back = a.rollback(good, "monitor: total does not match B's record")
    facts["after_rollback"] = a.read("report.md")
    facts["failed_state_still_readable"] = a.store.state(bad.id).read("report.md")
    facts["failed_transcript"] = a.store.state(bad.id).transcript.turns[0]["content"]
    facts["history"] = [s.meta.reason for s in a.history()]
    say(f"rolled back to {good.id[:10]}; history = {facts['history']}")
    assert back.parents[0] == bad

    # --- a kill between building and publishing a state ----------------------
    a.write("report.md", "# Report\n\nTotal revenue: 555 (unpublished draft)\n")
    a.backend.stage_all()
    tree = a.backend.write_tree()
    head = a.head
    try:
        # simulate: the ref moved underneath us (another brain, or a restart)
        a._commit(tree=tree, parents=[head.id], kind="checkpoint", reason="draft",
                  manifest=head.manifest, transcript=head.transcript, verdict=None, attempt=0,
                  expected_head="0" * 40)
    except StaleBranch:
        facts["stale_publish_rejected"] = True
    store.close()

    # --- reopen: consistent ---------------------------------------------------
    store = Store.open(root, "demo")
    a = store.branch("worker-a")
    facts["head_after_reopen"] = a.head.meta.reason
    facts["head_is_complete"] = a.head.manifest is not None and a.head.transcript is not None
    say(f"reopened: head = {facts['head_after_reopen']!r}")
    store.close()
    return facts


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp()) / "repo"
    out = run(root, echo=True)
    print("\nfacts:", out)
