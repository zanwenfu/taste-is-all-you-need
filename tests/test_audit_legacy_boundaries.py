"""Statistical and persisted-event regressions with independently known answers."""
import json

from taste.attempts import harvest_retries
from taste.evalrun import CellResult
from taste.kernel import Event
from taste.stats import PairedBlock, minimum_detectable_effect, paired_permutation, summarise_arm


def test_retry_harvest_consumes_actual_kernel_wire(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [Event("step.begin", {"id": "a", "attempt": i}) for i in (1, 2, 3)]
    path.write_text("".join(json.dumps(event.to_json()) + "\n" for event in events))
    assert harvest_retries(path) == 2


def test_repetitions_on_one_task_do_not_create_independent_evidence():
    blocks = [PairedBlock(instance="task", trial=i, repo="task", a=1, b=0) for i in range(20)]
    result = paired_permutation(blocks)
    # Only two independent sign arrangements exist, both equally extreme.
    assert result.p_value == 1
    assert result.n_clusters == 1 and result.n_blocks == 20


def test_mde_responds_to_declared_alpha_and_power():
    usual = minimum_detectable_effect(40, 1)
    assert minimum_detectable_effect(40, 1, alpha=0.01) > usual
    assert minimum_detectable_effect(40, 1, power=0.95) > usual


def test_missing_budget_score_cannot_improve_reported_mean():
    records = [CellResult(task="t1", arm="A", trial=1, status="completed", config_hash="h", score=1),
               CellResult(task="t2", arm="A", trial=1, status="budget", config_hash="h", score=None)]
    summary = summarise_arm("A", records)
    assert summary.mean is None
    assert summary.n_usable == 2 and summary.n_missing_score == 1
