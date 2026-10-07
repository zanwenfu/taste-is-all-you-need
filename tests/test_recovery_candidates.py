"""The calibration candidates follow their written rule: files-only tasks, stratified, in salted order."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "recovery_candidates", Path(__file__).resolve().parents[1] / "scripts" / "recovery_candidates.py")
candidates_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(candidates_module)


def diff(*files, lines=1):
    body = "".join(f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n"
                   + "-old\n+new\n" * lines for name in files)
    return body


def pro_task(root, name, *files, lines=1):
    task = root / "tasks" / name
    (task / "solution").mkdir(parents=True)
    (task / "task.toml").write_text('[task]\nname = "swebench-pro/x"\n')
    (task / "solution" / "gold_patch.diff").write_text(diff(*files, lines=lines))


def tb3_task(root, name, *, mode="separate", services=(), gpus=0, verifier_gpus=0):
    task = root / name
    (task / "environment").mkdir(parents=True)
    toml = f'[verifier]\nenvironment_mode = "{mode}"\n'
    if verifier_gpus:
        toml += f"[verifier.environment]\ngpus = {verifier_gpus}\n"
    toml += f"[environment]\ngpus = {gpus}\n"
    (task / "task.toml").write_text(toml)
    if services:
        (task / "environment" / "docker-compose.yaml").write_text(
            "services:\n" + "".join(f"  {name}:\n    image: x\n" for name in services) + "volumes:\n  data:\n")


def test_the_language_is_the_most_common_listed_one_and_lines_count_changes():
    assert candidates_module.patch_facts(diff("a.py", "b.py", "c.go", lines=2)) == ("python", 12)
    # A changelog or template file never decides the language.
    assert candidates_module.patch_facts(diff("changelogs/fragment.yml", "lib/x.py")) == ("python", 4)
    assert candidates_module.patch_facts(diff("src/a.tsx", "README.md", "docs/b.md")) == ("js-ts", 6)
    assert candidates_module.patch_facts(diff("Makefile")) == ("other", 2)


@pytest.fixture
def pro_root(tmp_path):
    root = tmp_path / "v2"
    for index in range(9):
        pro_task(root, f"instance_py__{index}", "x.py", lines=index + 1)
    for index in range(6):
        pro_task(root, f"instance_go__{index}", "x.go", lines=index + 1)
    pro_task(root, "instance_py__hard", "x.py", lines=3)
    (root / "hard51_ids.txt").write_text("instance_py__hard\n")
    return root


def test_hard51_is_left_out_and_sizes_are_terciles_within_a_language(pro_root):
    tasks = candidates_module.pro_tasks(pro_root)
    assert [task["task"] for task in tasks if task["hard51"]] == ["instance_py__hard"]
    sizes = candidates_module.size_terciles([task for task in tasks if not task["hard51"]])
    assert [sizes[f"instance_py__{index}"] for index in range(9)] == ["small"] * 3 + ["medium"] * 3 + ["large"] * 3
    assert [sizes[f"instance_go__{index}"] for index in range(6)] == ["small"] * 2 + ["medium"] * 2 + ["large"] * 2


def test_seats_are_proportional_and_the_order_is_fixed_by_the_salt(pro_root):
    tasks = candidates_module.pro_tasks(pro_root)
    chosen, strata = candidates_module.choose_pro(tasks, 10, "salt-a")
    assert sum(item["seats"] for item in strata.values()) == 10
    assert {key: item["tasks"] for key, item in strata.items()} == {
        "go/large": 2, "go/medium": 2, "go/small": 2, "python/large": 3, "python/medium": 3, "python/small": 3}
    assert all(len(chosen[key]) == strata[key]["seats"] for key in strata)
    assert "instance_py__hard" not in {name for names in chosen.values() for name in names}
    assert candidates_module.choose_pro(tasks, 10, "salt-a")[0] == chosen
    others = [candidates_module.choose_pro(tasks, 10, f"salt-{index}")[0] for index in range(8)]
    assert any(other != chosen for other in others)
    with pytest.raises(ValueError):
        candidates_module.choose_pro(tasks, 16, "salt-a")


def test_terminal_bench_tasks_qualify_only_on_their_own_files(tmp_path):
    tb3_task(tmp_path, "files-only")
    tb3_task(tmp_path, "shared", mode="shared")
    tb3_task(tmp_path, "sidecar", services=("main", "postgres", "api"))
    tb3_task(tmp_path, "main-only-compose", services=("main",))
    tb3_task(tmp_path, "gpu", gpus=1)
    tb3_task(tmp_path, "verifier-gpu", verifier_gpus=1)
    verdicts = {path.name: candidates_module.tb3_verdict(path) for path in sorted(tmp_path.iterdir())}
    assert verdicts == {
        "files-only": (True, ""), "main-only-compose": (True, ""),
        "shared": (False, "verifier shares the agent's container (read its tests)"),
        "sidecar": (False, "sidecar services: postgres, api"),
        "gpu": (False, "needs a GPU"), "verifier-gpu": (False, "needs a GPU")}


def test_the_record_names_its_rule_and_both_benchmarks(pro_root, tmp_path):
    tb3 = tmp_path / "tb3"
    tb3_task(tb3, "files-only")
    tb3_task(tb3, "sidecar", services=("main", "redis"))
    out = tmp_path / "candidates.json"
    candidates_module.main(["--pro", str(pro_root), "--tb3", str(tb3), "--pro-seats", "6",
                            "--salt", "salt-a", "--out", str(out)])
    record = json.loads(out.read_text())
    assert record["schema"] == "taste.recovery/CalibrationCandidates/1" and record["salt"] == "salt-a"
    assert "HARD-51" in record["rule"] and len(record["swebench_pro_v2"]["tasks"]) == 6
    tb3_part = record["terminal_bench_3"]
    assert (tb3_part["tasks"], tb3_part["excluded"]) == (["files-only"], {"sidecar": "sidecar services: redis"})
    # Each part pins the dataset it was chosen from.
    assert tb3_part["dataset_digest"] == candidates_module.dataset_digest(tb3)
    assert record["swebench_pro_v2"]["dataset_digest"] == candidates_module.dataset_digest(pro_root / "tasks")
    (tb3 / "files-only" / "task.toml").write_text('[verifier]\nenvironment_mode = "shared"\n')
    assert candidates_module.dataset_digest(tb3) != tb3_part["dataset_digest"]
