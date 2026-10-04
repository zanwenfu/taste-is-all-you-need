import hashlib
import json
import math

import pytest

from taste.benchmarks.task_split import (
    SplitRefused,
    TaskInfo,
    dataset_digest,
    load_tasks,
    make_split_record,
    select,
    split_tasks,
)


def task(directory, name, category="software-engineering", difficulty="medium", extra=""):
    path = directory / name
    path.mkdir(parents=True)
    (path / "task.toml").write_text(
        f'[metadata]\ncategory = "{category}"\ndifficulty = "{difficulty}"\n{extra}')
    (path / "instruction.md").write_text(f"do {name}\n")
    return path


def mixed(count=40):
    categories = ["software-engineering"] * 4 + ["security", "data-science", "games"]
    difficulties = ["medium", "medium", "hard", "easy"]
    return [TaskInfo(f"task-{index:02d}", categories[index % len(categories)],
                     difficulties[index % len(difficulties)]) for index in range(count)]


def test_same_rule_same_split_and_another_salt_another():
    tasks = mixed()
    first = split_tasks(tasks, tuning_size=10, salt="study-1")
    assert first == split_tasks(list(reversed(tasks)), tuning_size=10, salt="study-1")
    assert first != split_tasks(tasks, tuning_size=10, salt="study-2")


def test_every_task_lands_in_exactly_one_part():
    tasks = mixed()
    split = split_tasks(tasks, tuning_size=10, salt="s")
    assert len(split.tuning) == 10
    assert sorted(split.tuning + split.test) == sorted(item.name for item in tasks)
    assert not set(split.tuning) & set(split.test)


def test_difficulty_quotas_follow_largest_remainder():
    # 40 tasks: 20 medium, 10 hard, 10 easy. Ten seats: 5, 2.5, 2.5 -> floors
    # 5, 2, 2, and the one seat left goes to the tied remainders by name: easy.
    tasks = mixed()
    split = split_tasks(tasks, tuning_size=10, salt="s")
    chosen = {item.name: item for item in tasks if item.name in split.tuning}
    counts = {level: sum(1 for item in chosen.values() if item.difficulty == level)
              for level in ("easy", "medium", "hard")}
    assert counts == {"easy": 3, "medium": 5, "hard": 2}


def quotas(tasks, size):
    levels = sorted({item.difficulty for item in tasks})
    shares = {level: size * sum(item.difficulty == level for item in tasks) / len(tasks)
              for level in levels}
    seats = {level: math.floor(share) for level, share in shares.items()}
    for level in sorted(levels, key=lambda name: (-(shares[name] - seats[name]), name))[
            :size - sum(seats.values())]:
        seats[level] += 1
    return seats


@pytest.mark.parametrize("size", [3, 7, 10, 15])
@pytest.mark.parametrize("salt", [f"salt-{index}" for index in range(6)])
def test_quotas_are_met_whatever_the_order(size, salt):
    tasks = mixed(37)
    split = split_tasks(tasks, tuning_size=size, salt=salt)
    taken = [item for item in tasks if item.name in split.tuning]
    assert len(taken) == size
    assert {level: sum(item.difficulty == level for item in taken)
            for level in quotas(tasks, size)} == quotas(tasks, size)


def test_no_category_beyond_its_share_when_the_others_can_fill():
    tasks = mixed()
    split = split_tasks(tasks, tuning_size=10, salt="s")
    total = len(tasks)
    for category in {item.category for item in tasks}:
        size = sum(1 for item in tasks if item.category == category)
        taken = sum(1 for item in tasks if item.name in split.tuning and item.category == category)
        assert taken <= math.ceil(10 * size / total)


def test_order_inside_a_stratum_is_the_salted_hash():
    tasks = [TaskInfo(f"t{index}", f"c{index}", "medium") for index in range(8)]
    split = split_tasks(tasks, tuning_size=3, salt="pinned")
    ordered = sorted((hashlib.sha256(f"pinned\0{item.name}".encode()).hexdigest(), item.name)
                     for item in tasks)
    assert split.tuning == tuple(sorted(name for _, name in ordered[:3]))


def test_bad_sizes_are_refused():
    with pytest.raises(ValueError):
        split_tasks(mixed(), tuning_size=0, salt="s")
    with pytest.raises(ValueError):
        split_tasks(mixed(), tuning_size=40, salt="s")
    with pytest.raises(ValueError):
        split_tasks([*mixed(), TaskInfo("task-00", "games", "easy")], tuning_size=5, salt="s")


def test_tasks_and_digest_come_from_the_dataset(tmp_path):
    task(tmp_path, "alpha", "security", "hard")
    task(tmp_path, "beta")
    (tmp_path / "README.md").write_text("not a task\n")
    assert load_tasks(tmp_path) == (TaskInfo("alpha", "security", "hard"),
                                    TaskInfo("beta", "software-engineering", "medium"))
    before = dataset_digest(tmp_path)
    assert before.startswith("sha256:") and before == dataset_digest(tmp_path)
    (tmp_path / "beta/instruction.md").write_text("do something else\n")
    assert dataset_digest(tmp_path) != before


def test_a_task_without_category_or_difficulty_is_refused(tmp_path):
    (tmp_path / "gamma").mkdir()
    (tmp_path / "gamma/task.toml").write_text("[metadata]\ncategory = \"games\"\n")
    with pytest.raises(ValueError, match="difficulty"):
        load_tasks(tmp_path)


def make_dataset(tmp_path, count=12):
    dataset = tmp_path / "dataset"
    for index in range(count):
        task(dataset, f"task-{index:02d}", ["security", "games", "software-engineering"][index % 3],
             ["easy", "medium", "hard"][index % 3])
    return dataset


def test_the_record_states_the_rule_and_pins_the_dataset(tmp_path):
    dataset = make_dataset(tmp_path)
    record = make_split_record(dataset, dataset="example/dataset", tuning_size=4, salt="s")
    assert record["dataset"] == "example/dataset"
    assert record["dataset_digest"] == dataset_digest(dataset)
    assert record["rule"]["salt"] == "s" and record["rule"]["tuning_size"] == 4
    assert len(record["tuning"]) == 4 and len(record["test"]) == 8


def write_record(tmp_path, dataset):
    record = make_split_record(dataset, dataset="example/dataset", tuning_size=4, salt="s")
    path = tmp_path / "split.json"
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return path, record


def test_tuning_tasks_are_selected_without_a_study(tmp_path):
    dataset = make_dataset(tmp_path)
    path, record = write_record(tmp_path, dataset)
    chosen = select(path, dataset, "tuning", tmp_path / "run")
    assert chosen == record["tuning"]
    assert sorted(item.name for item in (tmp_path / "run").iterdir()) == record["tuning"]
    assert (tmp_path / "run" / record["tuning"][0] / "task.toml").is_file()


def test_test_tasks_need_a_registered_study_naming_this_split(tmp_path):
    dataset = make_dataset(tmp_path)
    path, record = write_record(tmp_path, dataset)
    studies = tmp_path / "studies"
    studies.mkdir()
    with pytest.raises(SplitRefused, match="registered study"):
        select(path, dataset, "test", tmp_path / "run1", registrations=studies)
    with pytest.raises(SplitRefused, match="not registered"):
        select(path, dataset, "test", tmp_path / "run2", study="go", registrations=studies)
    (studies / "go.json").write_text(json.dumps({"split_sha256": "0" * 64}))
    with pytest.raises(SplitRefused, match="another split"):
        select(path, dataset, "test", tmp_path / "run3", study="go", registrations=studies)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (studies / "go.json").write_text(json.dumps({"split_sha256": digest}))
    assert select(path, dataset, "test", tmp_path / "run4", study="go",
                  registrations=studies) == record["test"]


def test_a_changed_dataset_is_refused(tmp_path):
    dataset = make_dataset(tmp_path)
    path, record = write_record(tmp_path, dataset)
    (dataset / record["tuning"][0] / "instruction.md").write_text("changed\n")
    with pytest.raises(SplitRefused, match="dataset"):
        select(path, dataset, "tuning", tmp_path / "run")


def test_an_existing_target_is_refused(tmp_path):
    dataset = make_dataset(tmp_path)
    path, _ = write_record(tmp_path, dataset)
    (tmp_path / "run").mkdir()
    with pytest.raises(SplitRefused, match="exists"):
        select(path, dataset, "tuning", tmp_path / "run")
