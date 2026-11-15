"""Tests for studies.claude_code.tasks: a seeded chain of small coding tasks, the same for every arm."""

import ast
import re

import pytest

from studies.claude_code import tasks
from studies.claude_code.tasks import Task, make_tasks, template_order, write_starting_files


def test_there_are_at_least_thirty_distinct_templates():
    names = [name for name, _ in tasks.TEMPLATES]
    assert len(names) >= 30
    assert len(set(names)) == len(names)


def test_the_same_seed_gives_the_same_chain():
    assert make_tasks(30, seed=3) == make_tasks(30, seed=3)


def test_a_shorter_chain_is_the_start_of_a_longer_one():
    assert make_tasks(10, seed=5) == make_tasks(30, seed=5)[:10]


def test_another_seed_gives_another_chain():
    assert [t.prompt for t in make_tasks(30, seed=1)] != [t.prompt for t in make_tasks(30, seed=2)]


def test_every_template_is_used_once_before_any_repeats():
    order = template_order(seed=4, count=len(tasks.TEMPLATES) + 5)
    first_round = order[:len(tasks.TEMPLATES)]
    assert sorted(first_round) == sorted(name for name, _ in tasks.TEMPLATES)


def test_a_chain_longer_than_the_templates_still_has_one_folder_per_task():
    chain = make_tasks(len(tasks.TEMPLATES) + 6, seed=9)
    assert len({task.folder for task in chain}) == len(chain)
    assert [task.number for task in chain] == list(range(1, len(chain) + 1))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_prompt_competes_with_the_rules(seed):
    """The rules alone name an ls form, -B, or the cue words a reader keys on."""
    for task in make_tasks(len(tasks.TEMPLATES), seed):
        text = task.prompt.lower()
        assert "from now on" not in text and "for the rest of this session" not in text
        assert not re.search(r"\bls\b", text)
        assert " -b" not in text
        assert "hidden" not in text and "file size" not in text and "sizes" not in text


def test_every_prompt_names_its_own_folder():
    for task in make_tasks(len(tasks.TEMPLATES), seed=7):
        assert f"{task.folder}/" in task.prompt


def test_starting_python_files_parse_and_sit_in_the_task_folder():
    for seed in (0, 1, 2):
        for task in make_tasks(len(tasks.TEMPLATES), seed):
            for path, text in task.files.items():
                assert path.startswith(f"{task.folder}/")
                if path.endswith(".py"):
                    ast.parse(text, filename=path)


def test_planted_bug_tasks_come_with_their_tests():
    for task in make_tasks(len(tasks.TEMPLATES), seed=0):
        if task.template.startswith("fix_"):
            assert any(name.split("/")[-1].startswith("test_") for name in task.files), task.template
            assert "unittest.main()" in "".join(task.files.values())


def test_writing_a_task_makes_its_folder_and_files(tmp_path):
    chain = make_tasks(len(tasks.TEMPLATES), seed=0)
    with_files = next(task for task in chain if task.files)
    without = next(task for task in chain if not task.files)
    written = write_starting_files(tmp_path, with_files)
    assert written and all(path.exists() for path in written)
    assert {path.read_text() for path in written} == set(with_files.files.values())
    write_starting_files(tmp_path, without)
    assert (tmp_path / without.folder).is_dir()


def test_a_path_outside_the_project_is_refused_before_anything_is_written(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    task = Task(1, "x", "t01_x", "Do it.", {"t01_x/ok.txt": "fine\n", "../escaped.txt": "no\n"})
    with pytest.raises(ValueError):
        write_starting_files(project, task)
    assert not (tmp_path / "escaped.txt").exists()
    assert not (project / "t01_x").exists()
