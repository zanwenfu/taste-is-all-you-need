"""The job scripts' own logic, run by bash with stand-ins for the model server, the container and Python: a job that loses a row must say so."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _stub(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def world(tmp_path: Path) -> dict:
    log = tmp_path / "calls.log"
    bin_dir, study = tmp_path / "bin", tmp_path / "study"
    _stub(bin_dir / "curl", "exit 0")
    _stub(bin_dir / "nvidia-smi", "exit 0")
    _stub(bin_dir / "apptainer", 'while [[ "$1" != */python ]]; do shift; done; exec "$@"')
    python = f"""echo "$*" >> {log}
case "$*" in
  *"$FAIL_WHEN"*) [ -n "$FAIL_WHEN" ] && exit 3;;
esac
case "$*" in *studies.memory_probe.merge*) echo '{{"merged": true}}';; esac
exit 0"""
    for venv in ("venv-probe", "venv"):
        _stub(study / venv / "bin" / "python", python)
    (study / "code-abcdefabcdef").mkdir(parents=True)
    (study / "cases").mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "SLURM_JOB_ID": "1", "PR": "anthropic",
           "K": "1", "IN": str(study / "cases"), "OUT": str(study / "out"), "SERVER_URL": "http://127.0.0.1:1/v1", "PROCS": "2", "SMALL": "1", "FAIL_WHEN": ""}
    (study / "out").mkdir()
    return {"env": env, "log": log}


def _run(world: dict, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(ROOT / "studies/memory_probe/jobs/cell.sbatch")], env={**world["env"], **extra}, capture_output=True, text=True, timeout=120)


def test_a_clean_cell_exits_zero_and_merges_every_row(world: dict) -> None:
    done = _run(world)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "rows failed 0" in done.stdout
    assert world["log"].read_text().count("studies.memory_probe.merge") >= 10


def test_a_shard_that_fails_fails_its_row_and_the_job_with_no_merge(world: dict) -> None:
    done = _run(world, FAIL_WHEN="--arm a5q")
    assert done.returncode == 1 and "rows failed 1" in done.stdout
    assert not any("studies.memory_probe.merge" in line and "/turns/" in line for line in world["log"].read_text().splitlines())


def test_a_judge_that_fails_fails_the_job(world: dict) -> None:
    done = _run(world, FAIL_WHEN="studies.memory_probe.judge --results")
    assert done.returncode == 1 and "rows failed 0" not in done.stdout


def test_the_rows_asked_for_are_the_only_ones_run(world: dict) -> None:
    done = _run(world, ROWS="turns rules")
    assert done.returncode == 0, done.stdout + done.stderr
    calls = world["log"].read_text()
    assert "--arm a5q" in calls and "--arm a6 " in calls
    assert "--arm a1 " not in calls and "--arm a2cq" not in calls
    assert "studies.memory_probe.notes" not in calls and "studies.memory_probe.headers" not in calls, "these rows need neither"


def test_every_rule_list_row_and_every_notes_file_goes_to_the_judge(world: dict) -> None:
    assert _run(world).returncode == 0
    calls = world["log"].read_text()
    for row in ("rules/a6-", "rules-in-headers/a6h-", "rules-and-search/a7q-", "turns/a5q-", "notes-newest-told/a2cw-"):
        assert any("studies.memory_probe.judge" in line and row in line for line in calls.splitlines()), row
    for style in ("told", "neutral"):
        assert any("studies.memory_probe.judge" in line and "--notes" in line and f"notes-{style}.jsonl" in line for line in calls.splitlines()), style


def test_a_summary_told_to_keep_rules_gets_its_own_row_at_one_round(world: dict) -> None:
    assert _run(world).returncode == 0
    calls = world["log"].read_text()
    assert "studies.memory_probe.told_summary" in calls
    assert any("--arm a1 " in line and "told-anthropic" in line for line in calls.splitlines())
    world["log"].write_text("")
    assert _run(world, K="16").returncode == 0
    assert "told_summary" not in world["log"].read_text()


def test_a_cued_set_keeps_the_published_notes_and_headers(world: dict) -> None:
    assert _run(world, WRITERS="0").returncode == 0
    calls = world["log"].read_text()
    assert "studies.memory_probe.notes" not in calls and "studies.memory_probe.headers" not in calls
    assert "--notes" not in [word for line in calls.splitlines() if "studies.memory_probe.judge" in line for word in line.split()]


def test_a_long_run_seed_says_when_an_arm_failed(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    bin_dir, study = tmp_path / "bin", tmp_path / "study"
    _stub(bin_dir / "curl", "exit 0")
    _stub(bin_dir / "nvidia-smi", "exit 0")
    _stub(bin_dir / "apptainer", "exec sleep 1000")
    _stub(study / "long" / "venv" / "bin" / "python", f"""echo "$*" >> {log}
case "$*" in *"--arm taste "*) exit 1;; esac
exit 0""")
    (study / "long" / "venv" / "lib" / "python3.12" / "site-packages" / "minisweagent" / "config").mkdir(parents=True)
    (study / "long" / "venv" / "lib" / "python3.12" / "site-packages" / "minisweagent" / "config" / "mini.yaml").write_text("")
    (study / "code-abcdefabcdef").mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "SLURM_JOB_ID": "1", "SEED": "2",
           "ARMS": "compaction taste"}
    done = subprocess.run(["bash", str(ROOT / "studies/long_run/jobs/seed.sbatch")], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 1 and "arms failed 1" in done.stdout, done.stdout + done.stderr


def test_seeds_sharing_a_server_keep_folders_of_their_own_and_leave_nothing_on_the_node(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    bin_dir, study = tmp_path / "bin", tmp_path / "study"
    _stub(bin_dir / "curl", "exit 0")
    _stub(bin_dir / "nvidia-smi", "exec sleep 1000")
    _stub(bin_dir / "apptainer", f'echo "$*" >> {log}; exit 0')
    _stub(study / "long" / "venv" / "bin" / "python", f"""echo "TMPDIR=$TMPDIR $*" >> {log}
exit 0""")
    (study / "long" / "venv" / "lib" / "python3.12" / "site-packages" / "minisweagent" / "config").mkdir(parents=True)
    (study / "long" / "venv" / "lib" / "python3.12" / "site-packages" / "minisweagent" / "config" / "mini.yaml").write_text("")
    (study / "code-abcdefabcdef").mkdir()
    job = f"stand-in-{os.getpid()}"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "SLURM_JOB_ID": job, "ARMS": "compaction", "SERVER_URL": "http://127.0.0.1:1/v1"}
    for seed in ("2", "3"):
        done = subprocess.run(["bash", str(ROOT / "studies/long_run/jobs/seed.sbatch")], env={**env, "SEED": seed}, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0 and "using the running server" in done.stdout, done.stdout + done.stderr
        assert not Path(f"/tmp/aos-long-{job}-gpt-oss-anthropic-seed{seed}").exists(), "a seed's machines go with it"
    said = log.read_text()
    assert "vllm serve" not in said and said.count("--url http://127.0.0.1:1/v1") == 2
    for seed in ("2", "3"):
        assert f"TMPDIR=/tmp/aos-long-{job}-gpt-oss-anthropic-seed{seed}/machines" in said and f"--workdir /tmp/aos-long-{job}-gpt-oss-anthropic-seed{seed}/compaction" in said


def test_two_cells_on_one_job_do_not_share_a_working_folder(world: dict) -> None:
    assert _run(world, ROWS="summary").returncode == 0
    assert re.search(r"--workdir /tmp/aos-probe-1-\d+-summary-", world["log"].read_text())


def _bench_world(tmp_path: Path, view: bool = True) -> dict:
    bin_dir, study = tmp_path / "bin", tmp_path / "study"
    code = study / "code-abcdefabcdef" / "studies" / "memory_bench" / "jobs"
    code.mkdir(parents=True)
    (code / "env.sh").write_text((ROOT / "studies/memory_bench/jobs/env.sh").read_text())
    _stub(bin_dir / "curl", "exit 0")
    _stub(bin_dir / "nvidia-smi", "exec sleep 60")
    # The served model stays up until it is told to stop, as a real one does. Anything else asked of the container is a shard that finishes.
    _stub(bin_dir / "apptainer", """case "$*" in
  *"vllm serve"*) exec sleep 60;;
  *) echo '{"ran": 1}';;
esac""")
    _stub(study / "venv-probe" / "bin" / "python", "exit 0")
    views = study / "bench" / "runs" / "dev-abcdefabcdef" / "views"
    views.mkdir(parents=True)
    if view:
        (views / "lme.jsonl").write_text("{}\n")
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}", "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "SLURM_JOB_ID": "stand-in-job", "PART": "dev", "VIEW": "lme", "TOTAL": "2"}


def _job(script: str, env: dict) -> subprocess.CompletedProcess:
    # In a session of its own, so that what a job does to its own process group stops there.
    return subprocess.run(["bash", str(ROOT / script)], env={**os.environ, **env}, capture_output=True, text=True, timeout=120, start_new_session=True)


def test_a_mem0_job_with_no_embedder_reaches_its_end_rather_than_stopping_itself(tmp_path: Path) -> None:
    done = _job("studies/memory_bench/jobs/mem0.sbatch", _bench_world(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "shards ended" in done.stdout and "\nend " in done.stdout


def test_a_bench_job_that_fails_says_so_and_exits_one_rather_than_stopping_itself(tmp_path: Path) -> None:
    done = _job("studies/memory_bench/jobs/mem0.sbatch", _bench_world(tmp_path, view=False))
    assert done.returncode == 1, done.stdout + done.stderr
    assert "FAILED: no views file" in done.stdout


@pytest.mark.parametrize("script", sorted(p.relative_to(ROOT).as_posix() for folder in ("memory_bench", "memory_probe", "long_run") for p in (ROOT / "studies" / folder / "jobs").iterdir()))
def test_no_job_script_can_aim_a_signal_at_its_own_process_group_or_everything(script: str) -> None:
    for line in (ROOT / script).read_text().splitlines():
        for part in line.split(";"):
            words = part.split()
            if "kill" in words and "-0" not in words:
                aimed = words[words.index("kill") + 1:]
                assert not any(word in ("0", "-1") or word.endswith(":-0}") for word in aimed), line
