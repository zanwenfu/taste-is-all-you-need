"""The job scripts' own logic, run by bash with stand-ins for the model server, the container and Python: a job that loses a row must say so."""

from __future__ import annotations

import os
import re
import signal
import stat
import subprocess
import time
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
    _stub(bin_dir / "flock", "exit 0")  # not on every machine this test runs on
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
    assert world["log"].read_text().count("studies.memory_probe.merge") == 17, "the eight rows, the eight notes rows, and the summary told to keep rules"


def test_a_shard_that_fails_fails_its_row_and_the_job_with_no_merge(world: dict) -> None:
    done = _run(world, FAIL_WHEN="--arm a5q --condition i --budget 4096 --shard 0/")  # one of the row's two shards, and the first waited for
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
    for row in ("rules/a6-", "rules-in-headers/a6h-", "rules-and-search/a7q-", "turns/a5q-", "notes-newest-told/a2cw-", "flat-newest/a2a-"):
        assert any("studies.memory_probe.judge" in line and row in line for line in calls.splitlines()), row
    for style in ("told", "neutral"):
        assert any("studies.memory_probe.judge" in line and "--notes" in line and f"notes-{style}.jsonl" in line for line in calls.splitlines()), style


def test_the_header_list_goes_to_the_judge_since_a_header_may_say_the_rule_in_passing(world: dict) -> None:
    assert _run(world, SMALL="0").returncode == 0
    calls = world["log"].read_text()
    assert any("studies.memory_probe.judge" in line and "header-list/a3-" in line for line in calls.splitlines())


def test_each_summary_row_has_its_own_summaries_read_by_the_judge(world: dict) -> None:
    assert _run(world).returncode == 0
    judged = [line for line in world["log"].read_text().splitlines() if "studies.memory_probe.judge" in line and "--summaries" in line]
    assert any("/summary/a1-" in line and "cases/summaries.jsonl --prompt anthropic --k 1 " in line for line in judged), judged
    assert any("/summary-told/a1-" in line and "told-anthropic/summaries.jsonl --prompt anthropic --k 1 " in line for line in judged), judged


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
    assert _run(world, ROWS="summary").returncode == 0
    assert len(set(re.findall(r"--workdir (/tmp/aos-probe-1-\d+-summary-\S+)", world["log"].read_text()))) == 2


def test_a_writer_that_fails_is_asked_again_before_any_row_reads_its_file(world: dict, tmp_path: Path) -> None:
    calls = tmp_path / "header-calls"
    python = (Path(world["env"]["STUDY_ROOT"]) / "venv-probe" / "bin" / "python")
    body = python.read_text()
    _stub(python, f"""case "$*" in *studies.memory_probe.headers*) echo x >> {calls}; [ "$(wc -l < {calls})" -lt 3 ] && exit 1;; esac
{body.split(chr(10), 1)[1]}""")
    done = _run(world, ROWS="turns-headers")
    assert done.returncode == 0 and "headers exit 0 tries 3" in done.stdout, done.stdout + done.stderr
    log = world["log"].read_text()
    assert log.index("--arm a4q") > 0 and calls.read_text().count("x") == 3


def test_a_set_with_no_writers_of_its_own_writes_no_told_summaries(world: dict) -> None:
    done = _run(world, WRITERS="0", NOTES_FROM="", HEADERS_FROM="")
    assert "told summaries" not in done.stdout and "summary-told" not in done.stdout


def test_a_seed_stopped_by_its_job_waits_for_its_sessions_to_write_before_their_folder_goes(tmp_path: Path) -> None:
    bin_dir, study = tmp_path / "bin", tmp_path / "study"
    wrote, seen = tmp_path / "wrote", tmp_path / "work-at-write"
    _stub(bin_dir / "curl", "exit 0")
    _stub(bin_dir / "nvidia-smi", "exec sleep 60")
    # A session as the real one behaves: it sets its own handler, stays until told to stop, then takes a moment to write its record, and looks whether its folder is still there.
    python = study / "long" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f"""#!/usr/bin/env python3
import pathlib, signal, sys, time
arguments = sys.argv[1:]
if "studies.long_run.make_session" in arguments:
    sys.exit(0)
work = pathlib.Path(arguments[arguments.index("--workdir") + 1])
work.mkdir(parents=True, exist_ok=True)
def stop(number, frame):
    time.sleep(1)
    if work.is_dir():
        pathlib.Path({str(seen)!r}).write_text("there")
    pathlib.Path({str(wrote)!r}).write_text("done")
    sys.exit(1)
signal.signal(signal.SIGTERM, stop)
time.sleep(60)
""")
    python.chmod(0o755)
    (study / "long" / "venv" / "lib" / "python3.12" / "site-packages" / "minisweagent" / "config").mkdir(parents=True)
    (study / "long" / "venv" / "lib" / "python3.12" / "site-packages" / "minisweagent" / "config" / "mini.yaml").write_text("")
    (study / "code-abcdefabcdef").mkdir()
    job = f"stand-in-stop-{os.getpid()}"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "SLURM_JOB_ID": job, "ARMS": "compaction", "SERVER_URL": "http://127.0.0.1:1/v1", "SEED": "2"}
    # In a session of its own: the signal below goes to this job's own group, which Slurm would signal, and to nothing else.
    running = subprocess.Popen(["bash", str(ROOT / "studies/long_run/jobs/seed.sbatch")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    assert running.pid > 1 and os.getpgid(running.pid) == running.pid and running.pid != os.getpgrp()
    deadline = time.monotonic() + 30
    while not Path(f"/tmp/aos-long-{job}-gpt-oss-anthropic-seed2/compaction").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert Path(f"/tmp/aos-long-{job}-gpt-oss-anthropic-seed2/compaction").exists()
    time.sleep(0.5)  # the stand-in has set its handler
    os.killpg(running.pid, signal.SIGTERM)
    output, _ = running.communicate(timeout=30)
    assert wrote.read_text() == "done" and seen.read_text() == "there", output
    assert running.returncode == 143 and not Path(f"/tmp/aos-long-{job}-gpt-oss-anthropic-seed2").exists()


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
    _stub(study / "venv-probe" / "bin" / "python", '''case "$*" in *code_digest*) echo "the code's digest";; esac
exit 0''')
    views = study / "bench" / "runs" / "dev-abcdefabcdef" / "views"
    views.mkdir(parents=True)
    if view:
        (views / "lme.jsonl").write_text("{}\n")
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}", "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "DIGEST": "the code's digest", "SLURM_JOB_ID": "stand-in-job", "PART": "dev", "VIEW": "lme", "TOTAL": "2"}


def _job(script: str, env: dict) -> subprocess.CompletedProcess:
    # In a session of its own, so that what a job does to its own process group stops there.
    return subprocess.run(["bash", str(ROOT / script)], env={**os.environ, **env}, capture_output=True, text=True, timeout=120, start_new_session=True)


def test_a_mem0_job_with_no_embedder_reaches_its_end_rather_than_stopping_itself(tmp_path: Path) -> None:
    done = _job("studies/memory_bench/jobs/mem0.sbatch", _bench_world(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "shards ended" in done.stdout and "\nend " in done.stdout


def test_a_bench_job_on_a_copy_of_the_code_that_is_not_the_commit_stops_before_anything(tmp_path: Path) -> None:
    done = _job("studies/memory_bench/jobs/mem0.sbatch", {**_bench_world(tmp_path), "DIGEST": "another"})
    assert done.returncode == 1 and "is not commit" in done.stdout, done.stdout + done.stderr
    assert "start " not in done.stdout


def test_a_shard_that_did_not_finish_fails_the_mem0_job(tmp_path: Path) -> None:
    world = _bench_world(tmp_path)
    _stub(Path(world["PATH"].split(":")[0]) / "apptainer", """case "$*" in
  *"vllm serve"*) exec sleep 60;;
  *) echo 'it broke';;
esac""")
    done = _job("studies/memory_bench/jobs/mem0.sbatch", world)
    assert done.returncode == 1 and "DID NOT FINISH" in done.stdout and "shards failed 2" in done.stdout, done.stdout + done.stderr


def test_a_later_commit_carries_on_the_run_it_is_told_to(tmp_path: Path) -> None:
    world = _bench_world(tmp_path)
    study = Path(world["STUDY_ROOT"])
    views = study / "bench" / "runs" / "dev-first" / "views"
    views.mkdir(parents=True)
    (views / "lme.jsonl").write_text("{}\n")
    (study / "bench" / "runs" / "dev-abcdefabcdef" / "views" / "lme.jsonl").unlink()
    done = _job("studies/memory_bench/jobs/mem0.sbatch", {**world, "RUN": "first"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert (study / "bench" / "runs" / "dev-first" / "rankings" / "lme" / "mem0").is_dir()


def _records_world(tmp_path: Path, fail_when: str = "", part: str = "dev") -> dict:
    world = _bench_world(tmp_path)
    study = Path(world["STUDY_ROOT"])
    run = study / "bench" / "runs" / f"{part}-abcdefabcdef"
    (run / "views").mkdir(parents=True, exist_ok=True)
    for view in ("lme", "locomo"):
        (run / "views" / f"{view}.jsonl").write_text("{}\n")
        (run / "rankings" / view / "mem0").mkdir(parents=True)
        (run / "rankings" / view / "mem0" / "shard0.jsonl").write_text("{}\n")
    log = tmp_path / "calls.log"
    _stub(study / "venv-probe" / "bin" / "python", f'''echo "$*" >> {log}
case "$*" in *code_digest*) echo "the code's digest"; exit 0;; esac
case "$*" in *"{fail_when or 'never'}"*) echo broke; exit 3;; esac
case "$*" in *" views "*) shift; while [ "$1" != "--out" ]; do shift; done; echo "{{}}" > "$2"; exit 0;; esac
exit 0''')
    _stub(Path(world["PATH"].split(":")[0]) / "cmp", '''exec /usr/bin/cmp "$@"''')
    return {**world, "PART": part, "SWEEP": "0", "SLURM_CPUS_PER_TASK": "4"}


def test_a_row_that_fails_fails_the_records_job_so_nothing_reads_on(tmp_path: Path) -> None:
    done = _job("studies/memory_bench/jobs/records.sbatch", _records_world(tmp_path, fail_when="--system flat_bm25 --budget 8000"))
    assert done.returncode == 1 and "FAILED lme flat_bm25 8000" in done.stdout, done.stdout + done.stderr


def test_a_clean_records_job_passes_and_asks_for_the_whole_row_list(tmp_path: Path) -> None:
    done = _job("studies/memory_bench/jobs/records.sbatch", _records_world(tmp_path))
    assert done.returncode == 0 and "rows failed 0" in done.stdout, done.stdout + done.stderr
    calls = (tmp_path / "calls.log").read_text()
    assert "--system full" in calls and "--budget 118000" in calls and "--budget 126000" not in calls and "--workers 4" in calls


def test_views_that_differ_from_those_the_rankings_came_from_stop_the_records_job(tmp_path: Path) -> None:
    world = _records_world(tmp_path)
    (Path(world["STUDY_ROOT"]) / "bench" / "runs" / "dev-abcdefabcdef" / "views" / "lme.jsonl").write_text("another\n")
    done = _job("studies/memory_bench/jobs/records.sbatch", world)
    assert done.returncode == 1 and "not the ones the rankings came from" in done.stdout, done.stdout + done.stderr


def test_on_the_held_out_part_the_records_job_names_the_commit_and_its_digest(tmp_path: Path) -> None:
    done = _job("studies/memory_bench/jobs/records.sbatch", _records_world(tmp_path, part="held_out"))
    assert done.returncode == 0, done.stdout + done.stderr
    calls = (tmp_path / "calls.log").read_text()
    assert "--frozen-config abcdefabcdef0000 --frozen-digest the code's digest" in calls or "--frozen-digest the" in calls


def test_the_judges_read_the_file_the_prompts_came_from() -> None:
    env = (ROOT / "studies/memory_bench/jobs/env.sh").read_text()
    assert "LME_REF=$LME_DATA" in env and "oracle" not in env.split("LME_REF=")[1].splitlines()[0]


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
