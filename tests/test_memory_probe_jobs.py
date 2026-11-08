"""The job scripts' own logic, run by bash with stand-ins for the model server, the container and Python: a job that loses a row must say so."""

from __future__ import annotations

import contextlib
import os
import re
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

from studies.memory_bench.analyze import readers_of, required_rows

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


def _agent_calls(calls: str) -> list[str]:
    return [line for line in calls.splitlines() if any(step in line for step in ("studies.memory_probe.run ", "studies.memory_probe.notes ", "studies.memory_probe.told_summary "))]


def test_by_default_the_study_model_answers_with_the_package_settings(world: dict) -> None:
    assert _run(world).returncode == 0
    calls = world["log"].read_text()
    agent = _agent_calls(calls)
    assert agent and all("--model openai/gpt-oss-120b --effort low " in line for line in agent), agent[:3]
    assert not any("--max-tokens" in line or "--output-cap" in line for line in agent)
    assert "studies.memory_probe.headers" in calls and "studies.memory_probe.judge" in calls


def test_another_agent_answers_writes_its_own_notes_and_summaries_and_is_judged_apart(world: dict, tmp_path: Path) -> None:
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "turn-headers.jsonl").write_text("{}\n")
    done = _run(world, AGENT_MODEL="poolside/laguna-s-2.1", AGENT_EFFORT="none", AGENT_CAP="16384", JUDGE="later", HEADERS_FROM=str(kept), SMALL="0")
    assert done.returncode == 0, done.stdout + done.stderr
    calls = world["log"].read_text()
    agent = _agent_calls(calls)
    rows = [line for line in agent if "studies.memory_probe.run " in line]
    assert rows and all("--model poolside/laguna-s-2.1 --effort none --max-tokens 16384 " in line for line in rows)
    assert all("--model poolside/laguna-s-2.1 --effort none --max-tokens 16384 " in line for line in agent if "studies.memory_probe.notes " in line)
    assert any("studies.memory_probe.told_summary " in line and "--model poolside/laguna-s-2.1 --effort none --output-cap 16384 " in line for line in agent)
    assert "studies.memory_probe.judge" not in calls, "judged apart, on gpt-oss-120b"
    assert "studies.memory_probe.headers" not in calls, "the memory's own headers, as written for the study's model"
    assert (Path(world["env"]["OUT"]) / "turn-headers.jsonl").read_text() == "{}\n"


def _inputs(tmp_path: Path, **extra: str) -> tuple[subprocess.CompletedProcess, str, Path]:
    """inputs.sbatch with stand-ins: its Python logs each call and, as the export, writes the file the job checks for."""
    log, bin_dir, study = tmp_path / "inputs-calls.log", tmp_path / "ibin", tmp_path / "istudy"
    for name in ("curl", "nvidia-smi"):
        _stub(bin_dir / name, "exit 0")
    _stub(bin_dir / "apptainer", 'while [[ "$1" != */python ]]; do shift; done; exec "$@"')
    _stub(study / "venv" / "bin" / "python", f"""echo "$*" >> {log}
case "$*" in *studies.memory_probe.export*) out=${{@: -1}}; mkdir -p "$out"; echo x > "$out/delivered.jsonl";; esac
exit 0""")
    (study / "code-abcdefabcdef").mkdir(parents=True, exist_ok=True)
    out = study / "out"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "SLURM_JOB_ID": "1",
           "VARIANT": "revised", "OUT": str(out), "SERVER_URL": "http://127.0.0.1:1/v1", **extra}
    done = subprocess.run(["bash", str(ROOT / "studies/memory_probe/jobs/inputs.sbatch")], env=env, capture_output=True, text=True, timeout=120)
    return done, log.read_text() if log.exists() else "", out


def test_the_baselines_by_default_ask_the_study_model_with_the_package_settings(tmp_path: Path) -> None:
    done, calls, _ = _inputs(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    bridge = [line for line in calls.splitlines() if "conpact_bridge" in line]
    assert bridge and "--probe-model openai/gpt-oss-120b --effort low --output-cap 4096 " in bridge[0]
    assert "studies.context_rot.repeated" in calls
    assert all("--variant revised " in line for line in calls.splitlines()), "a variant reaches every step"


def test_another_agents_baselines_keep_the_study_readers_records(tmp_path: Path) -> None:
    kept = tmp_path / "kept"
    for part in ("h00-05", "h05-10"):  # the published cases' records were read in two slices
        (kept / "repeated" / "anthropic" / part).mkdir(parents=True)
        (kept / "repeated" / "anthropic" / part / "records.json").write_text(part)
    done, calls, out = _inputs(tmp_path, PROBE_MODEL="poolside/laguna-s-2.1", PROBE_EFFORT="none", PROBE_CAP="16384", REPEATED_FROM=str(kept), VARIANT="published")
    assert done.returncode == 0, done.stdout + done.stderr
    bridge = [line for line in calls.splitlines() if "conpact_bridge" in line]
    assert bridge and "--probe-model poolside/laguna-s-2.1 --effort none --output-cap 16384 " in bridge[0]
    assert "studies.context_rot.repeated" not in calls
    assert all((out / "repeated" / "anthropic" / part / "records.json").read_text() == part for part in ("h00-05", "h05-10"))
    assert "--variant" not in calls, "the published cases, unchanged"
    for missing, said in (("REPEATED_FROM", "rule records"), ("SERVER_URL", "SERVER_URL")):
        settings = {"PROBE_MODEL": "poolside/laguna-s-2.1", "REPEATED_FROM": str(kept)} | {missing: ""}
        refused, refused_calls, _ = _inputs(tmp_path / missing, **settings)
        assert refused.returncode == 1 and said in refused.stdout, (missing, refused.stdout)
        assert "conpact_bridge" not in refused_calls


def test_another_agent_refuses_without_the_memorys_headers_a_running_server_or_judging_apart(world: dict, tmp_path: Path) -> None:
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "turn-headers.jsonl").write_text("{}\n")
    other = {"AGENT_MODEL": "poolside/laguna-s-2.1", "JUDGE": "later", "HEADERS_FROM": str(kept)}
    for missing, said in (("HEADERS_FROM", "headers"), ("JUDGE", "judged apart"), ("SERVER_URL", "SERVER_URL")):
        done = _run(world, **{**other, missing: ""})
        assert done.returncode == 1 and said in done.stdout, (missing, done.stdout)
        assert "studies.memory_probe.run " not in world["log"].read_text() if world["log"].exists() else True


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


def test_the_long_runs_shared_server_has_the_same_settings_as_a_seeds_own() -> None:
    seed = (ROOT / "studies/long_run/jobs/seed.sbatch").read_text()
    serve = (ROOT / "studies/long_run/jobs/serve.sbatch").read_text()
    common = "--max-model-len 131072 --gpu-memory-utilization 0.92 --enable-auto-tool-choice"
    laguna, oss = "--tensor-parallel-size 2 --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --default-chat-template-kwargs", "--tool-call-parser openai"
    assert common in seed and laguna in seed and f"SERVE=({oss})" in seed
    assert f"SERVE=({common} {laguna}" in serve and f"SERVE=({common} {oss})" in serve


def test_the_shared_llama_server_has_the_same_settings_as_the_readers_own() -> None:
    env = (ROOT / "studies/memory_bench/jobs/env.sh").read_text()
    serve = (ROOT / "studies/long_run/jobs/serve.sbatch").read_text()
    line = "--tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.92"
    assert line in env and line in serve


def test_a_llama_reader_uses_a_running_server_when_it_is_told_to(tmp_path: Path) -> None:
    world = _bench_world(tmp_path)
    (Path(world["STUDY_ROOT"]) / "code-abcdefabcdef" / "studies" / "memory_bench" / "jobs" / "read.sbatch").write_text((ROOT / "studies/memory_bench/jobs/read.sbatch").read_text())
    done = _job("studies/memory_bench/jobs/read.sbatch", {**world, "READER": "llama", "SERVER_URL": "http://127.0.0.1:4242/v1", "SLURM_JOB_ID": "7"})
    assert "using the running server at http://127.0.0.1:4242/v1" in done.stdout, done.stdout + done.stderr
    assert "load_seconds" not in done.stdout


# ------------------------------------------------------------------ GPUs kept for the study: never idle, and work that needs no login node


def _keep(tmp_path: Path, *, util: str = "0", **more: str) -> tuple[dict, Path, Path]:
    bin_dir, study, ended = tmp_path / "bin", tmp_path / "study", tmp_path / "scancel.log"
    (tmp_path / "util").write_text(util + "\n")
    _stub(bin_dir / "nvidia-smi", f'''case "$*" in *" -l "*) exec sleep 30;; esac
cat {tmp_path / "util"}''')
    _stub(bin_dir / "scancel", f'echo "$*" >> {ended}')
    _stub(bin_dir / "setsid", 'exec python3 -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" "$@"')
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUDY_ROOT": str(study), "SLURM_JOB_ID": "77", "TICK": "0.1", "IDLE_SECONDS": "2", "HUNG_SECONDS": "5", **more}
    tasks = study / "logs" / "gpus-77" / "tasks"
    tasks.mkdir(parents=True)
    return env, tasks, ended


def _start_keep(env: dict) -> subprocess.Popen:
    return subprocess.Popen(["bash", str(ROOT / "studies/long_run/jobs/keep.sbatch")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def _until(test, seconds: float = 12.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if test():
            return True
        time.sleep(0.05)
    return test()


def _end_keep(job: subprocess.Popen, tasks: Path) -> str:
    """Stop what this test started, by exact ids: the job, then any task it left."""
    for pid in tasks.glob("*.pid"):
        if not pid.with_suffix(".exit").exists():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(int(pid.read_text()), signal.SIGTERM)
    if job.poll() is None:
        job.terminate()
    return job.communicate(timeout=20)[0]


def test_kept_gpus_with_nothing_to_do_end_the_job_by_its_own_id(tmp_path: Path) -> None:
    env, tasks, ended = _keep(tmp_path)
    job = _start_keep(env)
    assert job.wait(timeout=15) == 0
    out = _end_keep(job, tasks)
    assert ended.read_text().split() == ["77"] and "ending: its GPUs at rest and no work" in out


def test_a_working_gpu_keeps_the_job_and_the_clock_starts_again_when_it_rests(tmp_path: Path) -> None:
    env, tasks, ended = _keep(tmp_path, util="0\n80")
    job = _start_keep(env)
    time.sleep(3.5)
    assert job.poll() is None and not ended.exists(), "one GPU at work of two is work"
    (tmp_path / "util").write_text("0\n1\n")
    assert job.wait(timeout=15) == 0 and ended.read_text().split() == ["77"]
    _end_keep(job, tasks)


def test_a_loaded_server_with_no_one_asking_is_idle(tmp_path: Path) -> None:
    env, tasks, ended = _keep(tmp_path)
    (tasks / "server-laguna.sh").write_text("exec sleep 30\n")
    job = _start_keep(env)
    try:
        assert job.wait(timeout=15) == 0 and ended.read_text().split() == ["77"]
    finally:
        _end_keep(job, tasks)


def test_work_on_the_cpu_keeps_the_job_until_the_longer_limit(tmp_path: Path) -> None:
    env, tasks, ended = _keep(tmp_path)
    (tasks / "records.sh").write_text("exec sleep 30\n")
    job = _start_keep(env)
    try:
        time.sleep(3.5)
        assert job.poll() is None and not ended.exists(), "past the idle limit, with a task at work"
        assert job.wait(timeout=15) == 0 and ended.read_text().split() == ["77"], "and a task that does nothing for the longer limit does not hold the GPUs"
    finally:
        out = _end_keep(job, tasks)
    assert "whatever says it is running" in out


def test_when_nvidia_smi_cannot_say_the_job_is_not_ended(tmp_path: Path) -> None:
    env, tasks, ended = _keep(tmp_path, util="not a number")
    job = _start_keep(env)
    time.sleep(3.5)
    assert job.poll() is None and not ended.exists()
    out = _end_keep(job, tasks)
    assert out.count("nvidia-smi cannot say") == 1


def test_a_task_handed_over_before_the_job_starts_runs_once_and_says_how_it_ended(tmp_path: Path) -> None:
    env, tasks, _ = _keep(tmp_path, util="50")
    (tmp_path / "ran").write_text("")
    (tasks / "first.sh").write_text(f"echo ran >> {tmp_path / 'ran'}\necho its words\nexit 3\n")
    job = _start_keep(env)
    try:
        assert _until((tasks / "first.exit").exists)
        time.sleep(0.5)
        assert (tasks / "first.exit").read_text().strip() == "3" and (tasks / "first.out").read_text() == "its words\n"
        assert (tmp_path / "ran").read_text() == "ran\n", "a task runs once"
    finally:
        _end_keep(job, tasks)


def test_a_stop_file_stops_that_task_and_all_it_started_and_no_other(tmp_path: Path) -> None:
    env, tasks, _ = _keep(tmp_path, util="50")
    child = tmp_path / "child-done"
    (tasks / "a.sh").write_text(f"(sleep 3; touch {child}) &\nwait\n")
    (tasks / "b.sh").write_text("exec sleep 30\n")
    job = _start_keep(env)
    try:
        assert _until(lambda: (tasks / "a.pid").exists() and (tasks / "b.pid").exists())
        (tasks / "a.stop").touch()
        assert _until((tasks / "a.exit").exists)
        assert not (tasks / "b.exit").exists() and job.poll() is None
        time.sleep(3.5)
        assert not child.exists(), "what the task started goes with it"
    finally:
        _end_keep(job, tasks)


def test_a_task_killed_outright_is_over_and_does_not_hold_the_job(tmp_path: Path) -> None:
    env, tasks, ended = _keep(tmp_path, HUNG_SECONDS="600")
    (tasks / "a.sh").write_text("exec sleep 30\n")
    job = _start_keep(env)
    try:
        assert _until((tasks / "a.pid").exists)
        time.sleep(0.5)
        os.killpg(int((tasks / "a.pid").read_text()), signal.SIGKILL)
        assert _until((tasks / "a.exit").exists) and (tasks / "a.exit").read_text().strip() == "gone"
        assert job.wait(timeout=15) == 0 and ended.read_text().split() == ["77"], "with nothing left the idle limit applies"
    finally:
        _end_keep(job, tasks)


def test_a_task_stopped_as_it_starts_is_over_even_if_it_never_ran(tmp_path: Path) -> None:
    env, tasks, _ = _keep(tmp_path, util="50")
    (tasks / "a.sh").write_text("exec sleep 30\n")
    (tasks / "a.stop").touch()
    job = _start_keep(env)
    try:
        assert _until((tasks / "a.exit").exists), "stopped before or after its wrapper was ready"
    finally:
        _end_keep(job, tasks)


def test_a_task_is_handed_over_whole_and_never_over_another(tmp_path: Path) -> None:
    study = tmp_path / "study"
    give = ["bash", str(ROOT / "studies/long_run/jobs/give_task.sh"), "77", "one"]
    env = {**os.environ, "STUDY_ROOT": str(study)}
    assert subprocess.run(give, input="echo first\n", env=env, capture_output=True, text=True).returncode == 0
    again = subprocess.run(give, input="echo second\n", env=env, capture_output=True, text=True)
    assert again.returncode == 1 and "already has a task" in again.stderr
    tasks = study / "logs" / "gpus-77" / "tasks"
    assert (tasks / "one.sh").read_text() == "echo first\n" and not list(tasks.glob("*.writing"))
    empty = subprocess.run([*give[:-1], "two"], input="", env=env, capture_output=True, text=True)
    assert empty.returncode == 1 and not (tasks / "two.sh").exists()


def _url_world(tmp_path: Path) -> tuple[dict, Path, Path]:
    study = tmp_path / "study"
    tasks, url = study / "logs" / "gpus-77" / "tasks", study / "logs" / "long-serve-77-laguna" / "url"
    tasks.mkdir(parents=True)
    url.parent.mkdir(parents=True)
    return {**os.environ, "STUDY_ROOT": str(study), "SERVER_POLL": "0.05", "SERVER_WAIT": "1"}, tasks, url


def test_a_servers_address_comes_when_it_has_loaded_and_a_dead_server_says_so(tmp_path: Path) -> None:
    env, tasks, url = _url_world(tmp_path)
    run = ["bash", str(ROOT / "studies/long_run/jobs/server_url.sh"), "77", "laguna"]
    late = subprocess.run(run, env=env, capture_output=True, text=True, timeout=20)
    assert late.returncode == 1 and "after 1 seconds" in late.stderr
    (tasks / "server-laguna.pid").write_text("1")
    url.write_text("http://127.0.0.1:9/v1\n")
    assert subprocess.run(run, env=env, capture_output=True, text=True, timeout=20).stdout.strip() == "http://127.0.0.1:9/v1"
    (tasks / "server-laguna.exit").write_text("1")
    dead = subprocess.run(run, env={**env, "SERVER_WAIT": "60"}, capture_output=True, text=True, timeout=20)
    assert dead.returncode == 1 and "task has ended" in dead.stderr, "whatever address it left behind"
    (tasks / "server-laguna-2.pid").write_text("2")
    assert subprocess.run(run, env=env, capture_output=True, text=True, timeout=20).returncode == 0, "and a later server on the same GPUs will do"


def _stage_env(tmp_path: Path, script: str, body: str = "exit 0", **more: str) -> dict:
    study = tmp_path / "study"
    code = study / "code-abcdefabcdef" / "studies"
    (code / "memory_bench" / "jobs").mkdir(parents=True, exist_ok=True)
    (code / "long_run" / "jobs").mkdir(parents=True, exist_ok=True)
    for helper in ("server_url.sh", "job_gone.sh"):
        (code / "long_run" / "jobs" / helper).write_text((ROOT / "studies/long_run/jobs" / helper).read_text())
    (code / "memory_bench" / "jobs" / script).write_text(f"echo ran >> {tmp_path / 'ran'}\necho url ${{SERVER_URL:-none}} >> {tmp_path / 'ran'}\n{body}\n")
    return {**os.environ, "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "PART": "dev", "RUN": "r", "SLURM_JOB_ID": "77", "STAGE_POLL": "0.05", "SERVER_POLL": "0.05",
            "STAGE_BEAT": "0.2", "STAGE_CONFIRM_SECONDS": "0", **more}


def _stage(tmp_path: Path, name: str, script: str, after: str = "", *, body: str = "exit 0", server: str = "none", **more: str) -> subprocess.CompletedProcess:
    env = _stage_env(tmp_path, script, body, **more)
    return subprocess.run(["bash", str(ROOT / "studies/memory_bench/jobs/stage.sh"), name, server, script, after], env=env, capture_output=True, text=True, timeout=20)


def _marks(tmp_path: Path) -> Path:
    marks = tmp_path / "study" / "bench" / "runs" / "dev-r" / "stages"
    marks.mkdir(parents=True, exist_ok=True)
    return marks


def test_a_stage_runs_its_job_script_and_leaves_a_mark(tmp_path: Path) -> None:
    done = _stage(tmp_path, "records", "records.sbatch")
    assert done.returncode == 0 and (_marks(tmp_path) / "records.ok").exists()
    failed = _stage(tmp_path, "read", "read.sbatch", body="exit 4")
    assert failed.returncode == 4 and (_marks(tmp_path) / "read.failed").exists() and not (_marks(tmp_path) / "read.ok").exists()


def test_a_stage_says_while_it_runs_which_job_it_is_on(tmp_path: Path) -> None:
    seen = tmp_path / "seen"
    done = _stage(tmp_path, "read", "read.sbatch", body=f"cat {tmp_path / 'study' / 'bench' / 'runs' / 'dev-r' / 'stages' / 'read.running'} > {seen}")
    assert done.returncode == 0 and seen.read_text().strip() == "77"
    assert not (_marks(tmp_path) / "read.running").exists() and (_marks(tmp_path) / "read.ok").read_text().strip() == "abcdefabcdef0000"


def _no_such_job(tmp_path: Path) -> dict:
    _stub(tmp_path / "bin" / "squeue", 'echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1')
    code = tmp_path / "study" / "code-abcdefabcdef" / "studies" / "long_run" / "jobs"
    code.mkdir(parents=True, exist_ok=True)
    (code / "job_gone.sh").write_text((ROOT / "studies/long_run/jobs/job_gone.sh").read_text())
    return {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}


def test_a_stage_stops_waiting_when_the_job_of_the_stage_it_follows_has_gone_with_no_mark(tmp_path: Path) -> None:
    (_marks(tmp_path) / "records.running").write_text("55\n")
    done = _stage(tmp_path, "read", "read.sbatch", "records", **_no_such_job(tmp_path))
    assert done.returncode == 1 and "job left the queue with no mark" in done.stdout and not (tmp_path / "ran").exists()


def test_a_stage_that_only_needs_another_to_have_ended_goes_on_when_its_job_has_gone(tmp_path: Path) -> None:
    (_marks(tmp_path) / "judge.running").write_text("55\n")
    assert _stage(tmp_path, "judge-llama", "judge.sbatch", "judge?", **_no_such_job(tmp_path)).returncode == 0


def _job_still_there(tmp_path: Path) -> dict:
    _stub(tmp_path / "bin" / "squeue", "echo RUNNING")
    return {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}


def test_a_stage_stops_waiting_for_one_that_fell_silent_on_a_job_still_running(tmp_path: Path) -> None:
    # A stage killed outright leaves its job up and its mark unwritten, as when the job's own loop, not the stage, said how it went.
    marks = _marks(tmp_path)
    (marks / "records.running").write_text("55\n")
    (marks / "records.beat").write_text(f"{int(time.time()) - 3600}\n")
    done = _stage(tmp_path, "read", "read.sbatch", "records", **_job_still_there(tmp_path))
    assert done.returncode == 1 and "records has said nothing for" in done.stdout and not (tmp_path / "ran").exists()
    assert (marks / "read.failed").exists()
    (marks / "judge.running").write_text("55\n")
    (marks / "judge.beat").write_text(f"{int(time.time()) - 3600}\n")
    assert _stage(tmp_path, "judge-llama", "judge.sbatch", "judge?", **_job_still_there(tmp_path)).returncode == 0, "one that needs it only to have ended goes on"


def test_a_stage_that_ended_well_between_the_looks_is_not_taken_for_lost(tmp_path: Path) -> None:
    marks = _marks(tmp_path)
    (marks / "records.running").write_text("55\n")
    # The stage marks itself and its job leaves the queue after the follower looked for a mark and before it asked after the job.
    _stub(tmp_path / "bin" / "squeue", f'touch {marks / "records.ok"}; echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1')
    done = _stage(tmp_path, "read", "read.sbatch", "records", PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    assert done.returncode == 0 and (tmp_path / "ran").exists(), done.stdout


def test_a_stage_heard_from_lately_or_on_code_from_before_the_beat_is_waited_for(tmp_path: Path) -> None:
    marks = _marks(tmp_path)
    (marks / "records.running").write_text("55\n")
    (marks / "records.beat").write_text(f"{int(time.time())}\n")
    fresh = _stage(tmp_path, "read", "read.sbatch", "records", STAGE_WAIT_SECONDS="1", **_job_still_there(tmp_path))
    assert fresh.returncode == 1 and "seconds waiting for records" in fresh.stdout
    (marks / "records.beat").unlink()
    older = _stage(tmp_path, "read2", "read.sbatch", "records", STAGE_WAIT_SECONDS="1", **_job_still_there(tmp_path))
    assert older.returncode == 1 and "seconds waiting for records" in older.stdout, "no beat at all is not silence"


def test_a_silence_is_taken_for_an_end_only_when_it_lasts(tmp_path: Path) -> None:
    # A file system that stalls every node a while must not end what follows: the silence has to outlast a second look.
    marks = _marks(tmp_path)
    (marks / "records.running").write_text("55\n")
    (marks / "records.beat").write_text(f"{int(time.time()) - 3600}\n")
    done = _stage(tmp_path, "read", "read.sbatch", "records", STAGE_CONFIRM_SECONDS="30", STAGE_WAIT_SECONDS="1", **_job_still_there(tmp_path))
    assert done.returncode == 1 and "seconds waiting for records" in done.stdout


def test_a_stage_says_it_is_alive_while_it_runs_and_clears_up_when_it_ends(tmp_path: Path) -> None:
    marks = tmp_path / "study" / "bench" / "runs" / "dev-r" / "stages"
    seen = tmp_path / "seen"
    done = _stage(tmp_path, "read", "read.sbatch", body=f"echo 0 > {marks / 'read.beat'}; sleep 1; cat {marks / 'read.beat'} > {seen}")
    assert done.returncode == 0 and int(seen.read_text()) > time.time() - 60
    assert not (marks / "read.beat").exists() and not (marks / "read.running").exists()


def test_a_stage_killed_outright_falls_silent(tmp_path: Path) -> None:
    env = _stage_env(tmp_path, "read.sbatch", "sleep 30")
    marks = tmp_path / "study" / "bench" / "runs" / "dev-r" / "stages"
    proc = subprocess.Popen(["bash", str(ROOT / "studies/memory_bench/jobs/stage.sh"), "read", "none", "read.sbatch", ""], env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if (marks / "read.running").exists():
                break
            time.sleep(0.05)
        proc.kill()  # the stage's own shell only, as the kernel ends one process
        proc.wait(timeout=10)
        time.sleep(0.5)
        (marks / "read.beat").write_text("0\n")
        time.sleep(1)
        assert (marks / "read.beat").read_text().strip() == "0", "nothing it left behind speaks for it"
        assert (marks / "read.running").exists() and not (marks / "read.failed").exists()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # the group this test made: the job script's sleep


def test_a_stage_stops_waiting_at_its_limit(tmp_path: Path) -> None:
    done = _stage(tmp_path, "read", "read.sbatch", "records", STAGE_WAIT_SECONDS="1")
    assert done.returncode == 1 and "seconds waiting for records" in done.stdout and (_marks(tmp_path) / "read.failed").exists()


def test_a_controller_that_cannot_answer_is_not_a_job_gone(tmp_path: Path) -> None:
    gone = ["bash", str(ROOT / "studies/long_run/jobs/job_gone.sh"), "55"]

    def asked(body: str) -> int:
        _stub(tmp_path / "bin" / "squeue", body)
        return subprocess.run(gone, env={**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}, capture_output=True, timeout=20).returncode

    assert asked("exit 0") == 0, "no line for it: gone"
    assert asked('echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1') == 0
    assert asked("echo RUNNING") == 1
    assert asked('echo "slurm_load_jobs error: Unable to contact slurm controller" >&2; exit 1') == 1


def test_a_task_for_a_job_that_has_left_the_queue_is_refused(tmp_path: Path) -> None:
    _stub(tmp_path / "bin" / "squeue", 'echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1')
    env = {**os.environ, "STUDY_ROOT": str(tmp_path / "study"), "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    done = subprocess.run(["bash", str(ROOT / "studies/long_run/jobs/give_task.sh"), "55", "one"], input="echo first\n", env=env, capture_output=True, text=True, timeout=20)
    assert done.returncode == 1 and "left the queue" in done.stderr


def test_a_stage_after_a_failed_one_fails_without_running(tmp_path: Path) -> None:
    (_marks(tmp_path) / "records.failed").touch()
    done = _stage(tmp_path, "read", "read.sbatch", "records")
    assert done.returncode == 1 and "since records failed" in done.stdout and not (tmp_path / "ran").exists()


def test_a_stage_that_only_needs_another_to_have_ended_runs_after_a_failure(tmp_path: Path) -> None:
    (_marks(tmp_path) / "read-llama.ok").touch()
    (_marks(tmp_path) / "judge.failed").touch()
    assert _stage(tmp_path, "judge-llama", "judge.sbatch", "read-llama judge?").returncode == 0 and (tmp_path / "ran").exists()


def test_a_stage_does_not_start_before_the_parts_hour(tmp_path: Path) -> None:
    study = tmp_path / "study"
    code = study / "code-abcdefabcdef" / "studies" / "memory_bench" / "jobs"
    code.mkdir(parents=True)
    (code / "records.sbatch").write_text(f"touch {tmp_path / 'ran'}\n")
    env = {**os.environ, "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "PART": "held_out", "RUN": "r", "SLURM_JOB_ID": "77", "STAGE_POLL": "0.05", "STAGE_BEAT": "0.2",
           "OPENS_EPOCH": str(int(time.time()) + 3600)}
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(["bash", str(ROOT / "studies/memory_bench/jobs/stage.sh"), "records", "none", "records.sbatch", ""], env=env, capture_output=True, text=True, timeout=1.5, start_new_session=True)
    assert not (tmp_path / "ran").exists()


def test_a_stage_with_no_server_fails_and_says_so_and_one_with_a_server_is_given_it(tmp_path: Path) -> None:
    (tmp_path / "study" / "logs" / "gpus-77" / "tasks").mkdir(parents=True)
    done = _stage(tmp_path, "read", "read.sbatch", server="laguna", SERVER_WAIT="1")
    assert done.returncode == 1 and "no laguna server" in done.stdout and (_marks(tmp_path) / "read.failed").exists()
    url = tmp_path / "study" / "logs" / "long-serve-77-laguna" / "url"
    url.parent.mkdir(parents=True)
    url.write_text("http://127.0.0.1:9/v1\n")
    assert _stage(tmp_path, "read2", "read.sbatch", server="laguna").returncode == 0
    assert "url http://127.0.0.1:9/v1" in (tmp_path / "ran").read_text()


def _steps(tmp_path: Path, **more: str) -> tuple[subprocess.CompletedProcess, Path]:
    study, code = tmp_path / "study", tmp_path / "study" / "code-abcdefabcdef" / "studies" / "long_run" / "jobs"
    code.mkdir(parents=True, exist_ok=True)
    for name in ("give_task.sh", "job_gone.sh"):
        (code / name).write_text((ROOT / "studies/long_run/jobs" / name).read_text())
    _stub(tmp_path / "bin" / "squeue", "echo RUNNING")
    # GNU's date, for a laptop's.
    _stub(tmp_path / "bin" / "date", 'case "$1" in -d) echo $(( $(/bin/date +%s) + ${HOUR_IN:-3600} ));; *) exec /bin/date "$@";; esac')
    done = subprocess.run(["bash", str(ROOT / "studies/memory_bench/submit_part.sh"), "steps"], capture_output=True, text=True, timeout=60, start_new_session=True,
                          env={**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "STUDY_ROOT": str(study), "SHA": "abcdefabcdef0000", "DIGEST": "d", "PART": "dev", "RUN": "r",
                               "OSS_JOBS": "101 102", "LLAMA_JOBS": "103 104", **more})
    return done, study


def test_the_stages_as_tasks_keep_opens_order_on_the_jobs_they_are_given(tmp_path: Path) -> None:
    done, study = _steps(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr

    def task(job: int, name: str) -> str:
        return (study / "logs" / f"gpus-{job}" / "tasks" / f"dev-{name}.sh").read_text()

    assert "stage.sh records none records.sbatch \'\'" in task(101, "records")
    wanted = {("read-oss-lme", 101): "lme/.*(-8000 |-118000 )", ("read-oss-locomo", 101): "locomo/.*(-8000 |-118000 )",
              ("read-oss-lme-sweep", 102): "lme/.*-(2000|4000|16000) ", ("read-oss-locomo-sweep", 102): "locomo/.*-(2000|4000|16000) "}
    for (name, job), only in wanted.items():
        text = task(job, name)
        assert f"ONLY='{only}'" in text and "READER=oss" in text and f"stage.sh {name} gpt-oss read.sbatch records" in text
    first = "/(taste_git|taste_git_headers|flat_bm25|mem0|mempalace_product)-"
    assert f"ONLY='lme{first}'" in task(103, "read-llama-lme-first") and "READER=llama" in task(103, "read-llama-lme-first")
    assert "ONLY='lme/'" in task(103, "read-llama-lme") and "records\\ read-llama-lme-first\\?" in task(103, "read-llama-lme")
    assert f"ONLY='locomo{first}'" in task(104, "read-llama-locomo-first") and "read.sbatch records\n" in task(104, "read-llama-locomo-first"), "a second server does not wait for the first"
    assert "ONLY='locomo/'" in task(104, "read-llama-locomo") and "records\\ read-llama-locomo-first\\?" in task(104, "read-llama-locomo")
    share = "/(flat_dense|mempalace_locomo_raw_session)-"
    assert f"NOT='locomo{share}'" in task(104, "read-llama-locomo"), "the second server leaves the first server's share alone"
    assert f"ONLY='locomo{share}'" in task(103, "read-llama-locomo-share") and "records\\ read-llama-lme\\?" in task(103, "read-llama-locomo-share"), \
        "the first server takes a share of LoCoMo once LongMemEval is read"
    assert "read-oss-lme\\ read-oss-locomo\\ read-oss-lme-sweep\\ read-oss-locomo-sweep" in task(101, "judge")
    assert "read-llama-lme\\ judge\\?" in task(101, "judge-llama"), "only LongMemEval's answers are judged"
    for job, tasks in ((101, 5), (102, 2), (103, 3), (104, 2)):
        assert len(list((study / "logs" / f"gpus-{job}" / "tasks").glob("*.sh"))) == tasks


def _llama_rows(view: str) -> set[str]:
    """Llama's rows on a benchmark, from the protocol itself: LoCoMo's MemPalace rows are not LongMemEval's, and it has no oracle."""
    return {row for row in required_rows(view) if "llama" in readers_of(row)}


def _rows_of(task_text: str, view: str = "locomo") -> set[str]:
    """The rows a read stage asks, by its ONLY and NOT read the way read.sbatch reads them: grep -E, then grep -vE, over 'view/row cap effort' lines."""
    only = re.search(r"ONLY='([^']*)'", task_text).group(1)
    found = re.search(r"NOT='([^']*)'", task_text)
    lines = "".join(f"{view}/{row} 512 \n" for row in sorted(_llama_rows(view)))
    picked = subprocess.run(["grep", "-E", only], input=lines, capture_output=True, text=True).stdout
    if found:
        picked = subprocess.run(["grep", "-vE", found.group(1)], input=picked, capture_output=True, text=True).stdout
    return {line.split()[0].split("/", 1)[1] for line in picked.splitlines() if line.strip()}


def test_on_two_llama_servers_every_locomo_row_is_read_by_one_stage_alone(tmp_path: Path) -> None:
    # Two stages that both took a row would write one answers file at once; a row that neither took would leave the comparison short.
    done, study = _steps(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    tasks = study / "logs"
    share = _rows_of((tasks / "gpus-103" / "tasks" / "dev-read-llama-locomo-share.sh").read_text())
    rest = _rows_of((tasks / "gpus-104" / "tasks" / "dev-read-llama-locomo.sh").read_text())
    first = _rows_of((tasks / "gpus-104" / "tasks" / "dev-read-llama-locomo-first.sh").read_text())
    assert share and not share & rest and share | rest == _llama_rows("locomo"), (share, rest)
    assert first <= rest, "the comparison's rows go first on the server that reads the rest"


def test_the_first_servers_share_by_default_is_locomo_rows_that_even_the_two_servers(tmp_path: Path) -> None:
    # A share named by LongMemEval's rows takes next to nothing on LoCoMo (its full context row read in 137 seconds), and the second server reads about three hours longer.
    done, study = _steps(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    task = (study / "logs" / "gpus-103" / "tasks" / "dev-read-llama-locomo-share.sh").read_text()
    names = re.search(r"ONLY='locomo/\(([^)]*)\)-'", task).group(1).split("|")
    assert all(any(row.startswith(f"{name}-") for row in _llama_rows("locomo")) for name in names), names
    assert _rows_of(task) == {"flat_dense-8000", "mempalace_locomo_raw_session-8000"}


def test_the_first_servers_share_can_be_named_at_hand_over(tmp_path: Path) -> None:
    done, study = _steps(tmp_path, LLAMA_SHARE="/(full_llama|mempalace_locomo_raw_dialog)-")
    assert done.returncode == 0, done.stdout + done.stderr
    share = _rows_of((study / "logs" / "gpus-103" / "tasks" / "dev-read-llama-locomo-share.sh").read_text())
    rest = _rows_of((study / "logs" / "gpus-104" / "tasks" / "dev-read-llama-locomo.sh").read_text())
    assert share == {"full_llama-31000", "mempalace_locomo_raw_dialog-8000"} and not share & rest and share | rest == _llama_rows("locomo")


def test_a_share_that_would_take_a_row_the_comparison_turns_on_is_refused(tmp_path: Path) -> None:
    # The comparison's rows go first on the second server; a share holding one would read it twice over, on two servers at once.
    done, study = _steps(tmp_path, LLAMA_SHARE="/(full_llama|mem0)-")
    assert done.returncode == 1 and "share" in done.stdout and not (study / "logs").exists()


def test_one_llama_server_takes_both_benchmarks_in_turn(tmp_path: Path) -> None:
    done, study = _steps(tmp_path, LLAMA_JOBS="103")
    assert done.returncode == 0, done.stdout + done.stderr
    tasks = study / "logs" / "gpus-103" / "tasks"
    assert len(list(tasks.glob("*.sh"))) == 4
    assert "records\\ read-llama-lme\\?" in (tasks / "dev-read-llama-locomo-first.sh").read_text(), "LoCoMo follows LongMemEval"


def test_the_stages_on_the_held_out_part_carry_its_hour_and_are_not_handed_over_too_early(tmp_path: Path) -> None:
    done, study = _steps(tmp_path, PART="held_out")
    assert done.returncode == 0, done.stdout + done.stderr
    task = (study / "logs" / "gpus-101" / "tasks" / "held_out-records.sh").read_text()
    assert re.search(r"OPENS_EPOCH=\d{10}", task)
    early, study2 = _steps(tmp_path / "early", PART="held_out", HOUR_IN="30000")
    assert early.returncode == 1 and "more than 6 hours before" in early.stdout and not (study2 / "logs").exists()


def test_a_stage_still_running_is_not_asked_again(tmp_path: Path) -> None:
    _, study = _steps(tmp_path)
    marks = study / "bench" / "runs" / "dev-r" / "stages"
    (marks / "read-oss-lme.failed").touch()
    (marks / "read-llama-lme.running").write_text("103\n")
    (marks / "read-llama-locomo.running").write_text("104\n")
    (marks / "read-llama-locomo.beat").write_text(f"{int(time.time())}\n")
    # Silent on a job still up: its shell may yet be alive behind a stalled file system, and a second copy would write the same rows, so it is left for a person to stop.
    (marks / "judge.running").write_text("101\n")
    (marks / "judge.beat").write_text(f"{int(time.time()) - 3600}\n")
    again, _ = _steps(tmp_path, AGAIN="-2")
    assert again.returncode == 0 and "read-llama-lme is still running on job 103" in again.stdout
    assert "read-llama-locomo is still running on job 104" in again.stdout
    assert "judge has said nothing for" in again.stdout and "not asked again" in again.stdout.split("judge has said nothing for")[1].splitlines()[0]
    for job, name in ((103, "read-llama-lme"), (104, "read-llama-locomo"), (101, "judge")):
        assert not (study / "logs" / f"gpus-{job}" / "tasks" / f"dev-{name}-2.sh").exists()
    assert (study / "logs" / "gpus-101" / "tasks" / "dev-read-oss-lme-2.sh").exists()


def test_a_stage_asked_again_after_its_job_left_sets_aside_what_that_job_left(tmp_path: Path) -> None:
    # Else a stage handed over again but not yet started looks, to what follows it, like the one whose job has gone.
    _, study = _steps(tmp_path)
    marks = study / "bench" / "runs" / "dev-r" / "stages"
    (marks / "records.running").write_text("55\n")
    (marks / "records.beat").write_text("1\n")
    _stub(tmp_path / "bin2" / "squeue", 'case "$*" in *55*) echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1;; *) echo RUNNING;; esac')
    again, _ = _steps(tmp_path, AGAIN="-2", PATH=f"{tmp_path / 'bin2'}:{tmp_path / 'bin'}:{os.environ['PATH']}")
    assert again.returncode == 0 and (study / "logs" / "gpus-101" / "tasks" / "dev-records-2.sh").exists()
    assert not (marks / "records.running").exists() and not (marks / "records.beat").exists()
    assert (marks / "records.running.before-2").read_text() == "55\n"


def test_failed_stages_are_not_moved_aside_unless_new_tasks_are_named(tmp_path: Path) -> None:
    _, study = _steps(tmp_path)
    marks = study / "bench" / "runs" / "dev-r" / "stages"
    (marks / "records.failed").touch()
    again, _ = _steps(tmp_path)
    assert again.returncode == 1 and "AGAIN must name" in again.stdout and (marks / "records.failed").exists()


def test_a_mark_from_another_commit_does_not_count_as_ended_well(tmp_path: Path) -> None:
    _, study = _steps(tmp_path)
    marks = study / "bench" / "runs" / "dev-r" / "stages"
    (marks / "records.ok").write_text("another commit\n")
    again, _ = _steps(tmp_path, AGAIN="-2")
    assert again.returncode == 0 and (study / "logs" / "gpus-101" / "tasks" / "dev-records-2.sh").exists()


def test_the_patterns_for_llamas_first_rows_pick_those_rows_and_no_others() -> None:
    text = (ROOT / "studies/memory_bench/submit_part.sh").read_text()
    first = re.search(r"FIRST='([^']+)'", text).group(1)
    rows = ["flat_bm25-8000", "taste_git-8000", "taste_git_headers-8000", "mem0-8000", "mempalace_product-8000", "flat_dense-8000", "full_llama-31000", "newest-8000", "none-8000",
            "oracle_llama-31000", "mempalace_bench_raw-8000", "mempalace_bench_v4-8000"]
    picked = [row for row in rows if re.search("lme" + first, f"lme/{row} 512 ")]
    assert picked == rows[:5]


def test_settings_in_a_task_reach_the_shell_as_one_word_each(tmp_path: Path) -> None:
    done, study = _steps(tmp_path)
    assert done.returncode == 0
    text = (study / "logs" / "gpus-101" / "tasks" / "dev-read-oss-lme.sh").read_text().replace("exec bash", "echo")
    shown = subprocess.run(["bash", "-c", text + '\necho "[$ONLY][$READER][$PART][$RUN][$SHA]"'], capture_output=True, text=True)
    assert "[lme/.*(-8000 |-118000 )][oss][dev][r][abcdefabcdef0000]" in shown.stdout, shown.stdout + shown.stderr
    assert "SERVER_WAIT=2400" in text


def test_run_again_the_stages_ask_only_for_what_did_not_end_well(tmp_path: Path) -> None:
    _, study = _steps(tmp_path)
    marks = study / "bench" / "runs" / "dev-r" / "stages"
    (marks / "records.ok").write_text("abcdefabcdef0000\n")
    (marks / "read-oss-lme.failed").touch()
    again, _ = _steps(tmp_path, AGAIN="-2")
    assert again.returncode == 0 and "records has ended well" in again.stdout and not (marks / "read-oss-lme.failed").exists()
    tasks = study / "logs" / "gpus-101" / "tasks"
    assert (tasks / "dev-read-oss-lme-2.sh").exists() and not (tasks / "dev-records-2.sh").exists()
