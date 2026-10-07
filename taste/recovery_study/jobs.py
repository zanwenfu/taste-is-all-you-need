"""Every Harbor job the recovery study starts, built in one place.

A job is one call of ``infra/azure/run-harbor.sh``: one task, ``-k`` attempts
of it with one set of agent settings, each passed as ``--ak name=value``.
Branching (work package A) is reached only through settings, and their names
are the table ``SETTING``: a setting renamed there is changed nowhere else.
The command that exports a replay script from a finished trial is
``EXPORT_TEMPLATE``, for the same reason. A checker trial (work package B) is
the hosted checker run alone, with the settings in ``CHECKER``
(``checker_trial``).

What a job spec holds is enough to start it on the measurement host: the
commands that prepare it (exporting the replay scripts it reads), the
environment ``run-harbor.sh`` reads (``MODEL``, ``JOBS``) and its arguments.
``launcher`` writes them into one shell script that starts each job once and
keeps at most a given number of the study's jobs running.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

# The settings this study passes, by what they mean.
#   A, branching: the replay script exported from a finished trial; the step
#   its prefix ends at; how the prefix's files are made (rebuild: execute the
#   recorded commands; restore: restore a checkpoint); the checkpoint a rebuild
#   saves or a restore restores; whether the agent goes on live after the
#   prefix ("on") or the trial ends there ("off"); what the agent sees at the
#   prefix's last step ("append": the recorded output followed by a note;
#   "reject": the note instead of the output, with a non-zero exit) and the
#   note's file; a file whose text follows the task text; and a file whose
#   text replaces it, for a trial whose agent is not the recorded one (the
#   checker, run on the files the prefix brought back).
#   Taste's equal caps (taste/benchmarks/harbor_settings.py, harbor_agent.py):
#   the dollars a trial may spend and the agent time it is given.
SETTING = {
    "script": "branch", "step": "branch_step", "mode": "branch_mode",
    "checkpoint": "branch_checkpoint", "live": "branch_live",
    "override": "branch_override", "note": "branch_note", "suffix": "task_suffix",
    "task_text": "task_text",
    "spend_cap": "spend_cap_usd", "deadline": "agent_timeout_sec",
}
MODES = ("rebuild", "restore")
OVERRIDES = ("append", "reject")
CAPS = frozenset({SETTING["spend_cap"], SETTING["deadline"]})
# A job sets these itself (run-harbor.sh sets worker_python); they are never
# copied from the trial it branches from or repeats. A task suffix is copied:
# a branch of a trial that had one must give the agent the same task text.
OWN = frozenset({SETTING[name] for name in ("script", "step", "mode", "checkpoint", "live", "override",
                                            "note", "task_text")} | {"worker_python"})
# A's exporter: writes a finished trial's replay script (its ordered model
# calls and commands) from the trial's settled record.
EXPORT_TEMPLATE = "python3 -m taste.benchmarks.replay_export --record {record} --out {out}"
# The study's models: the agent's (GPT-6 Luna; GPT-6 Sol on a subset) and the checker's.
WORKER_MODEL = "gpt-6-luna"
CHECKER_MODEL = "gpt-6-sol"
# A checker trial (taste/agents/checker.py) runs the checker alone, with the
# trial's limits as backstops above its own (30 steps, $2, 900 s).
CHECKER = {"agent": "checker", "services": "none", "worker_max_calls": 35, "spend_cap_usd": 2.5,
           "agent_timeout_sec": 1500}
EFFORTS = ("low", "medium", "high")
RUN_HARBOR = str(Path(__file__).resolve().parents[2] / "infra" / "azure" / "run-harbor.sh")
UNIT_PREFIX = "taste-harbor-"
NAME_LIMIT = 100
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def digest(text, length=6):
    return hashlib.sha256(str(text).encode()).hexdigest()[:length]


def safe(text, limit=40):
    """Letters, digits, - and _ only, as run-harbor.sh requires of a job name."""
    return _UNSAFE.sub("_", str(text)).strip("_-")[:limit] or "x"


def run_key(run_id):
    """A short key for a run that can go in a job name: readable, and unique by its digest."""
    return f"{safe(run_id, 32)}-{digest(run_id)}"


def job_name(prefix, *parts):
    name = "-".join(safe(part, 48) for part in (prefix, *parts))
    return name if len(name) <= NAME_LIMIT else f"{name[:NAME_LIMIT - 7]}-{digest(name)}"


def model_flag(model):
    """A model as run-harbor.sh's MODEL names it: ``azure/<command-line name>``."""
    model = str(model)
    return model if "/" in model else "azure/" + model


def short_model(model):
    return str(model or "").rsplit("/", 1)[-1]


def value(raw):
    """A setting's value as the command line writes it."""
    if isinstance(raw, bool):
        return "on" if raw else "off"
    if isinstance(raw, float):
        return format(raw, ".6f").rstrip("0").rstrip(".") or "0"
    return str(raw)


def inherited(settings, *, drop=()):
    """A trial's settings, as a job that branches from it or repeats it starts from."""
    dropped = OWN | frozenset(drop)
    return {name: raw for name, raw in (settings or {}).items() if name not in dropped}


def branch(script, step, *, mode="rebuild", checkpoint=None, live=True, override=None, note=None):
    """Bring back the state at ``step`` of the trial ``script`` was exported from."""
    if mode not in MODES:
        raise ValueError("branch mode must be rebuild or restore")
    if mode == "restore" and not checkpoint:
        raise ValueError("a restore names the checkpoint it restores")
    if (override is None) != (note is None):
        raise ValueError("an override and its note come together")
    if override is not None and override not in OVERRIDES:
        raise ValueError("branch override must be append or reject")
    if int(step) != step or step < 0:
        raise ValueError("a branch step is a whole number of steps, 0 or more")
    values = {SETTING["script"]: str(script), SETTING["step"]: int(step), SETTING["mode"]: mode}
    if checkpoint:
        values[SETTING["checkpoint"]] = str(checkpoint)
    values[SETTING["live"]] = bool(live)
    if override is not None:
        values[SETTING["override"]] = override
        values[SETTING["note"]] = str(note)
    return values


def task_suffix(path):
    return {SETTING["suffix"]: str(path)}


def checker_trial(task_file, files, *, effort="medium"):
    """A checker trial: the checker alone, on ``files`` (the checked trial's final state, brought
    back with ``branch(..., live=False)``), given the text in ``task_file`` as its task."""
    if effort not in EFFORTS:
        raise ValueError("the checker's effort must be low, medium or high")
    return {**CHECKER, "worker_effort": effort, **files, SETTING["task_text"]: str(task_file)}


def caps(usd, seconds):
    """The dollars a trial may spend and the agent time it has: Taste's equal-cap settings."""
    if not usd > 0 or not seconds >= 1:
        raise ValueError("a trial needs a positive budget in dollars and in seconds")
    return {SETTING["spend_cap"]: round(float(usd), 6), SETTING["deadline"]: int(seconds)}


@dataclass(frozen=True)
class JobSpec:
    """One Harbor job: how to prepare it and the run-harbor.sh call that starts it."""

    name: str
    attempts: int
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    prepare: tuple[tuple[str, ...], ...] = ()

    def to_dict(self):
        return {"name": self.name, "attempts": self.attempts, "argv": list(self.argv),
                "env": dict(self.env), "prepare": [list(command) for command in self.prepare]}

    @classmethod
    def from_dict(cls, raw):
        return cls(raw["name"], int(raw["attempts"]), tuple(raw["argv"]),
                   tuple(sorted(raw.get("env", {}).items())), tuple(tuple(c) for c in raw.get("prepare", ())))

    @property
    def jobs_dir(self):
        return dict(self.env).get("JOBS", "")

    @property
    def concurrent(self):
        """The trials it runs at once (its ``-n``)."""
        return int(self.argv[self.argv.index("-n") + 1])

    def settings(self):
        """The ``--ak`` settings, in order, as written."""
        return {self.argv[i + 1].split("=", 1)[0]: self.argv[i + 1].split("=", 1)[1]
                for i in range(len(self.argv) - 1) if self.argv[i] == "--ak"}

    def shell(self):
        """The job as one shell command: its preparation, then the launch."""
        steps = [" ".join(shlex.quote(part) for part in command) for command in self.prepare]
        launch = " ".join(["env", *(shlex.quote(f"{key}={raw}") for key, raw in self.env),
                           *(shlex.quote(part) for part in self.argv)])
        return " && ".join([*steps, launch])


# A trial Harbor ended with an error is run again, up to this many times:
# Docker failed, or Taste never started the hosted agent
# (harbor_agent.HostedAgentNotStarted). A graded trial never is, nor one the
# time limit cut off.
RETRIES = 2


def harbor_job(name, *, tasks_dir, task, attempts, settings, model, jobs_dir, run_harbor=RUN_HARBOR,
               concurrent=8, env=None, prepare=(), retries=RETRIES):
    """``run-harbor.sh <name> <tasks dir> -i <task> -k <attempts> -n <n> --max-retries <r> --ak ...``."""
    if attempts < 1:
        raise ValueError("a job runs at least one attempt")
    if safe(name, len(name)) != name:
        raise ValueError(f"job name {name!r}: letters, digits, - and _ only")
    argv = [str(run_harbor), name, str(tasks_dir), "-i", str(task), "-k", str(attempts),
            "-n", str(max(1, min(int(concurrent), attempts))), "--max-retries", str(int(retries))]
    for key, raw in settings.items():
        argv += ["--ak", f"{key}={value(raw)}"]
    environment = {"MODEL": model_flag(model), "JOBS": str(jobs_dir), **(env or {})}
    return JobSpec(name, attempts, tuple(argv), tuple(sorted(environment.items())),
                   tuple(tuple(command) for command in prepare))


def export_script(record, out, template=EXPORT_TEMPLATE):
    """The command that writes a finished trial's replay script, if it is not written yet."""
    command = template.format(record=shlex.quote(str(record)), out=shlex.quote(str(out)))
    return ("sh", "-c", f"[ -e {shlex.quote(str(out))} ] || {command}")


def write_text(path, text):
    """Write a note or checker input the trial owner reads; never half-written."""
    path = Path(path)
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.chmod(0o644)
    os.replace(temporary, path)
    return path


def launcher(specs, *, prefix, max_trials, title=""):
    """A shell script that starts each job once, keeping the study's running trials at most ``max_trials``.

    A running job counts as the trials it runs at once (its ``-n``, read from
    its unit's command line, so jobs another round started count too); a job
    starts when its own fit beside them, or when nothing runs. It skips a job
    whose directory or systemd unit exists (run-harbor.sh never overwrites a
    job), runs each job's preparation before starting it, and at the end waits
    until none of the study's jobs is running, so that the driver can be run
    again on finished jobs (with WAIT=0 it returns once its jobs have started).
    """
    pattern = shlex.quote(f"{UNIT_PREFIX}{prefix}-*")
    lines = ["#!/bin/sh", f"# {title}".rstrip(),
             "# Run as root on the measurement host. A job starts while the study's running",
             "# trials, each running job's -n, stay at most MAX_TRIALS.", "set -u",
             f"MAX_TRIALS=${{MAX_TRIALS:-{int(max_trials)}}}",
             "units() {",
             f"  systemctl list-units --plain --no-legend --state=active,activating {pattern} 2>/dev/null"
             " | awk '{print $1}'",
             "}",
             "running() {",
             "  units | wc -l",
             "}",
             "trials() {",
             "  total=0",
             "  for unit in $(units); do",
             "    n=$(systemctl show -p ExecStart --value \"$unit\" | grep -o -- ' -n [0-9]*' | head -1 | tr -dc 0-9)",
             "    total=$((total + ${n:-1}))",
             "  done",
             '  echo "$total"',
             "}",
             "start() {",
             "  job=$1; dir=$2; n=$3; command=$4",
             f'  if [ -e "$dir/$job" ] || systemctl is-active --quiet "{UNIT_PREFIX}$job"; then',
             '    echo "already started: $job"; return 0',
             "  fi",
             '  while now=$(trials) && [ "$now" -gt 0 ] && [ $((now + n)) -gt "$MAX_TRIALS" ]; do sleep 20; done',
             '  sh -c "$command" || echo "could not start: $job" >&2',
             "}"]
    for spec in specs:
        lines.append(f"start {shlex.quote(spec.name)} {shlex.quote(spec.jobs_dir)} {spec.concurrent} "
                     f"{shlex.quote(spec.shell())}")
    # WAIT=0: return once every job has started, for a loop that runs the
    # driver again meanwhile, so one run's next probes need not wait for all.
    lines += ['[ "${WAIT:-1}" = 0 ] && exit 0',
              'while [ "$(running)" -gt 0 ]; do sleep 30; done',
              'echo "none of this study\'s jobs is running: run the driver again"', ""]
    return "\n".join(lines)
