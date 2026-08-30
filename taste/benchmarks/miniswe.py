"""mini-swe-agent under the instrument: a public scaffold, unchanged, observed.

The paper's Limitations names this as the next experiment: every number so
far comes from our own short-horizon harness. mini-swe-agent (Lieret and
Jimenez, 2025) is a ~100-line bash-only agent with a public leaderboard, and
its loop has exactly one seam that touches the working tree:
``DefaultAgent.execute_actions`` -> ``env.execute(action)``.

We replace nothing else. The scaffold keeps its own prompts, its own model
layer (litellm), its own step and cost limits, its own submission protocol.
Only the environment is ours: ``TasteEnvironment.execute`` runs the command
through the cell's ``SandboxRouter`` -- inside the benchmark's pinned
container, with host/container coherence -- and records an observation
(a commit on the hidden ref) whenever the command changed the tree. The
timeline that comes out is the same object the harness arms produce, so
``make_score`` replays and grades it with the identical instrument.

What this arm cannot report: harness-check attribution (there is no
monitor), so every event is silent by construction; and a dollar cap of
ours (the scaffold's own ``cost_limit`` applies).
"""

from __future__ import annotations

import contextlib
import json
import platform
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taste.cores import Plan, Step, Verification
from taste.kernel import RunResult
from taste.memory import Memory
from taste.shadow import ShadowLog

SCAFFOLD = "mini-swe-agent"

# The scaffold's default environment variables (config/benchmarks/swebench.yaml).
DEFAULT_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


class TasteEnvironment:
    """mini-swe-agent's ``Environment`` protocol, backed by our router.

    ``execute`` returns the dict shape the scaffold expects
    (``output``, ``returncode``, ``exception_info``) with stderr folded into
    ``output`` as the scaffold's DockerEnvironment does (``stderr=STDOUT``).
    """

    def __init__(
        self,
        router: Any,
        shadow: ShadowLog | None,
        *,
        cwd: str = "/testbed",
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> None:
        self.router = router
        self.shadow = shadow
        self.cwd = cwd
        self.timeout = timeout
        self.env = dict(DEFAULT_ENV if env is None else env)
        self.n_commands = 0
        self.observed = 0
        self.consecutive_errors = 0
        #: Set once the scaffold's agent exists: observations stamp its
        #: running cost, so per-episode dollar metrics mean the same thing
        #: they mean for the harness arms (whose kernel reads RunStats).
        self.agent: Any = None
        self.cost_box = {"usd": 0.0}

    #: Consecutive transport failures after which the environment is
    #: declared dead. The scaffold's own environment returns returncode -1
    #: for any exception and lets the model carry on; ours does the same for
    #: a one-off, but a container that is gone would otherwise be probed
    #: until the $3 cap, billed as agent behaviour.
    DEAD_AFTER = 3

    def cost_pair(self) -> tuple[float, float]:
        return (self.cost_box["usd"], self.cost_box["usd"])

    # -- protocol ---------------------------------------------------------
    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        self.n_commands += 1
        if self.agent is not None:
            self.cost_box["usd"] = float(getattr(self.agent, "cost", 0.0))
        exports = " ".join(f"{k}={_sh_quote(v)}" for k, v in self.env.items())
        prefix = f"export {exports}; " if exports else ""
        cd = f"cd {_sh_quote(cwd)} && " if cwd and cwd != self.cwd else ""
        # The scaffold's environment runs `docker exec ... bash -c cmd` with
        # stderr=STDOUT: one stream, in arrival order. Our sandbox demuxes,
        # which would hand the model all of stderr after all of stdout (a
        # traceback below the output that preceded it). Merge at the shell,
        # inside a group so heredocs and comments survive (defect 38).
        wrapped = f"{prefix}{cd}{{\n{command}\n}} 2>&1"
        try:
            result = self.router.exec(wrapped, timeout=timeout or self.timeout)
        except Exception as exc:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.DEAD_AFTER:
                raise RuntimeError(f"environment dead after {self.consecutive_errors} transport failures: {exc}") from exc
            return {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }
        self.consecutive_errors = 0
        # Observe after every command; the shadow log dedupes an unchanged tree,
        # so only mutating commands produce observations -- the same grid the
        # harness arms use under observe_tools.
        if self.shadow is not None:
            commit = self.shadow.observe(
                step_id=f"cmd-{self.n_commands:03d}", attempt=1, trigger="tool", tool="bash"
            )
            if commit is not None:
                self.observed += 1
        output = result.stdout + (("\n" + result.stderr) if result.stderr else "")
        if getattr(result, "timed_out", False):
            # The scaffold's shape for a timeout: returncode -1 and the
            # exception text, with the partial output kept.
            out = {
                "output": output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: Command timed out after {timeout or self.timeout} seconds",
                "extra": {"exception_type": "TimeoutExpired"},
            }
        else:
            out = {"output": output, "returncode": result.exit_code, "exception_info": ""}
        self._check_finished(out)
        return out

    @staticmethod
    def _check_finished(output: dict) -> None:
        """The scaffold's submission protocol lives in its environment
        (``DockerEnvironment._check_finished``): a command whose first output
        line is the sentinel, exiting 0, ends the run with the rest of the
        output as the submission. Reproduced verbatim so the run terminates
        the way it does under the scaffold's own environment."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            from minisweagent.exceptions import Submitted

            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission, "extra": {"exit_status": "Submitted", "submission": submission}})

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        base = {"cwd": self.cwd, "timeout": self.timeout, "env": self.env, "environment_class": "taste"}
        base.update(platform.uname()._asdict())
        base.update(kwargs)
        return base

    def serialize(self) -> dict[str, Any]:
        return {"info": {"config": {"environment": self.get_template_vars(), "environment_type": f"{__name__}.TasteEnvironment"}}}


def _sh_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


@dataclass
class ScaffoldStats:
    """The three numbers the sweep ledger reads from a run's stats."""

    total_cost_usd: float = 0.0
    total_work_usd: float = 0.0
    cache_delta_usd: float = 0.0


def load_scaffold_config(
    model_name: str | None = None, *, cost_limit: float | None = None, model_class: str | None = None
) -> dict:
    """The scaffold's own SWE-bench config, with only the model name swapped.

    ``model_class`` selects one of the scaffold's own model layers (e.g.
    ``litellm_response`` for the Responses API, which its leaderboard uses
    for GPT-5-class models; under chat completions gpt-5.6-sol answered
    three times without a tool call and the scaffold exited with
    RepeatedFormatError).
    """
    import yaml
    from minisweagent.config import get_config_path

    config = yaml.safe_load(Path(get_config_path("swebench.yaml")).read_text(encoding="utf-8"))
    if model_name:
        config.setdefault("model", {})["model_name"] = model_name
    if model_class:
        config.setdefault("model", {})["model_class"] = model_class
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    return config


def scaffold_pricing(model_name: str) -> dict | None:
    """litellm's price-map entry for a model our own table prices.

    The scaffold enforces its ``cost_limit`` through ``litellm.completion_cost``,
    which knows nothing about the snapshots the paper's arms ran on
    (``claude-sonnet-4-6``, ``gpt-5.6-sol``). Registering our verified rates
    keeps the scaffold's own cap meaningful for the same models, instead of
    the ``ignore_errors`` mode in which every call costs $0 and only the
    step limit ends a run. Returns ``None`` when our table has no entry,
    leaving litellm's map in charge.
    """
    from taste.pricing import PRICES

    bare = model_name.split("/", 1)[1] if "/" in model_name else model_name
    price = PRICES.get(bare)
    if price is None:
        return None
    rates = price.rates_for(0)
    entry = {
        "input_cost_per_token": rates.input / 1e6,
        "output_cost_per_token": rates.output / 1e6,
        "cache_read_input_token_cost": rates.cache_read / 1e6,
        "cache_creation_input_token_cost": rates.cache_write / 1e6,
        "litellm_provider": price.provider,
        "mode": "chat",
    }
    # register_model merges into litellm's existing entry, so a stale
    # long-context tier of theirs would survive unless ours is written over
    # it. litellm names the tier by its threshold in thousands of tokens.
    for _limit, long_rates in price.tiers[1:]:
        threshold = price.tiers[0][0]
        if threshold is None:
            break
        suffix = f"_above_{threshold // 1000}k_tokens"
        entry[f"input_cost_per_token{suffix}"] = long_rates.input / 1e6
        entry[f"output_cost_per_token{suffix}"] = long_rates.output / 1e6
        entry[f"cache_read_input_token_cost{suffix}"] = long_rates.cache_read / 1e6
        entry[f"cache_creation_input_token_cost{suffix}"] = long_rates.cache_write / 1e6
    return entry


def register_scaffold_pricing(model_name: str) -> bool:
    entry = scaffold_pricing(model_name)
    if entry is None:
        return False
    import litellm

    names = {model_name}
    if "/" in model_name:
        names.add(model_name.split("/", 1)[1])
    litellm.register_model({name: entry for name in names})
    return True


def build_run_result(
    *,
    task: str,
    session_id: str,
    branch: str,
    final_sha: str,
    started: float,
    exit_status: str,
    cost: float,
    exception: str | None = None,
) -> RunResult:
    """A RunResult the sweep ledger and scorer accept for an external agent.

    One pseudo-step stands in for the scaffold's whole run: ``outcomes`` is
    empty (no monitor verdicts exist), so ``steps_passed`` reads 0 for every
    cell and must not be interpreted; ``status`` follows the scaffold's exit
    status (``Submitted`` = completed).
    """
    failure_kind = None
    if exception:
        failure_kind = "infra"
    elif exit_status in ("LimitsExceeded", "TimeExceeded"):
        failure_kind = "budget"
    elif exit_status != "Submitted":
        failure_kind = "task"
    return RunResult(
        task=task,
        session_id=session_id,
        branch=branch,
        status="completed" if exit_status == "Submitted" else "failed",
        plan=Plan(task=task, steps=[Step(id="agent", description=f"{SCAFFOLD} run", verification=Verification(kind="shell", command="true"))]),
        outcomes=[],
        final_sha=final_sha,
        elapsed_seconds=round(time.time() - started, 2),
        failure_reason=exception or (None if exit_status == "Submitted" else exit_status),
        stats=ScaffoldStats(total_cost_usd=cost, total_work_usd=cost),  # type: ignore[arg-type]
        failure_kind=failure_kind,
    )


def make_miniswe_execute(
    *, model_name: str, cost_limit: float | None = None, model_class: str | None = None, model_factory=None
):
    """Build the ``execute`` callable for ``run_sweep``: one mini-swe-agent run per cell.

    ``model_factory`` (a zero-arg callable returning a scaffold ``Model``)
    exists for the scripted smoke test; production uses the scaffold's own
    ``get_model`` with its litellm layer.
    """

    def execute(cell, ctx) -> RunResult:


        if ctx.agent_sandbox is None:
            raise RuntimeError("mini-swe-agent must run routed (a pinned container); refusing the host path")
        config = load_scaffold_config(model_name, cost_limit=cost_limit, model_class=model_class)
        started = time.time()
        session_id = uuid.uuid4().hex[:8]
        ctx.session = session_id
        try:
            return _run_cell(ctx, config, session_id, started, model_name, model_factory)
        finally:
            # Whatever happened above -- a model layer that failed to build,
            # a dead container, a clean run -- the cell's container goes.
            if ctx.agent_sandbox is not None:
                with contextlib.suppress(Exception):
                    ctx.agent_sandbox.close()
                ctx.agent_sandbox = None

    return execute


def _run_cell(ctx, config: dict, session_id: str, started: float, model_name: str, model_factory) -> RunResult:
    from minisweagent.agents.default import DefaultAgent

    from taste.routing import SandboxRouter

    memory = Memory.open_session(ctx.workspace, session_id, base_ref="HEAD")
    try:
        # Transparent sync: the scaffold verifies its work with git diff /
        # git status inside the container, so the sync baseline must not
        # advance underneath it (see SandboxRouter.advance_baseline).
        router = SandboxRouter(
            ctx.agent_sandbox, ctx.workspace, workdir=ctx.agent_sandbox.workdir, advance_baseline=False
        )
        ctx.router = router
        Path(ctx.gitdir).mkdir(parents=True, exist_ok=True)
        env_cfg = config.get("environment", {})
        env = TasteEnvironment(
            router, None,
            cwd=env_cfg.get("cwd", ctx.agent_sandbox.workdir),
            timeout=int(env_cfg.get("timeout", 60)),
            env=env_cfg.get("env"),
        )
        shadow = ShadowLog(memory, gitdir=Path(ctx.gitdir), session=session_id, cost_pair_reader=env.cost_pair)
        env.shadow = shadow
        ctx.shadow_ref = shadow.ref if hasattr(shadow, "ref") else f"TASTE_SHADOW_HEAD_{session_id.upper()}"
        shadow.observe(step_id="run", attempt=0, trigger="run")

        if model_factory is not None:
            model = model_factory()
        else:
            from minisweagent.models import get_model

            register_scaffold_pricing(model_name)
            model = get_model(config=config.get("model", {}))
        agent_cfg = dict(config.get("agent", {}))
        agent_cfg["output_path"] = Path(ctx.gitdir) / "miniswe.traj.json"
        agent = DefaultAgent(model, env, **agent_cfg)
        env.agent = agent

        exit_status, exception, submission = "", None, ""
        try:
            info = agent.run(task=ctx.instance.problem_statement)
            exit_status = str(info.get("exit_status", ""))
            submission = str(info.get("submission", ""))
        except Exception as exc:
            exception = f"{type(exc).__name__}: {exc}"
            # The scaffold appends an exit message naming the exception
            # type before re-raising; keep its name as the exit status.
            with contextlib.suppress(Exception):
                exit_status = str(agent.messages[-1].get("extra", {}).get("exit_status", "")) or type(exc).__name__
        finally:
            env.cost_box["usd"] = float(agent.cost)
            with contextlib.suppress(Exception):
                shadow.observe(step_id="final", attempt=0, trigger="final", dedupe=False)
            # Grading diffs the host tree against the root commit, and a new
            # file is invisible to that diff until it is staged. The harness
            # arms stage through their checkpoints; this arm has none, so it
            # stages here -- everything except the scaffold's own patch.txt.
            with contextlib.suppress(Exception):
                memory.repo.git.add("--all", "--", ".", ":(exclude)patch.txt")
            manifest = {
                "session": session_id,
                "scaffold": SCAFFOLD,
                "scaffold_version": _scaffold_version(),
                "model": config.get("model", {}).get("model_name"),
                "agent_config": {k: v for k, v in agent_cfg.items() if k != "output_path"},
                "environment": env.get_template_vars(),
                "commands": env.n_commands,
                "observations": env.observed,
                "sync_skipped_paths": list(router.skipped),
                "network_mode": getattr(ctx.agent_sandbox, "network_mode", None) or "none",
                "streams": "merged (2>&1 at the shell)",
                "pricing_table_sha": _pricing_table_sha(),
                "exit_status": exit_status,
                "submission_chars": len(submission),
                "cost_usd": agent.cost,
                "api_calls": agent.n_calls,
                "created_at": started,
            }
            (Path(ctx.gitdir) / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
            ctx.llm_stats = ScaffoldStats(total_cost_usd=agent.cost, total_work_usd=agent.cost)
        final_sha = memory.head().sha
        return build_run_result(
            task=ctx.instance.problem_statement, session_id=session_id, branch=memory.branch, final_sha=final_sha,
            started=started, exit_status=exit_status, cost=agent.cost, exception=exception,
        )
    finally:
        memory.close()


def _pricing_table_sha() -> str:
    try:
        from taste.pricing import table_sha

        return str(table_sha())
    except Exception:
        return "unknown"


def _scaffold_version() -> str:
    try:
        from minisweagent import __version__

        return str(__version__)
    except Exception:
        return "unknown"


class ScriptedModel:
    """A scaffold ``Model`` that replays fixed bash commands: the smoke test's
    stand-in for litellm, so the whole path (prepare, route, observe, score,
    grade) runs with no API spend."""

    def __init__(self, commands: list[str]) -> None:
        self.commands = list(commands)
        self.config = type("Cfg", (), {"model_name": "scripted"})()

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {"model_name": "scripted", **kwargs}

    def format_message(self, **kwargs: Any) -> dict:
        return dict(kwargs)

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        if not self.commands:
            # The real model layer ends a run by raising Submitted with the
            # exit message; a returned "exit" message would be followed by an
            # observation and never terminate the scaffold's loop.
            from minisweagent.exceptions import Submitted

            raise Submitted(
                {"role": "exit", "content": "Submitted", "extra": {"exit_status": "Submitted", "submission": "", "cost": 0.0}}
            )
        command = self.commands.pop(0)
        return {"role": "assistant", "content": f"run: {command}", "extra": {"actions": [{"command": command}], "cost": 0.0}}

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict) -> list[dict]:
        return [{"role": "user", "content": "\n".join(f"<returncode>{o['returncode']}</returncode>\n{o['output']}" for o in outputs)}]

    def serialize(self) -> dict:
        return {"info": {"config": {"model": {"model_name": "scripted"}, "model_type": f"{__name__}.ScriptedModel"}}}
