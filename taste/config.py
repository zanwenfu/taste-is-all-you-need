"""One object that says what this harness is, and one hash that names it.

Every subsystem added to the kernel brought its own switch. Scattered, those
switches are how an experiment silently drifts: two runs differ in one flag
nobody recorded, and the difference surfaces months later as unexplained
variance. :class:`HarnessConfig` gathers them, and :meth:`HarnessConfig.hash`
reduces the whole configuration to a short digest that goes in the run
manifest. Two runs with the same digest were the same harness. Two runs with
different digests are not comparable, and the digest says so before anyone
plots them together.

The defaults reproduce the original kernel exactly: every subsystem off. That
is deliberate and tested — ``HarnessConfig()`` must produce the frozen
baseline signature, or "build to delete" is a claim rather than a property.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

from taste.guardrails import GuardConfig
from taste.recovery import ActionKind, RecoveryConfig

# Roles that may appear in cost telemetry. RunStats is keyed by (role, model),
# so two call sites sharing a role string silently pool their spend into one
# cell — which is why the set is closed rather than free-form.
ROLES = frozenset({"planner", "worker", "monitor", "diagnoser", "arbiter", "curator"})

# Every event-kind prefix and the module that owns it. A new subsystem adds a
# prefix here rather than widening an existing payload: the existing kinds are
# frozen so ablation equivalence stays a mechanical diff of the event stream.
EVENT_OWNERS = {
    "run": "kernel",
    "plan": "kernel",
    "wave": "kernel",
    "worktree": "kernel",
    "step": "kernel",
    "worker": "kernel",
    "monitor": "kernel",
    "recovery": "recovery",
    "journal": "journal",
    "guard": "guardrails",
    "merge": "integrate",
    "ipc": "ipc",
    "shadow": "shadow",
}


@dataclass(frozen=True)
class HarnessConfig:
    """The complete description of one harness configuration — one arm."""

    # --- execution
    max_retries: int = 2
    max_parallel: int = 4
    planner_model: str | None = None

    # --- subsystems, all off by default
    journal: bool = False
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    guardrails: GuardConfig = field(default_factory=GuardConfig)
    two_phase_merge: bool = False
    union_gate: bool = True
    shadow: bool = False
    regression_gate: bool = False
    gate_split: str = "all"
    """``all`` or ``half`` — see RegressionGate.split."""
    """Replace the planner-written verification with the repository's own
    tests (taste/regression_gate.py). A different verifier is a different
    harness, so this is configuration and changes the hash."""
    observe_tools: bool = False
    """Observe after every tool call, not only at step boundaries.

    Off by default because it changes the observation grid, which is a
    pre-registered quantity rather than an implementation detail — and
    because the frozen ablation signature is defined against the coarse grid.

    Measured on a real 5-instance sweep, the per-attempt grid produced **10
    observations across 5 runs**: only 5 adjacent pairs in which a PASS→FAIL
    transition could be seen at all, against 70 tool calls. Worse than low
    power, that grid is *treatment-dependent* — an arm that retries emits up
    to 3 observations per step where a no-retry arm emits 1, so the recovering
    arm samples its own timeline more finely purely by recovering, biasing the
    event count toward the hypothesis.
    """
    """Observational checkpointing. Required for the four regression metrics,
    and uniform across arms by design — an arm that never checkpoints has no
    timeline of its own to measure against."""

    # --- identity
    label: str = "baseline"
    """Human name for this arm. Never load-bearing — the hash is."""

    def hash(self) -> str:
        """A short digest of everything that could change behavior.

        ``label`` is excluded on purpose: renaming an arm must not make it
        look like a different harness, and two arms that differ only in name
        must collide so the mistake is visible.
        """
        payload = asdict(self)
        payload.pop("label", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

    def to_manifest(self) -> dict:
        """The block recorded in every run's manifest."""
        return {
            "label": self.label,
            "config_hash": self.hash(),
            "max_retries": self.max_retries,
            "max_parallel": self.max_parallel,
            "journal": self.journal,
            "recovery": {
                "enabled": self.recovery.enabled,
                "policy": self.recovery.policy,
                "fixed_action": self.recovery.fixed_action.value,
                "baseline_probe": self.recovery.baseline_probe,
                "max_actions": self.recovery.max_actions,
            },
            "guardrails": {
                "enabled": self.guardrails.enabled,
                "deny_git_mutations": self.guardrails.deny_git_mutations,
                "step_budget_usd": self.guardrails.step_budget_usd,
            },
            "two_phase_merge": self.two_phase_merge,
            "union_gate": self.union_gate,
            "shadow": self.shadow,
        }

    # ------------------------------------------------------------ presets

    @classmethod
    def baseline(cls) -> HarnessConfig:
        """The original kernel: every subsystem off. The ablation floor."""
        return cls(label="baseline")

    @classmethod
    def arm(cls, name: str, **overrides) -> HarnessConfig:
        """An experimental arm, by name.

        The arms differ in exactly one thing — what the harness does when a
        step fails — which is what makes a comparison between them a
        statement about recovery policy rather than about four codebases.

        A0        no-recovery control: the Monitor's FAIL is final.
        A2        repair in place: keep the work, fix forward.
        A3        monitor-gated rollback: reset and retry.
        A3prime   attempt-matched control: same retries, same guidance, no reset.
        tiered    diagnosis-routed recovery, with the baseline probe.
        full      everything on — the complete Agent OS.
        """
        arms: dict[str, HarnessConfig] = {
            "A0": cls(label="A0-no-recovery", journal=True, shadow=True, recovery=RecoveryConfig.arm("A0")),
            "A2": cls(label="A2-repair-in-place", journal=True, shadow=True, recovery=RecoveryConfig.arm("A2")),
            "A3": cls(label="A3-rollback", journal=True, shadow=True, recovery=RecoveryConfig.arm("A3")),
            "A3reg": cls(
                label="A3reg-rollback-regression-gated", journal=True, shadow=True,
                recovery=RecoveryConfig.arm("A3"), regression_gate=True,
            ),
            "A3reg2": cls(
                label="A3reg2-rollback-gated-on-half-the-tests", journal=True, shadow=True,
                recovery=RecoveryConfig.arm("A3"), regression_gate=True, gate_split="half",
            ),
            "A3prime": cls(
                label="A3prime-no-reset", journal=True, shadow=True, recovery=RecoveryConfig.arm("A3prime")
            ),
            "tiered": cls(label="tiered", journal=True, shadow=True, recovery=RecoveryConfig.arm("tiered")),
            "full": cls(
                label="full-agent-os",
                journal=True,
                shadow=True,
                recovery=RecoveryConfig.arm("tiered"),
                guardrails=GuardConfig(enabled=True),
                two_phase_merge=True,
            ),
        }
        if name not in arms:
            raise ValueError(f"unknown arm {name!r}; known: {sorted(arms)}")
        return replace(arms[name], **overrides) if overrides else arms[name]

    @classmethod
    def arm_names(cls) -> list[str]:
        return ["A0", "A2", "A3", "A3reg", "A3reg2", "A3prime", "tiered", "full"]


def kernel_kwargs(config: HarnessConfig) -> dict:
    """Translate a config into ``Kernel(...)`` arguments."""
    return {
        "max_retries": config.max_retries,
        "max_parallel": config.max_parallel,
        "planner_model": config.planner_model,
        "journal": config.journal,
        "recovery_config": config.recovery,
        "guard_config": config.guardrails,
        "two_phase_merge": config.two_phase_merge,
        "union_gate": config.union_gate,
        "shadow": config.shadow,
        "observe_tools": config.observe_tools,
    }


__all__ = [
    "EVENT_OWNERS",
    "ROLES",
    "ActionKind",
    "GuardConfig",
    "HarnessConfig",
    "RecoveryConfig",
    "kernel_kwargs",
]
