# AgentRewind's agent-facing text and pure helpers, copied unchanged from replay-agent-recorder
# (https://github.com/Futuresis/replay-agent-recorder), commit 6661046, replay/backtrack.py
# (sha256 31d12e7466ec96777a59c1023255a3a415f188f3b54eb608a23069c43ffa7dd9), under its licence:
#
# MIT License
#
# Copyright (c) 2026 Replay contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Each block between an "upstream lines" marker and the next is upstream's text, byte for byte;
# tests/test_agentrewind.py holds each block's sha256 as read from that commit. Nothing here is
# Taste's own: how Taste uses it is in taste/agents/agentrewind.py.

from __future__ import annotations

import json
import re
from typing import Any, Literal


# --- upstream lines 25-37 ---
BacktrackHistoryMode = Literal["deterministic", "llm-summary"]
BacktrackSelectionPolicy = Literal["agent-direct", "probabilistic-suffix"]
BacktrackMemoryPolicy = Literal["cumulative", "none"]
BacktrackContextPolicy = Literal["rollback", "preserve"]
BacktrackWorkspacePolicy = Literal["rollback", "preserve"]
DEFAULT_MEMORY_PREFIX = "Backtrack memory notebook from abandoned future branches:\n"
DEFAULT_CANDIDATE_LIMIT = 80
DEFAULT_SELECTION_POLICY: BacktrackSelectionPolicy = "agent-direct"
DEFAULT_SELECTION_TOP_K = 5
DEFAULT_SELECTION_DELTA = 0.7
DEFAULT_MEMORY_LIMIT = 20
DEFAULT_MEMORY_ITEM_CHAR_LIMIT = 900
DEFAULT_MEMORY_NOTEBOOK_CHAR_LIMIT = 8_000

# --- upstream lines 61-81 ---
ROLLBACK_CONTEXT_VISIBILITY_INSTRUCTIONS = (
    "At the beginning of a fork with backtrack memory, the abandoned branch's "
    "later messages and tool results are not visible except through that notebook. "
)
PRESERVE_CONTEXT_VISIBILITY_INSTRUCTIONS = (
    "At the beginning of a preserve-context fork, the abandoned branch's later "
    "messages and tool results remain visible by design. Their visibility is expected "
    "and does not mean that backtrack failed; this policy intentionally does not rewind "
    "agent messages. Treat filesystem claims in post-target observations as potentially "
    "stale until they are checked against the current controlled Git workspace. "
)
ROLLBACK_FUTURE_EVIDENCE_INSTRUCTIONS = (
    "Treat the notebook as the only carried evidence from the failed future branch; "
    "do not plan to inspect the old branch after the rewind. "
)
PRESERVED_STATE_EVIDENCE_INSTRUCTIONS = (
    "The Backtrack memory notebook carries explicit failure guidance, while the "
    "policy-preserved conversation or controlled Git workspace may also carry state "
    "from the failed future branch. Interpret that state using the policy-specific "
    "instructions below. "
)

# --- upstream lines 84-252 ---
BACKTRACK_TOOL_INSTRUCTIONS = (
    "Backtrack is a deliberate timeline fork escape hatch. Use it when the "
    "current execution path is shown by concrete external feedback or repeated "
    "failed repair evidence to be wrong, stuck, self-contradictory, or too "
    "expensive to repair in place. Do not use backtrack as an opening diagnostic "
    "before trying listed task tools and obtaining concrete failure evidence. "
    "Call backtrack_candidates first, then call backtrack_commit with a record_uid "
    "from those candidates when a rewind is the better next action. "
    "The candidate rendering mode and candidate count are controlled by the "
    "current run configuration; do not try to set tool parameters for them. "
    "Only use a top-level candidate record_uid returned by backtrack_candidates; "
    "those ids identify LLM decision records and are the only valid commit "
    "targets. Do not invent ids and do not use any tool/output ids from previews. "
    "After a failed verifier result, repair in place only once when the fix is "
    "obvious and local, such as one typo, one missing file, one known wrong path, "
    "one missing row, or one clearly wrong value. "
    "If the same verifier check, or the same semantic failure, appears again "
    "after a repair attempt, call backtrack_candidates before making more local "
    "edits. A same semantic failure means the same invariant, artifact, missing "
    "evidence, preservation requirement, malformed output, or error class is "
    "still failing even if the wording changed. Repeated failure means the "
    "current branch is likely structurally bad; do not keep rewriting the same "
    "artifact, guessing hidden prior state, or declaring the verifier wrong. "
    "Identify the first bad LLM decision: the planning, search-policy, "
    "data-source, or file-operation decision that introduced the strategy that "
    "now needs to be replaced. Commit to the closest returned LLM decision record "
    "before that bad decision, so the next fork can re-decide before entering the "
    "same strategy. Prefer the decision immediately before the first write, edit, "
    "move, destructive action, wrong source choice, wrong schema or manifest "
    "interpretation, wrong extraction/calculation/grouping strategy, or ignored "
    "preservation/provenance requirement that affected the repeatedly failing "
    "artifact or invariant. Before committing, check the candidate against the "
    "rewind boundary claimed in memory_summary. Inspect candidate tool_calls_preview: "
    "if memory says to rewind before a filesystem mutation, destructive/polluting "
    "file operation, or a specific move/write/edit/create boundary, do not choose "
    "a candidate at or after that relevant mutating decision; choose an earlier "
    "LLM record. If no returned candidate precedes the bad decision, choose the "
    "earliest returned candidate and state that limitation in selected_candidate_safety_check. "
    "If a later backtrack lands in the same repeated failure, choose an earlier "
    "root-cause decision next time. Before calling backtrack_commit, compare the "
    "memory_summary you plan to write with the Backtrack memory notebook already "
    "visible in the prompt. If the same failed_check or root_cause is already "
    "present, do not submit a semantically identical memory item. Instead, choose "
    "a different or earlier record_uid, name a concrete difference in the new "
    "strategy, or mark the previous diagnosis as uncertain and make the next fork "
    "re-check it. Changing only the order or count of read/list/search calls is "
    "not a concrete strategy difference if the first mutating action repeats the "
    "failed pattern. A strategy is meaningfully different only when it changes "
    "the first mutating action, delays mutation until required evidence or "
    "artifacts exist, or changes the source/schema/classification decision that "
    "made the previous mutation unsafe. "
    "Avoid rewinding to a later LLM turn that merely noticed the verifier "
    "failure, edited symptoms, called backtrack_candidates, or called "
    "backtrack_commit. "
    "Once backtrack_candidates has returned a candidate list for a stuck branch, "
    "treat that branch as commit-only: do not make more ordinary tool calls, do "
    "not verify again, do not call backtrack_candidates again, and do not stop. "
    "Call backtrack_commit. If backtrack_commit is rejected by policy because "
    "the target is too late, not a returned top-level candidate, or the memory "
    "is not executable, keep the same root-cause diagnosis and retry "
    "backtrack_commit with an earlier returned record_uid or a corrected "
    "memory_summary. "
    "The memory_summary passed to backtrack_commit is written for your past self: "
    "write it as structured operational memory with these fields: failed_check, "
    "observed_loop, why_local_repair_failed, root_cause, evidence_observed, "
    "assumptions_to_recheck, rewind_target_reason, "
    "selected_candidate_safety_check, do_not_repeat, corrected_strategy, "
    "do_not_call_before, must_satisfy_before_mutation, mutation_preflight, "
    "first_3_tool_calls, and success_invariant. Treat root_cause as a hypothesis "
    "supported by evidence, not an unquestionable fact. Never write a "
    "task-feasibility verdict into memory: do not state that a checkpoint, stage, "
    "or required artifact is impossible, infeasible, structurally blocked, or has "
    "no method. The verifier is the sole authority on feasibility; a checkpoint "
    "the verifier still reports as failed is pending, not impossible. Record any "
    "suspected limitation under assumptions_to_recheck together with its concrete "
    "evidence, so the next fork tests it with a real submission before treating "
    "it as fact. Judge candidate solutions only by verifier feedback, never by "
    "your own self-check: if the current branch holds a produced artifact that "
    "has not been submitted since it was last modified, submit it and read the "
    "verifier feedback before calling backtrack_commit. evidence_observed must "
    "quote the verifier's latest failure feedback verbatim and "
    "say what tool output, verifier feedback, or file content supports it; "
    "assumptions_to_recheck must name any uncertain source choice, schema "
    "interpretation, missing evidence, or tool-capability assumption the next "
    "fork should verify before relying on it. rewind_target_reason must name "
    "the concrete first bad LLM decision and confirm the selected record_uid is "
    "the closest returned LLM decision before that bad decision. "
    "selected_candidate_safety_check must confirm the selected record precedes "
    "the stated rewind boundary, or explicitly say no returned candidate does. "
    "observed_loop must say what repeated repair attempts happened; "
    "why_local_repair_failed must say why those repairs did not change the "
    "verifier outcome or remove the root cause. do_not_repeat and "
    "do_not_call_before are binding red-line constraints for the next fork, not "
    "background advice; name concrete tool calls or action classes that must not "
    "happen before their prerequisites are true, such as move_file before an "
    "original-layout manifest exists or create_directory for a verifier-rejected "
    "month. must_satisfy_before_mutation must list the evidence, artifact, or "
    "classification decision that must exist before any write_file, move_file, "
    "create_directory, edit, delete, or similarly mutating tool call. "
    "mutation_preflight must state the check the next fork should apply before "
    "its first mutating tool call. Include literal filenames, paths, commands, verifier "
    "keywords, or required evidence when known. Do not write vague memory such as "
    "'re-examine the source evidence'; write the operational correction, such as "
    "which search pattern to stop using and which file contents or paths to "
    "inspect instead. The memory must be executable with the tools currently "
    "listed for the agent. Assume the listed tools are sufficient for the task "
    "unless task instructions or concrete tool feedback prove otherwise. Do not "
    "use missing tool, no hashing tool, need runner/operator, or cannot complete "
    "with current tools as the final root_cause merely because a direct helper is "
    "not listed; write it as an assumption_to_recheck and search for an in-task "
    "or listed-tool path instead. Do not invent unavailable tools, shell/Python "
    "access, hashing helpers, external scripts, or human calculation. If the task "
    "needs exact arithmetic, indexing, hashes, or global classification and no "
    "direct compute helper is listed, corrected_strategy must still use existing "
    "read/write/list/search tools and visible task evidence, with "
    "verify_task_state only as a pass/fail check on concrete artifacts instead "
    "of guessing or repeatedly patching symptoms. "
    "If a verifier-checked field requires a cryptographic digest, exact transform, "
    "or other value that is not available from listed tools or existing task "
    "evidence, do not replace it with an empty field, blank field, unavailable "
    "marker, raw-content path, provenance token, external-review note, or "
    "conditional plan to use hidden environment capability. "
    "Instead, rewind to the earliest decision where the strategy should have "
    "searched for authoritative in-task evidence or avoided committing to that "
    "unavailable-value-dependent artifact. "
    "Do not write placeholder checksum values such as UNKNOWN or "
    "SHA256_UNAVAILABLE, do not claim unspecified environment utilities can "
    "compute values for you, and do not rely on the verifier to reveal hidden "
    "expected answers. Do not claim the LLM/model can internally compute "
    "cryptographic hashes or long exact transforms for arbitrary file contents. "
    "first_3_tool_calls must name real "
    "available tool calls, not hypothetical helper calls. "
    "After backtrack_commit, the current agent path is expected to stop; the runner "
    "replays to the chosen LLM record, injects the cumulative backtrack memory "
    "notebook, and continues on a new materialized timeline fork. "
    + ROLLBACK_CONTEXT_VISIBILITY_INSTRUCTIONS
    + ROLLBACK_FUTURE_EVIDENCE_INSTRUCTIONS
    + "In a backtracked fork, the Backtrack "
    "memory notebook is a binding operational constraint for this fork, not "
    "background context or optional advice; it overrides your default task-solving "
    "routine when they conflict. One exception: feasibility verdicts are never "
    "valid memory. If a notebook item claims a checkpoint, stage, or artifact is "
    "impossible, blocked, or has no method, treat that claim as an untested "
    "assumption to re-check with a real attempt and submission, not as a "
    "constraint; only the verifier's own feedback decides feasibility. Your first "
    "non-trivial actions after rewind must "
    "address the notebook directly. Do not resume the old workflow until you have "
    "satisfied the notebook's must_satisfy_before_mutation items or gathered the "
    "evidence required by corrected_strategy. Before any filesystem action in the "
    "new fork, identify the applicable memory item, restate the do_not_repeat "
    "constraint in your own words, and decide whether the diagnosis is stable "
    "enough to mutate. Before every mutating tool call (write_file, move_file, "
    "create_directory, edit, delete, or any task tool that changes state), apply "
    "the memory as a preflight gate: ask whether this action repeats the failed "
    "mutation pattern, whether the required evidence or artifact already exists, "
    "and whether the first mutating action is materially different from the "
    "abandoned branch. If a planned mutating action conflicts with do_not_repeat, "
    "do_not_call_before, failed_check, or mutation_preflight, do not call that "
    "tool yet; first gather the missing evidence, create the required prerequisite "
    "artifact, choose a non-mutating check, or backtrack again. If the memory "
    "indicates uncertain source choice, missing evidence, schema ambiguity, or "
    "tool-capability uncertainty, first re-check the relevant visible evidence "
    "before mutating files. If the memory identifies a concrete wrong mutation "
    "and a concrete replacement edit, you may mutate directly only when the "
    "replacement does not violate the notebook constraints. Choose a first action "
    "that implements corrected_strategy or gathers the evidence it requires, and "
    "only depart from the memory when new visible evidence in this fork proves it "
    "wrong."
)

# --- upstream lines 306-451 ---
BACKTRACK_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "backtrack_candidates",
        "description": (
            "Return candidate LLM records that can be used as backtrack targets. "
            "Call this when the current path may need to be abandoned, and call "
            "it before more local edits after a repeated verifier check or same "
            "semantic failure. Each returned item includes a record_uid plus "
            "decision and tool previews to help choose the LLM decision before the bad "
            "outcome while preserving enough useful context. The record_uid is the "
            "LLM decision that will be replayed/re-decided with backtrack memory; "
            "select the closest returned LLM decision before the first bad "
            "planning/search/data-source/file-operation decision. If the same "
            "failure repeats, pick an earlier root-cause predecessor instead of a "
            "late verification or repair turn. Only top-level candidate "
            "record_uid values are valid backtrack targets. In probabilistic "
            "selection mode, use the returned required_estimate_count and "
            "candidate record_uid fields to call backtrack_commit_distribution with "
            "exactly that many returned record_uid values. If there are more "
            "returned candidates than required estimates, choose the most likely "
            "candidates from anywhere in the returned list, not necessarily the "
            "earliest ones. Candidate count and rendering mode are fixed by the "
            "current run configuration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "Brief explanation of why the current path may need "
                        "backtracking. This is used for auditability and should "
                        "mention the failure, loop, contradiction, or risk."
                    ),
                },
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "backtrack_commit",
        "description": (
            "Commit to abandoning the current future branch and rewind to a prior "
            "LLM record. Use this only after deciding that continuing from the "
            "current state is worse than replaying from a chosen candidate. The "
            "record_uid should normally come from backtrack_candidates and names "
            "the closest returned LLM decision before the first bad decision to "
            "replace. Do not pass the bad decision itself when an earlier returned "
            "LLM predecessor exists. Do not pass a tool record id or a record id copied from a "
            "tool preview; pass only a top-level candidate record_uid. This call "
            "does not return a normal tool result: in "
            "SDK/runner-native mode it raises BacktrackRequested so the runner can "
            "stop the agent, create a materialized fork, inject the cumulative "
            "memory notebook, and re-exec."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "record_uid": {
                    "type": "string",
                    "description": (
                        "LLM record uid to rewind to, usually selected from "
                        "backtrack_candidates. The target must be kind='llm'; "
                        "tool record ids are not valid commit targets."
                    ),
                },
                "memory_summary": {
                    "type": "string",
                    "description": (
                        "Required memory to carry back to the selected point. Write "
                        "a concise but operational note for the past agent using "
                        "these fields: failed_check, observed_loop, "
                        "why_local_repair_failed, root_cause, evidence_observed, "
                        "assumptions_to_recheck, rewind_target_reason, "
                        "selected_candidate_safety_check, do_not_repeat, "
                        "corrected_strategy, do_not_call_before, "
                        "must_satisfy_before_mutation, mutation_preflight, "
                        "first_3_tool_calls, and "
                        "success_invariant. Treat root_cause as a hypothesis, not "
                        "an unquestionable fact; evidence_observed should name "
                        "the concrete verifier/tool/file evidence supporting it, "
                        "and assumptions_to_recheck should name uncertainties the "
                        "next fork must verify before relying on them. observed_loop "
                        "should summarize the "
                        "repeated attempts on the abandoned branch; "
                        "why_local_repair_failed should explain why those attempts "
                        "did not change the verifier outcome or remove the root cause. "
                        "rewind_target_reason must name the first bad LLM decision "
                        "and confirm that the selected record_uid is the closest "
                        "returned LLM decision before it. selected_candidate_safety_check "
                        "must confirm that the selected record precedes the stated "
                        "rewind boundary, or explicitly say no returned candidate does. If the "
                        "root cause is a filesystem mutation, destructive/polluting "
                        "file operation, or specific move/write/edit/create boundary, "
                        "do not choose a candidate at or after the relevant mutating "
                        "decision shown by tool_calls_preview. "
                        "do_not_repeat and do_not_call_before are binding red-line "
                        "constraints for the next fork, not optional advice. "
                        "must_satisfy_before_mutation must name the evidence, "
                        "artifact, or classification decision required before "
                        "write_file, move_file, create_directory, edit, delete, "
                        "or any other mutating action. mutation_preflight must "
                        "state how the next fork should check that its first "
                        "mutating action is not the same failed pattern again. "
                        "Changing only read/list/search ordering is not a concrete "
                        "strategy difference if the first mutating action repeats "
                        "the failed pattern. "
                        "Include literal filenames, paths, commands, verifier "
                        "keywords, or required evidence when known. Avoid vague "
                        "advice such as 're-examine evidence'; say exactly what to "
                        "stop doing and what to inspect or write instead. The plan "
                        "must be executable with the tools currently listed for the "
                        "agent. Compare this memory with any Backtrack memory "
                        "notebook already visible in the prompt; if the same "
                        "failed_check or root_cause is already present, do not "
                        "write a semantically identical memory item. Choose a "
                        "different or earlier record_uid, state a concrete strategy "
                        "difference, or mark the previous diagnosis uncertain and "
                        "re-check it in the next fork. Assume the listed tools are "
                        "sufficient unless task instructions or concrete tool "
                        "feedback prove otherwise. Do not use missing tool, no "
                        "hashing tool, need runner/operator, or cannot complete "
                        "with current tools as the final root_cause merely because "
                        "a direct helper is not listed. Do not invent unavailable "
                        "tools, shell/Python access, hashing helpers, external "
                        "scripts, LLM-internal hashing, "
                        "placeholder checksum values, verifier-as-answer-oracle "
                        "plans, human calculation, raw-content/path tokens in "
                        "checksum fields, empty checksum fields, external-review "
                        "notes, or conditional hidden-environment capability. "
                        "first_3_tool_calls must name real available tool calls."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Optional audit reason for this committed backtrack."
                    ),
                },
            },
            "required": ["record_uid", "memory_summary"],
            "additionalProperties": False,
        },
    },
)

# --- upstream lines 831-907 ---
_FEASIBILITY_VERDICT_PHRASES = (
    "impossible",
    "infeasible",
    "not feasible",
    "unsolvable",
    "unachievable",
    "not achievable",
    "no method exists",
    "no approach exists",
    "no way to",
    "structurally block",
    "structural block",
    "structurally impossible",
    "dead end",
    "dead-end",
    "give up",
    "hopeless",
    "cannot succeed",
    "cannot be done",
    "cannot be achieved",
    "cannot be completed",
    "cannot be satisfied",
)

MEMORY_VERDICT_REJECTION = (
    "memory_summary rejected: it contains a task-feasibility verdict ({phrase!r}). "
    "The verifier is the only authority on feasibility; a checkpoint the verifier "
    "still reports as failed is pending, not impossible. Rewrite the memory as "
    "operational facts only: quote the verifier's latest failure feedback verbatim "
    "in evidence_observed, list the concrete actions and edits already tried in "
    "observed_loop, and record the suspected limitation under assumptions_to_recheck "
    "so the next fork tests it with a real submission instead of inheriting it as "
    "fact. If the current branch holds an untested candidate artifact, submit it and "
    "read the verifier feedback before backtracking."
)

MEMORY_DUPLICATE_REJECTION = (
    "memory_summary rejected: it is near-identical (token overlap {overlap:.0%}) to "
    "an existing Backtrack memory notebook item. Re-submitting the same diagnosis "
    "does not change the next fork's behavior. Either choose an earlier or different "
    "record_uid with a materially different corrected_strategy (a different first "
    "mutating action), or, if the current branch holds an untested candidate "
    "artifact, submit it for verification instead of backtracking again."
)

MEMORY_DUPLICATE_OVERLAP_THRESHOLD = 0.75


def _memory_summary_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_./-]+", text.lower()) if len(token) > 2}


def find_feasibility_verdict(memory_summary: str) -> str | None:
    lowered = " ".join(memory_summary.lower().split())
    for phrase in _FEASIBILITY_VERDICT_PHRASES:
        if phrase in lowered:
            return phrase
    return None


def find_duplicate_memory_overlap(
    memory_summary: str,
    memory_items: list[dict[str, Any]],
    *,
    threshold: float = MEMORY_DUPLICATE_OVERLAP_THRESHOLD,
) -> float | None:
    new_tokens = _memory_summary_tokens(memory_summary)
    if not new_tokens:
        return None
    for item in memory_items:
        prior_tokens = _memory_summary_tokens(str(item.get("memory_summary") or ""))
        if not prior_tokens:
            continue
        overlap = len(new_tokens & prior_tokens) / len(new_tokens | prior_tokens)
        if overlap >= threshold:
            return overlap
    return None

# --- upstream lines 1306-1318 ---
def candidate_from_segment(
    llm_record: dict[str, Any],
    segment_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "record_uid": llm_record.get("record_uid"),
        "record_kind": "llm",
        "created_at": _record_created_at(llm_record),
        "decision_hint": _decision_hint(llm_record),
        "tool_calls_preview": _tool_calls_preview(llm_record),
        "tool_records_preview": [_tool_record_preview(record) for record in segment_records if record.get("kind") == "tool"],
        "source_mode": "deterministic",
    }

# --- upstream lines 1610-1722 ---
def render_backtrack_memory_notebook(
    memory_items: list[dict[str, Any]],
    *,
    limit: int | None = DEFAULT_MEMORY_LIMIT,
    memory_prefix: str = DEFAULT_MEMORY_PREFIX,
    context_policy: BacktrackContextPolicy = "rollback",
    workspace_policy: BacktrackWorkspacePolicy = "rollback",
    target_record_uid: str | None = None,
    context_source_record_uid: str | None = None,
    item_char_limit: int = DEFAULT_MEMORY_ITEM_CHAR_LIMIT,
    notebook_char_limit: int = DEFAULT_MEMORY_NOTEBOOK_CHAR_LIMIT,
) -> str:
    rendered_items = select_backtrack_memory_items(memory_items, limit=limit)
    lines = [memory_prefix.rstrip(), ""]
    policy_lines = backtrack_memory_policy_banner(
        context_policy=context_policy,
        workspace_policy=workspace_policy,
        target_record_uid=target_record_uid,
        context_source_record_uid=context_source_record_uid,
    )
    if policy_lines:
        lines.extend(policy_lines)
        lines.append("")
    if not rendered_items:
        return "\n".join(lines).rstrip()
    for index, item in enumerate(rendered_items, start=1):
        summary = str(item.get("memory_summary") or "").strip()
        if not summary:
            continue
        summary = _truncate(summary, limit=item_char_limit)
        detail = f"{index}. {summary}"
        target_record_uid = item.get("target_record_uid")
        reason = item.get("reason")
        if target_record_uid or reason:
            annotations = []
            if target_record_uid:
                annotations.append(f"target={target_record_uid}")
            if reason:
                annotations.append(f"reason={reason}")
            detail += f" ({'; '.join(annotations)})"
        lines.append(detail)
    rendered = "\n".join(lines).rstrip()
    if notebook_char_limit and len(rendered) > notebook_char_limit:
        return rendered[:notebook_char_limit].rstrip() + "\n...[backtrack memory truncated]"
    return rendered


def backtrack_memory_policy_banner(
    *,
    context_policy: BacktrackContextPolicy,
    workspace_policy: BacktrackWorkspacePolicy,
    target_record_uid: str | None = None,
    context_source_record_uid: str | None = None,
) -> list[str]:
    if context_policy == "rollback" and workspace_policy == "rollback":
        return []
    target_label = target_record_uid or "unknown"
    context_label = context_source_record_uid or "unknown"
    lines = [
        "Backtrack execution policy for this fork:",
        "",
        f"- Selected backtrack target: {target_label}. This is an internal Replay ID for the LLM decision selected by the accepted backtrack request.",
    ]
    if context_policy == "preserve":
        lines.append(
            f"- Conversation context source: {context_label}. This is the latest LLM record from the abandoned failed branch; agent messages were intentionally not rewound to {target_label}."
        )
    else:
        lines.append(
            f"- Conversation context state: agent messages were rewound to the input messages for target LLM record {target_label}."
        )
    if workspace_policy == "preserve":
        lines.append(
            "- Controlled Git workspace source: the failed branch at the moment backtrack was requested. "
            f"Files were intentionally not restored to the checkpoint for {target_label}."
        )
    else:
        lines.append(
            "- Controlled Git workspace state: restored to the recorded filesystem checkpoint immediately before "
            f"the tool calls produced by target LLM record {target_label} would execute."
        )
    lines.append(
        "The state combination above is intentional for this ablation policy and does not mean that backtrack failed."
    )
    if context_policy == "preserve":
        lines.append(
            "Messages and tool observations after the selected target remain visible by design, but their filesystem claims may be stale. Do not backtrack again solely to remove those messages or because they conflict with current files."
        )
    if workspace_policy == "preserve":
        lines.append(
            "Failed-branch files remain present by design. Do not backtrack again solely because those files were not restored to the target checkpoint."
        )
    lines.append(
        "First inspect the current controlled Git workspace. Backtrack again only when new tool or verifier evidence shows that the strategy needs a different decision point."
    )
    return lines


def select_backtrack_memory_items(
    memory_items: list[dict[str, Any]],
    *,
    limit: int | None = DEFAULT_MEMORY_LIMIT,
) -> list[dict[str, Any]]:
    valid_items = [
        item
        for item in memory_items
        if isinstance(item.get("memory_summary"), str) and item["memory_summary"].strip()
    ]
    if limit is None:
        return valid_items
    if limit <= 0:
        return []
    return valid_items[-limit:]

# --- upstream lines 1910-2005 ---
def _record_created_at(record: dict[str, Any]) -> Any:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get("created_at")
    return None


def _decision_hint(record: dict[str, Any]) -> str:
    output = record.get("output")
    if not isinstance(output, dict):
        return ""
    content = output.get("content")
    if isinstance(content, str) and content:
        return _truncate(content)
    raw_response = output.get("raw_response")
    reasoning = _raw_reasoning_content(raw_response)
    if reasoning:
        return _truncate(reasoning)
    tool_calls_preview = _tool_calls_preview(record)
    return tool_calls_preview if tool_calls_preview else ""


def _raw_reasoning_content(raw_response: Any) -> str:
    if not isinstance(raw_response, dict):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    reasoning = message.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else ""


def _tool_calls_preview(record: dict[str, Any]) -> str:
    output = record.get("output")
    tool_calls = output.get("tool_calls") if isinstance(output, dict) else None
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    previews = []
    for tool_call in tool_calls:
        parsed = _tool_call_preview(tool_call)
        if parsed:
            previews.append(parsed)
    return _json_preview(previews)


def _tool_call_preview(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None
    if isinstance(tool_call.get("function"), dict):
        function = tool_call["function"]
        return {
            "name": function.get("name"),
            "arguments": function.get("arguments"),
        }
    return {
        "name": tool_call.get("name"),
        "arguments": tool_call.get("arguments", tool_call.get("args")),
    }


def _tool_record_preview(record: dict[str, Any]) -> dict[str, Any]:
    input_record = record.get("input") if isinstance(record.get("input"), dict) else {}
    error = record.get("error")
    return {
        "tool_name": input_record.get("tool_name"),
        "arguments_preview": _json_preview(input_record.get("arguments")),
        "status": "error" if error else "ok",
        "output_preview": _json_preview((record.get("output") or {}).get("value") if isinstance(record.get("output"), dict) else None),
        "error_preview": _json_preview(error),
        "effects_preview": _json_preview(record.get("effects")),
    }


def _json_preview(value: Any, *, limit: int = 400) -> str:
    if value is None:
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = repr(value)
    return _truncate(text, limit=limit)


def _truncate(value: str, *, limit: int = 400) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."
