---
title: "Beyond the Harness: An Operating System for AI Agents"
date: "2026-03-28"
description: "Why the industry's biggest problem isn't the model — it's the infrastructure. And why git, not markdown, should be the memory layer."
tags: ["Agent Harness", "Git Worktree", "Context Window", "Tool Use", "Inter-Agent Communication", "Orchestration", "Monitoring", "Rollback Strategies"]
---

## The Infrastructure Problem Nobody Wants to Solve Twice

There are thousands of agent startups right now. AI coding assistants, customer support bots, autonomous research agents, financial analysts, DevOps automators. They're all building different applications. And they're all rebuilding the same infrastructure from scratch.

Every team that ships a non-trivial agent eventually hits the same walls: How do you keep an agent coherent when the task exceeds a single context window? How do you roll back when an agent goes off the rails at step 47? How do you coordinate multiple agents working on the same problem without them overwriting each other's work? How do you let a human see what an agent actually did during a four-hour autonomous run?

These aren't application-level problems. They're infrastructure problems. And right now, every team solves them ad hoc — a `progress.md` file here, a compaction strategy there, a retry loop bolted on after the first demo fails in production.

Here's a scenario that teams shipping with Claude Code, Codex, and Cursor will recognize: you kick off a long-running coding agent to build a feature. It works for two hours, making steady progress. At step 87, it silently introduces a regression — a small change that breaks an assumption three files away. By step 140, the regression has compounded: the data pipeline is broken, and the agent has confidently built 50 more steps on top of the broken foundation. You now have two choices: throw away two hours of work and restart, or spend an hour manually untangling which changes were good and which were contaminated. Neither is acceptable.

This is the React-before-React moment. Before React, every frontend team built their own DOM manipulation, state management, and component lifecycle. It was wasteful, error-prone, and it meant teams spent 60% of their time on plumbing instead of product. React gave them a declarative abstraction: describe what your UI should look like, and the framework handles the rendering.

The agent ecosystem needs the same thing. Not another framework. Not another SDK. An operating system — a general-purpose infrastructure layer where developers plug in their agents and the system handles the heavy lifting: orchestration, context management, state persistence, monitoring, rollback, inter-agent communication.

I've been thinking about this architecture since I started building multi-agent systems, and the timing is right to write it up. Anthropic, OpenAI, LangChain, and a growing community of practitioners have all converged on the same insight in the past few months: **the harness is the bottleneck, not the model.** They're all publishing about agent harnesses, and they're all independently reaching for the same analogy — the harness is an operating system. But nobody is designing it like one.

This post is my attempt to do that.

![Agent OS Stack Mapping](/assets/agent_harness/figure-1-agent-os-stack.svg)

---

## Everyone Sees the OS Analogy. Nobody Finishes It.

Phil Schmid [framed it explicitly](https://www.philschmid.de/agent-harness-2026): the model is the CPU, the context window is RAM, the agent harness is the operating system, the agent is the application. LangChain's [Anatomy of an Agent Harness](https://blog.langchain.com/the-anatomy-of-an-agent-harness/) reinforced it: "Agent = Model + Harness. If you're not the model, you're the harness." Anthropic's [context engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) describes context as "a finite resource with diminishing marginal returns" — the same way OS designers describe physical memory.

The analogy is right. But everyone stops at the metaphor.

Real operating systems don't just have a CPU and some RAM. They have process schedulers that assign work to cores. They have virtual memory systems that load pages from disk on demand. They have inter-process communication protocols so processes can coordinate. They have file systems that are navigable, hierarchical, and persistent. They have permission models and sandboxing.

Current agent harnesses have none of this as integrated, composable architecture. They have individual features — compaction here, skills there, a progress file somewhere — stitched together in framework-specific ways. It's the equivalent of building an operating system by writing a collection of shell scripts.

Here's what it looks like when you take the analogy seriously:

| OS Concept | Agent OS Equivalent |
|---|---|
| **CPU (Multi-Core)** | LLM models — each core specialized for a role (planner, implementer, reviewer, monitor) |
| **Process Scheduler** | Task decomposition engine — give each agent minimal viable work with minimal context |
| **Persistent Storage** | Git worktree — structured, versioned, navigable state |
| **RAM / Virtual Memory** | Context window — selective checkout of relevant commits and branches on demand |
| **File System** | Everything is AI-readable: tools, states, agents all have structured descriptions |
| **IPC** | Agents read/write shared state through git; orchestrator routes messages mid-execution |
| **Syscalls** | CLI for local operations, MCP for remote/authenticated services |
| **Process Isolation** | Each agent runs in its own branch/worktree — safe, isolated, rollbackable |
| **Package Manager** | Agent registry (`agent_desp.md`) + tool registry (`tool_desp.md`) — install, discover, compose |
| **Task Manager** | Memory dashboard — every tool call, every decision, every rollback visible to end users |

The rest of this post unpacks each of these mappings and explains why they're not just metaphors — they're design decisions with concrete engineering implications.

---

## CPU: Multi-Core Orchestration and Task Decomposition

In a real CPU, cores don't all do the same thing. Modern processors have performance cores and efficiency cores. GPUs have thousands of specialized compute units. The insight translates directly: **not every agent task needs the same model, and no single model should do everything.**

Bassim Eledath [describes this pattern at Level 7](https://www.bassimeledath.com/blog/levels-of-agentic-engineering) of his agentic engineering framework: "I routinely dispatch Opus for implementation, Gemini for exploratory research, and Codex for review, and the cumulative output is stronger than any single model working alone." He also makes the crucial point that you should never let the same model grade its own exam — separate the implementer from the reviewer.

In the Agent OS, the CPU is a multi-core processor where each core is a specialized LLM:

**Core 0 — Planner.** Takes the high-level task, performs decomposition, produces a structured execution plan. This is where you spend the most reasoning compute. LangChain's [harness engineering experiments](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) validated a "reasoning sandwich" pattern — `xhigh` reasoning on planning and verification, `high` on implementation — that outperformed uniform reasoning across all steps. The planner's job is to break work into the smallest viable units so each subsequent agent gets exactly the context it needs and nothing more.

**Core 1–N — Workers.** Each worker agent gets a single, well-scoped subtask with the minimal context required to complete it. This is the most important scheduling principle in the entire architecture: **every token in an agent's context window that isn't directly relevant to its current subtask is noise.** A frontend agent doesn't need database schemas. A data retrieval agent doesn't need UI design prompts. Minimal context means maximal focus, less context rot, and cheaper inference.

**Core N+1 — Monitor.** A dedicated evaluation thread that runs at every state transition, not as a post-hoc review. The monitor writes lightweight tests or assertions based on the current checkpoint, evaluates whether to proceed, rollback, or inject corrective feedback. This is the agent equivalent of CI/CD — automated quality gates that catch drift before it compounds. Why a separate agent? Because models are terrible at evaluating their own work. Anthropic's latest [harness design post](https://www.anthropic.com/engineering/harness-design-long-running-apps) found that when asked to self-evaluate, agents "tend to respond by confidently praising the work—even when, to a human observer, the quality is obviously mediocre." Their solution: a separate evaluator agent — the same architecture the Agent OS proposes as a first-class OS primitive.

**Core N+2 — Orchestrator.** Handles inter-agent communication, merge conflict resolution, and dynamic context routing. When Worker A produces output that Worker B needs, the orchestrator commits the output to shared state (git) and notifies B to selectively check out only the relevant pieces. When two workers modify overlapping state, the orchestrator resolves the conflict or escalates to the human. This is the kernel — the one component that sees the full picture.

### Why Single-Agent Harnesses Hit a Ceiling

Today's best tools — Claude Code, Codex, Cursor — run a single agent with a large context window. When context fills up, they compact (summarize and restart). When state needs to persist, they write to `progress.md` or a `docs/` directory. This works remarkably well for individual coding tasks.

But it's fundamentally single-threaded. It's like running an entire operating system on one CPU core and hoping the clock speed is fast enough. You can't parallelize. You can't specialize. You can't isolate failures. When the single agent drifts at step 87 of a 200-step task, your only option is to restart or hope compaction preserved the right context.

Multi-core means parallelism, specialization, and isolation. It means each core can run a smaller, cheaper model with a tighter context window. It means a failure in one worker doesn't corrupt the entire system. And it means a dedicated monitor thread catches problems before they compound — not after the agent has confidently declared victory.

![Single-Agent vs Multi-Core Orchestration](/assets/agent_harness/figure-2-single-vs-multicore.svg)

---

## Memory: Doc Is Doomed — Git Worktree as the Structured Memory Layer

This is the section where I'm going to be the most opinionated, because I think the entire industry is converging on the right problem but reaching for the wrong solution.

### The Problem Everyone Sees

Jim Yagmin's [Your Docs Directory Is Doomed](https://yagmin.com/blog/your-docs-directory-is-doomed/) articulates it precisely. You start with a `CLAUDE.md`, add an `ARCHITECTURE.md`, start generating specs, extract shared context into more markdown files, and before you know it you have a `/docs` directory that's become the de facto memory system for your agents. The problem is that this memory system has no integrity guarantees:

**Discoverability.** How does an agent know it should read a specific document at the right time? The doc might exist, but if it's not referenced in the right context, it's invisible.

**Doc rot.** Small inconsistencies accumulate silently. There's no trigger when a doc goes stale. No automated way to detect that the architecture doc now contradicts the implementation.

**No hierarchy.** Flat markdown files have no inherent structure connecting them. Agents can't navigate relationships between documents — they can only read them linearly, one at a time.

**Velocity mismatch.** Codebase context changes with every commit. Architecture docs might not change for months. There's no mechanism to manage context that changes at different rates.

**Not composable.** You can't build process context on top of concept context. There's no layering, no inheritance, no structured relationships.

An ETH Zurich paper, [Evaluating AGENTS.md](https://arxiv.org/pdf/2602.11988), put hard numbers on this. Across multiple coding agents and models, LLM-generated context files **decreased** task success rates while increasing inference costs by over 20%. Even developer-written context files provided only marginal improvement (+4% on average). The researchers concluded that "unnecessary requirements from context files make tasks harder." Flat markdown that gets dumped into context is noise masquerading as signal.

### The Partial Solutions the Industry Has Found

The best teams have discovered that git helps, but they're only using a fraction of what it offers.

**Anthropic's approach** ([Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)): Each agent session starts by reading `claude-progress.md` and scanning `git log`. A JSON feature list tracks what's done and what isn't. Git commits provide rollback. The initializer/coder agent split ensures the first session sets up structured artifacts for future sessions.

**OpenAI's approach** ([Harness Engineering](https://openai.com/index/harness-engineering/)): Knowledge lives in a structured `docs/` directory that serves as the system of record. A short `AGENTS.md` (~100 lines) is the table of contents. Every discussion, decision, and architectural pattern gets pushed into the repo, because "anything the agent can't access in-context effectively doesn't exist."

Both approaches are smart. Both use git. **Neither uses git as the architecture.** They treat it as a persistence layer — save progress, revert mistakes, read history. That's using git like a save button — valuable, but a fraction of what the primitive offers.

### The Proposal: Git IS the Memory System

What I'm proposing is that every OS-level memory concept maps directly onto a git primitive — and that the mapping is so clean it can't be a coincidence:

**Branches are execution contexts.** Each agent operates on its own branch. Agent A works on `feature/dcf-analysis`. Agent B works on `feature/technical-indicators`. They're isolated from each other — the same reason your OS runs each process in its own memory space.

**Commits are checkpoints.** Not at the end of a task — at every meaningful state transition. A fine-grained commit history means a new agent session doesn't need to parse a progress file. It runs `git log --oneline` and sees the exact trajectory. It runs `git diff HEAD~3` to understand recent changes. It runs `git show HEAD:src/analysis.py` to inspect any specific file. All structured. All precise. No guessing.

**Merge conflicts are coordination signals.** When two agents modify overlapping state, git surfaces this as a conflict. This is a structured coordination primitive — far more reliable than two agents writing to the same `progress.md` and hoping for the best.

**Worktrees are parallel execution environments.** `git worktree` creates separate working directories for different branches, all sharing the same repository. Each agent gets its own worktree. They run in parallel without filesystem interference, and the orchestrator merges their work when they're done.

The industry is already moving toward this architecture. [GitButler](https://gitbutler.com), backed by [a16z's Series A](https://www.a16z.news/p/investing-in-gitbutler) and founded by GitHub co-founder Scott Chacon, recently shipped [agent integration](https://blog.gitbutler.com/gitbutler-agent-assist) with parallel branch support for AI coding workflows. But their approach uses virtual branches in a single working directory rather than true git worktrees — and users have [already surfaced the isolation problems](https://github.com/gitbutlerapp/gitbutler/discussions/12228) this creates: concurrent file modifications can overwrite each other, and there's no physical separation between agents' work. Worktree-based isolation is [planned but has no timeline](https://github.com/gitbutlerapp/gitbutler/discussions/12228). This is precisely the failure mode that the Agent OS architecture avoids by design — worktrees provide real filesystem-level process isolation, not a UI abstraction over a shared working directory.

**Selective checkout is context loading.** Instead of dumping a progress file into context, an agent runs `git show other-branch:path/to/file` to load exactly what it needs from another agent's work. This is the virtual memory analogy: load pages from disk into RAM on demand, not all at once.

**Git log is the audit trail.** Every decision, state change, and rollback is recorded with a descriptive commit message. The execution history is navigable, searchable, and immutable. No separate progress file needed — the progress IS the commit history.

### A Concrete Comparison

Here's what a new agent session looks like under Anthropic's approach:

```bash
# Current approach: parse unstructured text
cat claude-progress.md    # Read the whole progress file (500+ lines for a big project)
git log --oneline -20      # Scan recent commits
# Then the agent reads, interprets, and hopes the progress file is accurate
```

Here's the same moment under the Agent OS:

```bash
# Agent OS approach: structured queries against versioned state
git log --oneline -10                          # Trajectory at a glance
git diff main..HEAD                            # What has this branch actually changed?
git show monitor-branch:checkpoints/latest.json # Did the last checkpoint pass or fail?
git log --all --oneline --graph                # What are all agents doing right now?
git show worker-b:output/analysis.json         # Load a specific result from another agent
```

Every query returns structured data. There's no "reading a file and hoping it's current" because git history is immutable. There's no discoverability problem because branches, tags, and paths are inherent navigation mechanisms. And this scales — a git repo handles millions of commits; a progress file becomes unwieldy after 50 entries.

![Memory System Comparison](/assets/agent_harness/figure-3-memory-comparison.svg)

---

## Tools: CLI-First, MCP-Second

The interface layer of an OS — the syscalls — determines how applications interact with the kernel. In the Agent OS, the equivalent question is: how do agents invoke capabilities?

### CLI as the Default

Peter Steinberger, creator of [OpenClaw](https://openclaw.im) (the fastest-growing repo in GitHub history), said it plainly: "Most MCPs should be CLIs. The agent will try the CLI, get the help menu, and from now on we're good." His agent famously transcribed a voice message with zero pre-built capability — it inspected the file header, found FFmpeg on the system, discovered an OpenAI API key in the environment, and called the Whisper API via `curl`. Pure CLI. Zero tool definitions loaded into context.

Bassim Eledath [observes the same trend](https://www.bassimeledath.com/blog/levels-of-agentic-engineering): "It's becoming common for LLMs to use CLI tools instead of MCPs... The reason is token efficiency. MCP servers inject full tool schemas into context on every turn whether the agent uses them or not. CLIs flip this."

Anthropic's own [code execution with MCP blog](https://www.anthropic.com/engineering/code-execution-with-mcp) demonstrates the math: agents that discover tools by walking a filesystem — listing directories, reading individual tool definitions on demand — achieved a **98.7% reduction in token usage** compared to loading all definitions upfront. That's not a marginal improvement. It's a different paradigm.

### Where MCP Wins

The CLI-only take has one real blind spot: **authentication.** OAuth flows for Google, Slack, Jira, Salesforce — handling these in a shell context is painful. MCP servers manage auth at the server level and expose clean interfaces. The 80% use case for knowledge workers — searching Slack, creating Jira tickets, sending emails, querying databases — involves remote, authenticated APIs. You can do this with `curl` and manual token management, but you're reimplementing what MCP already abstracts.

### The Agent OS Position

**CLI for local operations. MCP for remote/authenticated services.** This isn't a compromise — it's a principled design decision:

- File manipulation, git, scripts, code execution → CLI (the agent already has a shell)
- SaaS APIs, web search, anything requiring OAuth → MCP (managed auth is the value prop)

The tool registry builds on this with **lazy loading and filesystem-based discovery**:

- Tools organized in directories with `tool_desp.md` descriptions at each level — the LLM navigates the hierarchy, reading descriptions to find what it needs
- Recently used tools are cached (temporal locality — the same principle that makes CPU caches work)
- Agents can compose tools dynamically by writing code, creating new capabilities on the fly

This is Anthropic's `Skills` concept — "folders of reusable instructions, scripts, and resources" — formalized as an OS-level primitive with a structured discovery mechanism.

---

## RAM: Context Engineering as Virtual Memory

The context window is RAM. This isn't just an analogy — the engineering principles are identical.

**What to load (demand paging).** An agent doesn't load the entire repo into context. It runs targeted git queries — `git show`, `git diff`, path-filtered `git log` — to load specific data on demand. This is demand paging: bring pages from disk into RAM only when they're accessed.

Consider a concrete example: Worker B needs the revenue data that Worker A produced. In a flat-file system, someone dumps the entire `output/` directory into B's context — 50K tokens, most of which is irrelevant. In the Agent OS, Worker B runs `git show worker-a:output/revenue_summary.json` — loading exactly the 2K tokens it needs. That's a 96% reduction in context consumption for a single cross-agent data access. Multiply this across dozens of interactions in a long-running task and the cumulative context savings are enormous.

**What to evict (page replacement).** Anthropic's [compaction](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) is page eviction — summarizing old context to make room for new. The Agent OS improves on this: instead of compacting conversation history (lossy), commit intermediate results to git (lossless) and start fresh, loading only what's needed via selective checkout. Nothing is lost — it's just moved from RAM (context window) to disk (git history), retrievable on demand.

**Process isolation (per-agent address spaces).** Each agent operates in its own branch/worktree with its own context window. One agent's context can't pollute another's. This directly addresses the [context rot](https://research.trychroma.com/context-rot) problem: as tokens accumulate, model performance degrades. Small, focused contexts per agent prevent this.

---

## Network: Agent Communication Through Shared State

Most multi-agent frameworks use message passing: Agent A sends a structured message to Agent B. This is synchronous, creates bottlenecks, and loses history once messages are consumed.

The Agent OS uses a different model borrowed from distributed systems: **agents communicate through shared state in git.** Agent A commits its output. Agent B checks out that output when it's ready. The orchestrator manages notifications and dependency ordering.

The advantages over message passing:

**Asynchronous by default.** Agents don't block waiting for each other. They commit results and move on. Dependent agents check out what they need when they're ready.

**Full audit trail.** Every piece of inter-agent communication is a commit with a message, a diff, and a timestamp. You can replay the entire coordination history. When something goes wrong, you can trace exactly which agent produced which state and when.

**Natural conflict detection.** When two agents modify overlapping state, git surfaces this as a merge conflict rather than silently overwriting. The orchestrator can resolve conflicts automatically (if the resolution is deterministic) or escalate to the human.

This is conceptually similar to what Anthropic's [Agent Teams](https://code.claude.com/docs/en/agent-teams) does: multiple instances work in parallel on a shared codebase. Anthropic used 16 parallel agents to build a C compiler from scratch. Cursor reportedly ran hundreds of concurrent agents to build a web browser. Both teams reported the same hard lesson: without structured coordination, agents become risk-averse and churn without progress. Git-based shared state provides that structure.

---

## Software: The Developer Experience

Here's the part that turns architecture into product.

Building an agent in LangGraph today is 200–500 lines of boilerplate: state schemas, node definitions, edge configurations, tool handlers, persistence setup. Every agent team writes this from scratch. It's the equivalent of every web developer hand-writing their own HTTP server before React existed.

The Agent OS developer experience:

**50 lines of code plus a markdown file.** Define your agent's capability in `agent_desp.md` — what it does, what tools it needs, what inputs it expects, and what natural-language patterns should route to it. Write a small function for its core logic. Register it with the OS. The kernel reads the description, schedules the agent appropriately, manages its context window, handles checkpointing and rollback, and coordinates communication with other agents.

Here's what that looks like in practice:

```markdown
# agent_desp.md
name: sec_data_retriever
description: Retrieves and validates SEC EDGAR filings for financial analysis
tools: [search_api, validate_schema]
model: claude-sonnet
triggers: ["financial data", "SEC filing", "10-K", "10-Q"]
```

```python
@agent(config="agent_desp.md")
def sec_data_retriever(query: str) -> dict:
    filings = search_api(query, source="edgar")
    validated = validate_schema(filings, schema="sec_10k")
    return validated
```

That's it. The `triggers` field tells the kernel which natural-language patterns should route work to this agent — when a planner decomposes a task and a subtask mentions "SEC filing" or "10-K", the scheduler knows to dispatch it here. The OS handles scheduling, context management, checkpointing, rollback, and inter-agent communication. Compare this to the 200–500 lines of LangGraph boilerplate that the same agent requires today.

The industry is already validating this description-driven pattern. [AGENTS.md](https://agents.md/) — an open standard under the Linux Foundation, supported by OpenAI, Anthropic, Google, Cursor, and 60,000+ repos — is a markdown file that describes agent capabilities for a repository. The concept of "describe what you need in markdown and the system figures out the rest" is taking hold. The Agent OS extends this from repo-level instructions to a full agent lifecycle.

The registry works like `apt` or `brew`: discover agents by capability, install them by description, compose them dynamically. Ephemeral agents spin up for one-off tasks and tear down when complete — serverless computing applied to agent orchestration.

---

## The Transparency Layer: Why Structured Harnesses Enable End-User Visibility

This is the part most agent builders don't think about, and I think it's one of the most important consequences of the Agent OS architecture.

Today, when a long-running agent works for four hours and delivers a result, you get the output and maybe some logs. You don't know which tools it called at step 23. You don't know that it rolled back a failed approach at step 56 and tried something different. You don't know that it spent 40 minutes going down a dead end before self-correcting. The execution is opaque.

This isn't a logging problem. It's a structural problem. Current harnesses don't have a unified state model that captures every decision. Progress files capture summaries, not decisions. Conversation logs are ephemeral and context-window-sized. Traces require specialized tooling to interpret.

The Agent OS changes this because **every decision is already a commit.** The git history IS the decision log. A dashboard built on top of this structured state can show end users:

- Every tool invocation at every step — what was called, what it returned, why the agent chose it
- Every checkpoint evaluation — what the monitor tested, whether it passed or failed, what action was taken
- The full branch/merge topology — which agents worked on what, when they started and finished, how their work was integrated
- Every rollback — when the system reverted to a previous state, why, and what was tried instead
- Real-time progress — not a stale progress file, but the live commit stream showing work as it happens

![End-User Transparency Dashboard](/assets/agent_harness/figure-4-dashboard.svg)

This is `htop` for agents. Not a debugging tool for developers — a first-class interface that makes autonomous systems trustworthy by making them legible. For regulated industries, for enterprise customers, for anyone who needs to explain "what did the AI actually do?" — this is the answer. And it's only possible because the underlying memory architecture is structured and auditable by design.

---

## Build to Delete: The Bitter Lesson Applied to Agent Infrastructure

Rich Sutton's [Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) argues that general methods leveraging computation beat hand-coded human knowledge every time. We're watching this play out in agent development right now. Phil Schmid [makes the connection explicit](https://www.philschmid.de/agent-harness-2026): capabilities that required complex, hand-coded pipelines in 2024 are handled by a single context-window prompt in 2026. Manus refactored their harness five times in six months. LangChain re-architected their coding agent three times in a year. Vercel [removed 80% of their agent's tools](https://vercel.com/blog/we-removed-80-percent-of-our-agents-tools) and got better results with less.

Anthropic's own team [demonstrated this concretely](https://www.anthropic.com/engineering/harness-design-long-running-apps): when they moved from Opus 4.5 to 4.6, they dropped their entire sprint decomposition system because the newer model could sustain coherence without it. As they put it: "every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing."

This is the uncomfortable truth about harness engineering: **every harness component is a bet against the model.** The Monitor exists because models can't reliably self-evaluate today. The Planner exists because models lose coherence on long task decompositions today. The multi-core split exists because single-agent context windows fill up and degrade today. If any of these assumptions become false with the next model generation — and they might — the corresponding harness component doesn't just become unnecessary. It becomes harmful. An overly prescriptive harness can actively fight a model's native capabilities, forcing it through scaffolding it no longer needs, adding latency and cost for no gain.

The design principle this demands is **composability over rigidity.** Every subsystem in the Agent OS is opt-in, not load-bearing. When a future model can sustain coherence for 500 steps natively, you turn off the Monitor's per-step checkpointing without the system collapsing. When a model can reliably self-evaluate, you remove the separate evaluator agent and let the worker self-check. When context windows grow to a million tokens with minimal degradation, you reduce the multi-core split and let a single agent handle more.

The architecture makes this possible because the subsystems have clean interfaces. The Monitor reads git state and writes checkpoint results — if you remove it, the git state is still there, unaffected. The Planner writes a structured plan to a file — if you remove it, a human or the model itself can write that file. No component assumes another component exists. That's what separates an OS from a monolith.

The stable layer is the abstractions: branches as execution contexts, commits as checkpoints, merge conflicts as coordination signals, selective checkout as context loading. These are borrowed from git — a system battle-tested for two decades. The unstable layer is everything built on top: which model plays which role, how aggressive monitoring should be, whether to decompose or run single-threaded. Those are configuration knobs, not architecture. When the next model generation ships, you turn the knobs. The OS stays.

That said, architectural confidence should come with honest accounting of what's unsolved.

### Open Problems

**Merge conflict resolution by LLMs is hard.** Git surfaces conflicts, but resolving them requires semantic understanding of the code. Current models can handle simple conflicts but struggle with complex semantic merges. This is an active area of research — and one where the monitor thread helps by catching bad resolutions early.

**Git wasn't designed for high-frequency micro-commits.** If every agent checkpoint is a commit, a long-running task could generate hundreds of commits. Git handles this fine at the storage level, but the commit history becomes noisy. Squashing, tagging, and hierarchical branch strategies can mitigate this, but it adds complexity.

**The monitoring overhead is non-trivial.** Running a dedicated evaluation thread at every state transition adds latency and cost. The trade-off is reliability: catching drift at step 10 is cheaper than throwing away 100 steps of contaminated work. But for latency-sensitive tasks, the monitoring frequency needs to be configurable.

**Multi-model orchestration is still early.** Routing the right subtask to the right model — and managing the interface differences between providers — is a systems problem that nobody has solved cleanly. The Agent OS assumes this will mature, but it's a bet on the near future, not the present.

---

## What I'm Building Next

I'm building this — a general-purpose agent harness that implements these primitives: git-native memory, multi-core orchestration, CLI-first tool discovery, structured monitoring, and an end-user dashboard.

The goal is to be for agents what React is for frontend: a declarative abstraction layer where developers describe what their agents do, and the framework handles the orchestration, state management, and operational infrastructure.

If you're building agent systems and hitting the same infrastructure walls — context management that doesn't scale, coordination that breaks under parallelism, transparency that's an afterthought — I'd love to hear about it. [Email me](mailto:zanwen.fu@duke.edu) or find me on [LinkedIn](https://linkedin.com/in/zanwenfu).

---

*Zanwen (Ryan) Fu is a Software Engineer and MS Computer Science student at Duke University, focused on building production-grade agentic AI systems. He joins Robinhood's Agentic AI team as an MLE intern in May 2026. More at [zanwenfu.com](https://zanwenfu.com).*