# What a version-controlled memory still loses

Registered on 2026-11-05, before any full run. The code is `studies/memory_probe/` and `studies/long_run/`, and the
memory under test is `taste/memstore/`. Settings still to be chosen on the small runs are named as such below, and are
frozen in a commit before the first full cell.

## Why

An agent whose history is compacted again and again can lose what it was told, and nothing tells it so. Keeping an
agent's memory in version control is established practice. What has not been measured is how much such a memory still
loses, and at which step, when the agent has to look for what it no longer knows it forgot.

## Question

Across long runs and repeated compaction, how much of what an agent was told does a version-controlled memory with
search and fetch still lose, against compaction alone and a flat markdown file, and at which step?

**Context rot** here means the loss, as a run gets longer and is compacted more often, of information the agent was
told earlier, measured as the agent not using it correctly at the moment it applies, although the same model does when
told directly at that moment. Four steps are reported apart:

- **stored**: the record still holds it (true by construction for an append-only record, so it proves little);
- **looks**: the agent searches, or the system delivers;
- **finds**: the result holds the statement;
- **uses**: the action obeys it. On the rule-loss cases this is the A or B choice; only the long run scores actions.

Effective retention is reported as the rule-loss benchmark defines it: (with memory − without the statement) /
(statement restated at the moment − without the statement).

## Arms

The same model, histories and memory budget in every arm, counted with one tokenizer.

| Arm | Store | How the agent gets it |
|---|---|---|
| Compaction | none | the summary |
| Compaction, told | none | the summary, from the same prompt with one sentence added: keep anything the user asked the assistant to keep doing |
| Flat, delivered | the whole conversation as a flat file | read whole, cut to the budget |
| Notes, last version | a notes file one model keeps as the conversation goes | search and fetch over its last version |
| Notes, every version | the same writes, every version kept in git | search and fetch across all its versions |
| Notes, newest read whole | the same writes | the newest version whole, with no search |
| Notes, newest and older | the same writes, every version kept in git | the newest version whole, then what one search finds in older ones |
| Taste memory | each turn a commit in `taste.memstore` | search and fetch tools |
| Taste memory, headers | the same, with a model-written header per commit | search and fetch tools |
| Delivered rules | rule records from a named reader | put in front of the agent at every call |
| Delivered rules, in the headers | the same records, each in the header of the commit of its turn in `taste.memstore` | read back from the store and put in front of the agent at every call |
| Both | the header arm and the delivered rules | both |
| Searched for it | the Taste store, with and without headers, and both notes files | the harness searches once with the request's own words and delivers what it finds |
| Header index | the Taste store, with a model-written header per commit | what each header says, listed in the prompt and cut to the budget; fetch offered, no search |

A flat list searched by the same code holds the same text under the same tokenizer and scorer as Taste with headers
off, so the two must rank alike: a test holds them to it and they are one row. Any gain over compaction in that row
belongs to search, not to git. With an append-only record, revised rules tie too, since a recency tie-break works on
file position as well as on commit order. Headers could sit in a flat file, so they test structure, not git. Version
control can separate only where content is overwritten in place, which is what the two notes arms test.

The note-taker rewrites the whole file after each round. Whether it is also told to keep what the user asked it to keep
doing is a setting, since that clause may matter more than the store. Both are run, and the same sentence is run on the
summary prompts, so that the store and the instruction can be read apart. Whole-text matching cannot see a rule put in the
note-taker's own words, so what each notes arm delivers is read by the benchmark's retention judge, the one the study
calibrated at one compaction.

The delivered rules come from the study's reader (`studies/context_rot/criteria.py`), a model reading each user message with a
prompt close to the benchmark paper's own fix, so that arm is a baseline. Writing the same records into commit headers changes
where they are kept, not who wrote them, so that row is read the same way and names the reader wherever it is reported. Each
summary prompt's run keeps the records it delivered, since the reader's readings differ between runs in 17 of the 150 cases.

## Prompt conditions

The benchmark's developer prompt says "Do not execute a tool for the current request", and its user prompt asks for one
letter. A model asked this way does not search: in the first 250 probes, across conditions i to v, it searched in none. So
"looks" cannot be measured in the benchmark's own format. Whether an agent looks is the point of the study, so it is
measured on the same histories given as plain requests, and in the long run.

| Condition | Developer prompt |
|---|---|
| i | the benchmark's words, tools offered with neutral names and descriptions |
| ii | i, and one sentence saying a searchable record of the earlier conversation exists |
| iii | that one clause narrowed to "Do not execute a tool to carry out the current request" |
| iv | iii and the sentence |
| v | iv and one sentence telling the agent to check that record before it answers |

| vi | a plain request: the summary and the bare request, no options, the memory tools and the benchmark's 13 simulated tools offered as real tools |
| vii | vi and the sentence |
| viii | vii and the sentence telling the agent to check that record |

Nothing in any condition mentions rules or instructions. Conditions v and viii are designs an operator could choose. Under
vi to viii no option is given, so these report "looks" and whether the agent acted, not "uses". Since the agent does not
look under i to v, "finds" and "uses" come from the harness searching once with the request's own words, as memory products
usually do. Which of these the full matrix runs is chosen from the small run and frozen before it.

## Situations

**Cued.** The same history, rule and options, with one hand-written cue inside the request naming the rule's subject
without its content. The 15 cues are in `studies/memory_probe/variants.py`, written before any run of these cases, after
the first 100 probes of the unchanged cases had been read.

**Uncued.** The benchmark as published. Every published case states its rule in the first user turn, so anything that
keeps the start of a history wins by position alone, and anything that keeps the end loses by it. The flat file keeps
the newest text, as a context window does, and the same histories are also run with the rule in the median and in the
last user turn, where the benchmark's own builder puts it.

**Revised and withdrawn.** The same histories with a later statement in the median user turn that reverses the rule
(the right answer flips) or takes it back (the old rule's option is then read against the no-rule rate). The 15
revisions are in the same file and were written at the same point, before any run of these cases and after those 100
probes had been read. Git keeps old versions, and a search can return a stale one; each hit says when it was recorded.

**Long run.** One coding agent with compaction on (below).

## Rule-loss matrix

Hermes, the first 10 histories (150 cases); one and 16 rounds of compaction; the Pi and Anthropic summary prompts.
Reader gpt-oss-120b on vLLM 0.32.0, as in the benchmark's own table, reasoning effort low, up to 6 tool rounds. Each case
gets a fresh store: the 15 cases of a history differ only by the rule. Added in this order as time allows: WildChat, and 2,
4 and 8 rounds.

Frozen from the small runs on the first 50 cases, before the first full cell. The memory budget is 4,096 tokens, with
1,200 and 8,000 reported on one cell as a sweep. The multiple-choice cases run under condition i with the harness searching,
for "finds" and "uses": the turns with headers off and on, each notes file under both note-taker wordings, and the delivered
rules alone, in the headers, and with search. The header list runs under condition i, the flat file keeps the newest text, and
the plain requests run under conditions vi, vii and viii with the turns searched by the agent, for "looks". The retention judge
reads what each harness-searched arm delivers. Every case keeps its record, and the first full cell is read before any other
starts. Cued, moved, revised and withdrawn cases follow, once each has been read on the first 50.

Added after the first full cell was read, before any other cell ran, because the small runs showed the search over every notes version
favours short early versions: the newest notes read whole, with and without one search of the older versions after them, under both
wordings. They run on every cell, the first included.

## The long run

mini-swe-agent 2.4.6, its default agent, with three hooks: a compaction step at a stated token count, using the two
published summary prompts; a user message at a chosen model call; and the memory tools answered on the host, with
nothing run in the task's container for them. The paper says the agent was changed. A rule stated once in the first
message, a rule given at a later call, and a fact the agent finds out itself and needs later are planted, and the harness
never sends them again; after each compaction the raw prompt is checked for that, and a run where the harness re-sent
one is invalid. Arms: compaction only; a flat list searched by the same code; Taste's memory, with nothing said about it, with
one sentence saying a searchable record exists, and with that sentence and one saying to check the record; with ceilings of no
compaction and of the statement restated. Size: a free dry run through every stage, then 3 to 5 sessions, then more only after
reading them.

**Tasks and machines.** A session chains public Terminal-Bench 2.1 tasks in one conversation, in an order a seed fixes, each on a
fresh machine built from the task's published image, leaving out tasks too big for a share of a shared node, needing a machine
inside the machine, or with no Python on the machine. Machines run under Apptainer on a shared cluster, not Docker, as root inside
a user namespace, with the package installer told not to drop to its own user; a package whose install changes file owners may
only half install. The task's own score plays no part. The model is gpt-oss-120b, served on the same cluster. mini-swe-agent's
published prompts are used, with the submit command added to the system prompt, because a summary may lose the first message.

**What is planted and how it is scored.** All three were chosen before any session ran, and scored by patterns over the command
record. Whenever you run Python, pass it -B (told with the first task; agents do not do this unprompted). When you list a folder, use
ls -1 (told at a chosen model call). A tag in a file on the first machine only, which the agent reads itself and must print in
every task, scored from the second task on, in commands or their output. No message the harness sends holds the tag.

## Before scaling

- The compaction baseline must lose something on the chosen model and prompt. If it does not, that is the finding.
- A model whose tool calls parse less than 95% of the time stops the tool arms until fixed.
- A run stops at the first surprise in any batch, and a batch's answers are read, not only counted.

## Claims and analysis

"Stored: 0 of N lost, by construction." For any other step, "0 of N (95% upper bound about 3/N)", naming the step.
Compliance is never read against zero: each probe has two options, so a model the rule never reached obeys it about half
the time (15 of 30 in the first harness-search probes), and every arm is read against that rate, the benchmark's no-rule
rate on the same cases.
Paired tests per case against the compaction arm, intervals by a bootstrap over histories, since the 15 cases of a
history share it. Every clean result is reported, whichever way it falls.

## Record

Every case keeps each tool call whole with its output, the final answer, the steps, the token counts and a ledger entry
naming the code that ran: a Taste row refuses to count unless its process imported `taste.memstore` and made the calls
it reports. Zero searches are allowed only where the agent chose, and that rate is what "looks" reports.
