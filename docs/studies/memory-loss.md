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

- **stored**: the record still holds it. A store of every turn holds it by construction, so that row counts only the cases with the rule's whole
  text in a turn and proves little. A notes file or a rule list may not hold it, so the retention judge reads each: the newest version of a notes
  file first, then its older versions from the first until one holds the rule. Whichever row is read, the judge never sees what the agent did;
- **looks**: the agent searches, or the system delivers;
- **finds**: the result holds the statement;
- **uses**: the action obeys it. On the rule-loss cases this is the A or B choice; only the long run scores actions.

Effective retention is reported as the rule-loss benchmark defines it: (with memory − without the statement) /
(statement restated at the moment − without the statement).

## Arms

The same model, histories and memory budget in every arm, counted with one tokenizer, on a memory's own words. The system prompt is in no store and no file,
and the note-taker is not shown it, since the reader has it already. Where a rule list comes with search, the list is paid for first and what the search brings fills what is left, so
the whole block keeps to the budget. Every row of a run is read against the summary alone and the rule restated in that same run, on the very summaries it was given.

| Arm | Store | How the agent gets it |
|---|---|---|
| Compaction | none | the summary alone, no tools |
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
summary prompts at one round, with its own summaries, so that the store and the instruction can be read apart. Whole-text matching cannot see a rule put in the
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
the first 100 probes of the unchanged cases had been read. A cue changes only the request, so a cued case keeps the published case's summaries, notes and headers,
and is read against the no-rule rate of its own question.

**Uncued.** The benchmark as published. Every published case states its rule in the first user turn, so anything that
keeps the start of a history wins by position alone, and anything that keeps the end loses by it. The flat file keeps
the newest text, as a context window does, and the same histories are also run with the rule in the median and in the
last user turn, where the benchmark's own builder puts it.

**Revised and withdrawn.** The same histories with a later statement in the median user turn that reverses the rule
(the right answer flips) or takes it back. For a rule taken back, "obeyed" means still obeying what was withdrawn, so that row reports how often
the old option is chosen beside the no-rule rate, more or less often, and has no effective retention. The 15
revisions are in the same file and were written at the same point, before any run of these cases and after those 100
probes had been read. Git keeps old versions, and a search can return a stale one. So what the harness delivers says when each piece
was recorded, says so if a notes version has since been replaced (a turn never is), goes oldest first, and comes under a line saying that where two
disagree the later one stands. An agent's own search returns the same marks with no such line, since nothing in any condition mentions rules. The analysis counts whether a reversed rule's old text came back. Three reversals (rules 2, 4 and 6) only
permit something, and the model often does not take them up even when restated, so each rule is read against its own restated row.

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
one is invalid. This is checked on what is about to be sent at every model call, and the whole prompt after each compaction is kept. Arms: compaction only; Taste's memory, with nothing said about it, with one sentence saying a searchable record
exists, and with that sentence and one saying to check the record (a flat list searched by the same code holds the same text and
ranks alike, by the test above, so it is one row with these); with ceilings of no compaction and of every rule told so far restated in the
system prompt at every call. A block a memory delivers does not count toward the compaction trigger, so every arm compacts at the same point of the
same talk. An agent's search and its results stay in the conversation until folded, and they do count, as they would in a real agent: one that searches compacts
sooner, a cost of searching that the record shows. They stay out of the store, since a store holding its own searches finds them again. Added
after the first sessions were read, because those arms hardly searched: the rule list delivered at every call, read back from the
commit headers and written by the study's reader, and the notes file under both note-taker wordings. Added on 2026-11-06, after a look at 39 tasks found 3 memory calls across nine arms and before any session of the arm ran: the harness itself searches the store with each new task's own words, as the harness-searched cases do with the request's, and what comes back stands in the system prompt until the next task, within the same budget. The agent is offered no memory tool in that arm, so its wish to look plays no part. What comes back is headed as history and not new requests, any task in it being over unless the current task says otherwise: the first session of this arm, under the rule-loss cases' header, went back to an earlier task in 30 of its 201 commands, where the nine other arms did so in none of 1,527, because a rule told with a task comes back with that task's words. That header is kept as a variant arm, so that the record shows what the label buys. Every search, its calls and what it brought are on the record. Obedience is read task by task, only at tasks that began after the rule was told, since the rule is live in the task it is told in: where the search brought it back, where it did not but the rule was still in the conversation, and where it was in neither. A fact watched by its value counts as back with that value, and whether the words asking for it came too is read apart. Size: a free dry run through
every stage, then 3 to 5 sessions, then more only after reading them.

**Tasks and machines.** A session chains public Terminal-Bench 2.1 tasks in one conversation, in an order a seed fixes, each on a
fresh machine built from the task's published image, leaving out tasks too big for a share of a shared node, needing a machine
inside the machine, or with no Python on the machine. Machines run under Apptainer on a shared cluster, not Docker, as root inside
a user namespace, with the package installer told not to drop to its own user; a package whose install changes file owners may
only half install. Each machine is an instance with its own process list and its own /tmp, so that files and work left in the background last from one command to the next, a command past its time keeps running there until the machine closes, and a
command can signal only that machine. Before anything the agent sends reaches it, a machine reads what it can see, and one that sees more than 30 processes, or this account's home or the
study's code, is refused like a machine that will not build; what each saw is on the record. The network is the node's, so tasks that open a port are left out, which removes 4 of the 43 usable tasks. The task's own score plays no part. The model is gpt-oss-120b, served on the same cluster; Laguna S 2.1, a stronger open coding
model, was added on 2026-11-06 after gpt-oss-120b's restated ceiling proved low, and is reported apart. mini-swe-agent's published prompts are used, including its
message for a reply with no command, with the submit command added to the system prompt, because a summary may lose the first message. Why a reply had no
command is on record, and three such replies in a row end that task, as in mini-swe-agent, and the next task follows. A summary or a notes version gets a higher output cap than an agent's turn, since reasoning counts toward it; one that comes back cut off is asked again, and a notes version still cut off leaves the file as it was, counted. A summary is what its published prompt asked for and nothing around it: the block for one prompt, the headings for the other. On the first Laguna S 2.1 session 39 of 161 summaries had other text around the block and 16 held no whole summary, 7 of them none at all, one of which left the agent with no task before it. So what surrounds a summary is dropped, a reply with none is asked again, up to three tries, and after the last the fold waits for the next call with nothing thrown away. Every reply and every fold that waited are on the record, for every arm alike. The commands at work on an earlier task are counted for every arm: those naming a file that only an earlier task's words name, and no file of this task's. On Laguna S 2.1 the note-taker's prompt alone made the model carry on the work, with tool calls in 6 of 12 versions of a first look, so in the long run the instruction comes again after the conversation, and a reply holding tool calls or only the words for no notes is asked again and then counted the same way. The rule-loss cases, written once on gpt-oss-120b, keep the prompt as it was. Sessions run under both summary prompts.

**What is planted and how it is scored.** All three were chosen before any session ran, and scored by patterns over the command
record. Whenever you run Python, pass it -B (told with the first task; agents do not do this unprompted). When you list a folder, use
ls -1 (told at a chosen model call). A tag in a file on the first machine only, which the agent reads itself and must print in
every task, scored from the second task on, in commands or their output. No message the harness sends holds the tag. A reply can hold the tag without running
anything, so the tag is reported as ran and as known: known reads what the agent ran and said and any command's output, and not what a memory tool merely showed it. A command's output is kept and scored up to its first 200,000 characters. Whether a summary kept a rule is read by a pattern as well as by its exact words, since a summary uses its own.
Every rule is also scored task by task, and arms compare only on tasks both reached. The first machine that opens gets the rules and the tag; one that fails to take either is left, and the next one gets them. A session is invalid, and left out of every total though still counted, if the harness re-sent a planted text, its ledger was refused, or nothing was planted.

## Before scaling

- The compaction baseline must lose something on the chosen model and prompt. If it does not, that is the finding.
- A model whose tool calls parse less than 95% of the time stops the tool arms until fixed.
- A run stops at the first surprise in any batch, and a batch's answers are read, not only counted.

## Claims and analysis

"Stored: 0 of N lost, by construction", for the append-only store only. For any other step, "0 of N (95% upper bound about 3/N)", naming the step.
Compliance is never read against zero: each probe has two options, so a model the rule never reached obeys it about half
the time (15 of 30 in the first harness-search probes), and every arm is read against that rate, the benchmark's no-rule
rate on the same cases.
Rates are over the answers that could be read, as the benchmark's own summary does, with the unread counted beside them, and a difference is over the pairs where both could be. One table comes from one version of the code and refuses two.
Paired tests per case against the compaction arm, intervals by a bootstrap over histories, since the 15 cases of a
history share it (`studies/memory_probe/analyze.py`). In the long run the unit is the session, each compared on the tasks both arms reached, since the tasks of one conversation are not independent; counts over tasks are description only (`studies/long_run/analyze.py`). Sixteen rounds are not sixteen equal rounds: a history with one very long turn
leaves rounds that fold almost nothing, so effective rounds are reported beside them. Every clean result is reported, whichever way it falls.

**The long run's analysis, fixed on 2026-11-08 before any session on the final code had run.** The primary comparison is each arm
against compaction alone, for each planted rule and each summary prompt apart, with the session as the unit: a session counts for
the arm when the arm obeyed the rule at a higher rate than compaction alone over the tasks both reached after the rule was told,
against it when lower, and the two-sided sign test is over those sessions (`studies/long_run/analyze.py`). Such a test cannot fall
under .05 with fewer than six sessions, even if every one agrees (five of five gives .0625), so a summary prompt with fewer than six
valid sessions is reported as description only, beside the rule-loss matrix, which carries the tests. Secondary, and description
only: obedience by compactions so far (0, 1 to 2, 3 to 5, 6 to 10, 11 to 20, 21 to 50, 51 to 100, over 100), the commands at
work on an earlier task, and looks, finds and uses for the memory arms. How many sessions run is set by GPU time and not by any
result: the session filler sizes each session to end by a fixed time and starts the next only if the last ended cleanly and not far
shorter than asked. The paper reports every valid session that ended by 6 am ET on Mon 2026-11-16, and no session is stopped,
dropped or added because of what it showed.

An independent review of this harness on 2026-11-06 found that the search also held the system prompt, that what the harness delivered had lost each hit's date and
replacement mark, and that the long run's message for a reply with no command was not mini-swe-agent's. All were fixed, and the runs before that commit are not reported.

A later review the same day found, among others, that the note-taker was shown the system prompt, that a rule list could pass the budget, and that "found" had been read from less than what was delivered, and the same night all nine long sessions on one node were killed together, most likely by a command of one agent, which had just listed their process numbers, since a machine then shared the node's process list. All of that is fixed as written above. A fourth review found, among others, that a session stopped by its job left no record, that a command's output could fill a node's memory, and that a table could pool two readers or two versions of the code. Those are fixed too: a session now writes its record when stopped, one that died without a record is still counted, each table holds one reader, one summary prompt and one version of the code, and rates by rule and by what was held are over readable answers. The memory runs and the long sessions before these fixes are not reported, and no earlier number is reused: every cell and session is run again on the frozen code.

## Record

Every case keeps each tool call whole with its output, the final answer, the steps, the token counts and a ledger entry
naming the code that ran, in the rule-loss runs and the long sessions alike: a Taste row refuses to count unless its process imported `taste.memstore` and made the calls
it reports. Zero searches are allowed only where the agent chose, and that rate is what "looks" reports.
