"""Draw the README's architecture figure, in a light and a dark version.

    python docs/img/architecture.py        # writes architecture-{light,dark}.svg beside it

Standard library only. The numbers on the figure are the steps of "How a goal
runs" in the README; change one and change the other.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1126, 884
# Three columns: what goes in and the model on the left, the system in the
# centre, what comes out and the task environment on the right.
LEFT, LEFT_W = 24, 190
MID, MID_W = 254, 620
RIGHT, RIGHT_W = 910, 192

SANS = "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

LIGHT = {
    "page": "#ffffff", "frame": "#e2e8f0", "ink": "#0f172a", "body": "#475569", "faint": "#94a3b8",
    "card": "#ffffff", "line": "#334155", "badge": "#0f172a", "badge_ink": "#ffffff",
    "brain": ("#eef2ff", "#6366f1"), "worker": ("#ecfdf5", "#10b981"), "monitor": ("#fffbeb", "#f59e0b"),
    "memory": ("#f8fafc", "#94a3b8"), "world": ("#f0f9ff", "#0ea5e9"), "model": ("#faf5ff", "#a855f7"),
    "edge": ("#f8fafc", "#cbd5e1"), "bad": "#ef4444", "shadow": "0.10",
}
DARK = {
    "page": "#0d1117", "frame": "#30363d", "ink": "#f0f6fc", "body": "#b1bac4", "faint": "#6e7681",
    "card": "#161b22", "line": "#c9d1d9", "badge": "#f0f6fc", "badge_ink": "#0d1117",
    "brain": ("#1b1f3a", "#818cf8"), "worker": ("#0f2a22", "#34d399"), "monitor": ("#2d2208", "#fbbf24"),
    "memory": ("#11161d", "#6e7681"), "world": ("#0c2333", "#38bdf8"), "model": ("#251535", "#c084fc"),
    "edge": ("#161b22", "#30363d"), "bad": "#f87171", "shadow": "0.45",
}


class Figure:
    def __init__(self, palette: dict) -> None:
        self.p, self.out, self.heads = palette, [], {}

    # ---- primitives -------------------------------------------------------
    def add(self, markup: str) -> None:
        self.out.append(markup)

    def head(self, color: str) -> str:
        """The id of an arrowhead in this colour (`context-stroke` is not drawn everywhere)."""
        return self.heads.setdefault(color, f"head{len(self.heads)}")

    def box(self, x, y, w, h, fill, stroke, *, r=14, sw=1.5, shadow=False) -> None:
        extra = ' filter="url(#shadow)"' if shadow else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{extra}/>')

    def text(self, x, y, value, *, size=13, weight=400, fill=None, anchor="start", family=SANS,
             spacing=None) -> None:
        fill = fill or self.p["body"]
        extra = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{escape(value)}</text>')

    def lines(self, x, y, values, *, size=12.6, gap=18, **kwargs) -> None:
        for index, value in enumerate(values):
            self.text(x, y + index * gap, value, size=size, **kwargs)

    def path(self, points, *, color=None, dash=None, arrow=True, width=1.6) -> None:
        color = color or self.p["line"]
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' marker-end="url(#{self.head(color)})"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{extra}/>')

    def badge(self, x, y, number) -> None:
        self.add(f'<circle cx="{x}" cy="{y}" r="11" fill="{self.p["badge"]}"/>')
        self.text(x, y + 4.5, str(number), size=12.5, weight=700, fill=self.p["badge_ink"], anchor="middle")

    def pill(self, right, y, label, stroke) -> None:
        width = 18 + len(label) * 7.3
        self.add(f'<rect x="{right - width}" y="{y - 13}" width="{width}" height="19" rx="9.5" fill="none" '
                 f'stroke="{stroke}" stroke-width="1.2"/>')
        self.text(right - width / 2, y + 0.5, label, size=10.5, weight=600, fill=stroke, anchor="middle",
                  spacing="0.4")

    def dot(self, x, y, color, *, state="plain") -> None:
        p = self.p
        if state == "failed":
            self.add(f'<circle cx="{x}" cy="{y}" r="7" fill="{p["card"]}" stroke="{p["bad"]}" stroke-width="2"/>')
            self.add(f'<path d="M{x - 3},{y - 3} L{x + 3},{y + 3} M{x + 3},{y - 3} L{x - 3},{y + 3}" '
                     f'stroke="{p["bad"]}" stroke-width="1.8" stroke-linecap="round"/>')
            return
        fill = color if state != "hollow" else p["card"]
        self.add(f'<circle cx="{x}" cy="{y}" r="7" fill="{fill}" stroke="{color}" stroke-width="2"/>')
        if state == "certified":
            self.add(f'<path d="M{x - 3.2},{y + 0.2} L{x - 0.8},{y + 2.8} L{x + 3.4},{y - 2.6}" fill="none" '
                     f'stroke="{p["card"]}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>')

    # ---- composite parts --------------------------------------------------
    def card(self, x, y, w, h, kind, title, body) -> None:
        fill, stroke = self.p[kind]
        self.box(x, y, w, h, fill, stroke, shadow=True)
        self.text(x + 18, y + 30, title, size=16, weight=700, fill=self.p["ink"])
        self.lines(x + 18, y + 54, body)

    def inner(self, x, y, w, h, stroke, title, body, *, tag=None) -> None:
        self.box(x, y, w, h, self.p["card"], stroke, r=10, sw=1.2)
        self.text(x + 14, y + 24, title, size=14.5, weight=700, fill=self.p["ink"])
        if tag:
            self.pill(x + w - 10, y + 21, tag, stroke)
        self.lines(x + 14, y + 45, body, size=12.4, gap=17)

    def worker(self, x, y, *, numbered) -> None:
        p = self.p
        w, h = 295, 250
        fill, stroke = p["worker"]
        self.box(x, y, w, h, fill, stroke, shadow=True)
        self.text(x + 18, y + 30, "Worker", size=16, weight=700, fill=p["ink"])
        self.pill(x + w - 14, y + 26, "OWN PROCESS", stroke)
        self.box(x + 16, y + 44, w - 32, 76, p["card"], stroke, r=10, sw=1.2)
        self.text(x + 30, y + 66, "contract in, claim out", size=13.5, weight=700, fill=p["ink"])
        self.lines(x + 30, y + 86, ["model call, tool call, result, again", "on its own memory branch"],
                   size=12.4, gap=17)
        mfill, mstroke = p["monitor"]
        self.box(x + 16, y + 148, w - 32, 88, mfill, mstroke, r=10, sw=1.2)
        self.text(x + 30, y + 170, "Monitor", size=14.5, weight=700, fill=p["ink"])
        self.pill(x + w - 26, y + 167, "OBSERVE-ONLY", mstroke)
        self.lines(x + 30, y + 189, ["reads the transcript as it grows", "sends verdicts back to the worker",
                                     "certifies the end state, or refuses"], size=12.4, gap=16)
        # The monitor's verdicts go back up into the worker's loop.
        self.path([(x + 66, y + 148), (x + 66, y + 124)], color=mstroke)
        if numbered:
            self.badge(x + 92, y + 134, 5)
            self.text(x + 110, y + 138, "verdicts", size=11.5)


def draw(palette: dict) -> str:
    f = Figure(palette)
    p = palette

    f.add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
          f'role="img" aria-labelledby="t d">')
    f.add('<title id="t">Taste architecture</title>')
    f.add('<desc id="d">A goal enters the central brain, whose planner writes one contract per worker. '
          'The supervisor starts each worker as its own process on its own memory branch, with an '
          'independent monitor beside it. Workers act on the task environment through a terminal broker. '
          'Certified results are delivered into an integration branch, and the brain replans or replies. '
          'Everything is recorded in one git repository.</desc>')
    f.box(6, 6, W - 12, H - 12, p["page"], p["frame"], r=20, sw=1.2)

    # ---- row 1: the goal, the central brain, the result ---------------------
    f.text(LEFT, 40, "CONTROL", size=11, weight=700, fill=p["faint"], spacing="1.6")
    f.card(LEFT, 84, LEFT_W, 124, "edge", "Goal", ["a task in plain words", "success criteria", "a budget and a deadline"])
    f.card(RIGHT, 84, RIGHT_W, 124, "edge", "Result", ["the final reply", "certified outputs", "the complete record"])

    bfill, bstroke = p["brain"]
    f.box(MID, 48, MID_W, 196, bfill, bstroke, shadow=True)
    f.text(MID + 20, 78, "Central brain", size=17, weight=700, fill=p["ink"])
    f.text(MID + 136, 78, "plans, supervises, and decides when the goal is met", size=12.6)
    planner, runtime, supervisor = MID + 20, MID + 220, MID + 420
    f.inner(planner, 96, 180, 128, bstroke, "Planner",
            ["one contract per worker", "judges every criterion", "writes the final reply", "reads only the record"],
            tag="LLM")
    f.inner(runtime, 96, 180, 128, bstroke, "Runtime",
            ["the durable cycle:", "collect and deliver,", "replan or finish;", "budgets, deadlines"])
    f.inner(supervisor, 96, 180, 128, bstroke, "Supervisor",
            ["starts, stops and reaps", "worker processes;", "enforces deadlines;", "no process escapes"])
    f.path([(planner + 180, 176), (runtime - 2, 176)])
    f.path([(runtime + 180, 176), (supervisor - 2, 176)])
    f.badge(planner + 190, 152, 2)
    f.path([(LEFT + LEFT_W, 146), (MID - 2, 146)])
    f.badge((LEFT + LEFT_W + MID) / 2, 126, 1)
    f.path([(MID + MID_W, 146), (RIGHT - 2, 146)])
    f.badge((MID + MID_W + RIGHT) / 2, 126, 7)

    # ---- row 2: model API, workers with their monitors, the task environment
    f.text(LEFT, 300, "EXECUTION", size=11, weight=700, fill=p["faint"], spacing="1.6")
    f.card(LEFT, 356, LEFT_W, 148, "model", "Model API",
           ["used by the planner,", "workers and monitors.", "Each call is recorded,", "priced and budgeted."])
    first, second = MID, MID + 325
    f.worker(first, 304, numbered=True)
    f.worker(second, 304, numbered=False)
    wfill, wstroke = p["world"]
    f.box(RIGHT, 304, RIGHT_W, 250, wfill, wstroke, shadow=True)
    f.text(RIGHT + 16, 334, "Task environment", size=16, weight=700, fill=p["ink"])
    f.text(RIGHT + 16, 354, "a container or a repo", size=12.6)
    f.inner(RIGHT + 14, 372, RIGHT_W - 28, 118, wstroke, "Terminal broker",
            ["one command at a time", "each one on record", "a timeout ends only", "that command"])
    f.lines(RIGHT + 16, 513, ["Workers share it.", "The brain stays out."], size=12.4, gap=17)

    # the supervisor starts the workers (3); a worker's report returns to the runtime (6)
    down = supervisor + 90
    f.path([(down, 224), (down, 302)])
    f.path([(down, 278), (first + 216, 278), (first + 216, 302)])
    f.badge(down, 258, 3)
    f.text(down + 18, 266, "start", size=11.5)
    up = runtime + 90
    f.path([(first + 76, 304), (first + 76, 262), (up, 262), (up, 226)])
    f.badge(first + 146, 262, 6)
    f.text(first + 164, 256, "report", size=11.5)
    # model calls on the left, commands on the right
    f.path([(MID - 2, 430), (LEFT + LEFT_W + 2, 430)], color=p["model"][1])
    f.path([(MID + MID_W, 430), (RIGHT - 2, 430)], color=wstroke)
    f.badge((MID + MID_W + RIGHT) / 2, 410, 4)

    # ---- row 3: memory ------------------------------------------------------
    mfill, mstroke = p["memory"]
    f.box(LEFT, 604, W - 2 * LEFT, 256, mfill, mstroke, shadow=True)
    f.text(LEFT + 20, 636, "Memory", size=17, weight=700, fill=p["ink"])
    f.text(LEFT + 96, 636, "one git repository: a branch per actor, a commit per checkpoint, rollback is an append",
           size=12.6)
    green = p["worker"][1]
    for x, state, label in ((756, "hollow", "checkpoint"), (860, "certified", "certified"),
                            (950, "failed", "failed, and kept")):
        f.dot(x, 632, green, state=state)
        f.text(x + 14, 636, label, size=11.5)
    for y, name, caption in ((688, "control", "plans, decisions, model receipts"),
                             (730, "integration", "certified results only"),
                             (772, "worker-1", "files, transcript, verdicts"),
                             (814, "worker-2", "a failed attempt stays readable")):
        f.text(LEFT + 20, y - 2, name, size=13, weight=700, fill=p["ink"], family=MONO)
        f.text(LEFT + 20, y + 14, caption, size=11.2)
    f.path([(250, 688), (1082, 688)], color=bstroke, arrow=False, width=2)
    f.path([(250, 730), (1082, 730)], color=p["line"], arrow=False, width=2)
    f.path([(384, 772), (624, 772)], color=green, arrow=False, width=2)
    f.path([(734, 814), (1008, 814)], color=green, arrow=False, width=2)
    # control: the brain's own record of the goal
    for x, label in ((262, "goal"), (332, "plan 1"), (402, "start"), (642, "collect"), (698, "plan 2"),
                     (754, "start"), (1022, "collect"), (1068, "reply")):
        f.dot(x, 688, bstroke)
        f.text(x, 672, label, size=10.8, anchor="middle")
    # integration: the base state, then one commit per delivery
    f.dot(262, 730, p["line"], state="hollow")
    f.dot(612, 730, p["line"])
    f.dot(992, 730, p["line"])
    # worker 1: checkpoints, certified, delivered
    f.path([(402, 695), (402, 765)], color=green, dash="3 4", width=1.3)
    for x in (402, 472, 542):
        f.dot(x, 772, green, state="hollow")
    f.dot(612, 772, green, state="certified")
    f.path([(612, 763), (612, 739)], color=p["line"])
    f.text(626, 755, "deliver", size=11.2)
    # worker 2: a failed attempt, a rollback, then certified and delivered
    f.path([(754, 695), (754, 807)], color=green, dash="3 4", width=1.3)
    f.dot(754, 814, green, state="hollow")
    f.dot(816, 814, green, state="hollow")
    f.dot(878, 814, green, state="failed")
    f.add(f'<path d="M878,805 C878,780 816,780 816,804" fill="none" stroke="{p["bad"]}" stroke-width="1.6" '
          f'stroke-dasharray="4 3" marker-end="url(#{f.head(p["bad"])})"/>')
    f.text(847, 776, "rollback", size=11.2, fill=p["bad"], anchor="middle")
    f.dot(936, 814, green, state="hollow")
    f.dot(992, 814, green, state="certified")
    f.path([(992, 805), (992, 739)], color=p["line"])
    f.text(1006, 776, "deliver", size=11.2)

    # workers write their checkpoints into memory
    for x in (first + 147, second + 147):
        f.path([(x, 554), (x, 602)], color=green, dash="4 4")
        f.text(x + 14, 583, "checkpoints", size=11.5)

    heads = "".join(
        f'<marker id="{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7.5" markerHeight="7.5" '
        f'orient="auto-start-reverse"><path d="M0,0.8 L9,5 L0,9.2 z" fill="{color}"/></marker>'
        for color, name in f.heads.items())
    defs = (f'<defs>{heads}<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">'
            f'<feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#000000" '
            f'flood-opacity="{p["shadow"]}"/></filter></defs>')
    f.out.insert(3, defs)  # after <svg>, <title> and <desc>
    f.add("</svg>")
    return "\n".join(f.out) + "\n"


def main() -> None:
    here = Path(__file__).resolve().parent
    for name, palette in (("light", LIGHT), ("dark", DARK)):
        (here / f"architecture-{name}.svg").write_text(draw(palette), encoding="utf-8")


if __name__ == "__main__":
    main()
