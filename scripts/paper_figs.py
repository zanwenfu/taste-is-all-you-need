import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np

# ---- typography and style: match the paper (Times) and keep the ink restrained
for f in ("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
          "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
          "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf"):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Nimbus Roman", "font.size": 8,
    "axes.linewidth": 0.5, "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.color": "#444444", "ytick.color": "#444444", "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
ACCENT, ACCENT_LIGHT, SIGNAL, GREY, GREY_LIGHT = "#2f5f8f", "#7fa3c9", "#b23a3a", "#8a8a8a", "#cfcfcf"

def ev(root):
    out = {}
    for f in glob.glob(f"{root}/ledger/evidence/*.json") + glob.glob(f"{root}/rescored/evidence/*.json"):
        d = json.load(open(f)); out[d["instance_id"]] = d
    return out
a3 = ev("/root/pilot40d")

def label_bars(ax, bars, fmt, dy, color="#222222", size=7):
    for b, v in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v), ha="center", va="bottom", fontsize=size, color=color)

# ---- Figure 2 (paper): events on the timeline vs failures visible in the final state, per bearing run
rows = []
for name, E in (("calib.", ev("/root/oai10b")), ("x40", a3)):
    for inst, d in sorted(E.items()):
        n = d.get("contamination_events_declared", 0)
        if not n: continue
        g = d.get("grade") or {}
        p, t = (int(x) for x in g["pass_to_pass"].split("/"))
        fs = max(0, t - p - len(d.get("never_passed", [])))
        rows.append((inst.split("__")[1][:18] + (" (calib.)" if name == "calib." else ""), n, fs))
fig, ax = plt.subplots(figsize=(5.4, 1.55))
x = np.arange(len(rows)); w = 0.38
b1 = ax.bar(x - w / 2, [r[1] for r in rows], w, color=SIGNAL, label="events on the timeline", linewidth=0)
b2 = ax.bar(x + w / 2, [r[2] for r in rows], w, color=GREY_LIGHT, label="failures visible in the final patch", linewidth=0)
label_bars(ax, zip(b1, [r[1] for r in rows]), "{:d}", 0.8, color=SIGNAL)
label_bars(ax, zip(b2, [r[2] for r in rows]), "{:d}", 0.8, color="#555555")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=7, rotation=20, ha="right")
ax.set_ylabel("regression events", fontsize=8)
ax.set_ylim(0, max(r[1] for r in rows) * 1.22)
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
ax.legend(loc="upper left", ncol=1, handlelength=1.0, handleheight=0.8, borderaxespad=0.2)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.5); ax.set_axisbelow(True)
plt.tight_layout(pad=0.4)
plt.savefig("/root/taste/paper/latex/fig_undercount.pdf"); plt.savefig("/root/taste/paper/latex/fig_undercount.png", dpi=200)

# ---- Figure 3 (paper): the five recovery arms
def arm_stats(root):
    E = {}
    for f in glob.glob(f"{root}/ledger/evidence/*.json") + glob.glob(f"{root}/rescored/evidence/*.json"):
        d = json.load(open(f)); E[d["instance_id"]] = d
    graded = [d for d in E.values() if d.get("resolved") is not None]
    res = sum(1 for d in graded if d["resolved"]) / 40.0
    contam = 0
    for d in graded:
        g = d.get("grade") or {}
        if g.get("pass_to_pass"):
            pa, to = (int(v) for v in g["pass_to_pass"].split("/"))
            if to - pa - len(d.get("never_passed", [])) > 0: contam += 1
    spend = 0.0
    for f in glob.glob(f"{root}/ledger/*.json"):
        d = json.load(open(f))
        if isinstance(d, dict) and d.get("task") and d.get("billed_usd") is not None: spend += d["billed_usd"]
    return res, contam, spend, len(graded)
arm_roots = [("rollback", "/root/pilot40d"), ("gated", "/root/contrast40_A3reg"), ("split", "/root/contrast40_A3reg2"),
             ("repair", "/root/contrast40_A2"), ("none", "/root/contrast40_A0")]
arm_roots = [(n, r) for n, r in arm_roots if os.path.isdir(f"{r}/ledger")]
stats = [arm_stats(r) for _, r in arm_roots]
arms = [n for n, _ in arm_roots]
resolve = [s[0] for s in stats]; contam = [s[1] for s in stats]; spend = [s[2] for s in stats]
colors = {"rollback": GREY, "gated": ACCENT, "split": ACCENT_LIGHT, "repair": GREY_LIGHT, "none": GREY_LIGHT}
fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.45))
panels = [(resolve, "resolve rate (of 40)", "{:.0%}"), (contam, "cells with a contaminated final tree", "{:d}"), (spend, "total sweep cost (USD)", "${:.2f}")]
for ax, (vals, title, fmt) in zip(axes, panels):
    bars = ax.bar(range(len(arms)), vals, color=[colors[a] for a in arms], width=0.66, linewidth=0)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, fontsize=7)
    ax.set_title(title, fontsize=7.6, loc="left", pad=3)
    label_bars(ax, zip(bars, vals), fmt, (max(vals) or 1) * 0.02, size=6.8)
    ax.set_ylim(0, (max(vals) or 1) * 1.25)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5); ax.set_axisbelow(True)
axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axes[2].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:.0f}"))
plt.tight_layout(pad=0.4, w_pad=1.2)
plt.savefig("/root/taste/paper/latex/fig_contrast.pdf"); plt.savefig("/root/taste/paper/latex/fig_contrast.png", dpi=200)
print("contrast arms:", arms, "| graded:", [s[3] for s in stats], "| undercount rows:", len(rows))
