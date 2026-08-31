import sys, pathlib
sys.path.insert(0, "/root/taste/scripts")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from undercount import load, base_test_id, suite_died, _failed_by_count

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 8,
    "axes.linewidth": 0.5, "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.color": "#444444", "ytick.color": "#444444", "xtick.labelsize": 6.9, "ytick.labelsize": 8.0,
    "legend.frameon": False, "legend.fontsize": 7.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})
SIGNAL, GREY = "#b23a3a", "#cfcfcf"

def rows_of(root, empty_set=()):
    """One entry per bearing run in ONE root: merging roots by instance id
    would let a zero-episode cell shadow a real one (pilot40d's flask)."""
    out = []
    for inst, d in sorted(load(pathlib.Path(root)).items()):
        broke = {base_test_id(e["probe"]) for e in (d.get("episodes") or []) if e.get("probe")}
        if not broke or suite_died(d):
            continue
        never = {base_test_id(t) for t in (d.get("never_passed") or [])}
        survived = {base_test_id(t) for t in (d.get("grade_failed") or [])} - never
        left = len(broke & survived)
        if not d.get("grade_failed") and _failed_by_count(d) > 0:
            left = _failed_by_count(d)
        out.append((inst.split("__")[-1], len(broke), left, inst in empty_set))
    return out

EMPTY_A3 = {"django__django-11276", "matplotlib__matplotlib-26291", "mwaskom__seaborn-3069", "pytest-dev__pytest-6197"}
scaffold = sum((rows_of(r) for r in
    ["/root/mswe40_sonnet2_s1", "/root/mswe40_sonnet2_s2", "/root/mswe40_sonnet2_s3",
     "/root/mswe40_gpt2_s1", "/root/mswe40_gpt2_s2", "/root/mswe40_gpt2_s3"]), [])
harness = rows_of("/root/pilot40d", EMPTY_A3) + [
    (lab + " (calib.)", b, l, h)
    for lab, b, l, h in rows_of("/root/oai10b", {"pallets__flask-5014"})
]

labels = [r[0] for r in scaffold + harness]
broken = [r[1] for r in scaffold + harness]
left   = [r[2] for r in scaffold + harness]
hatch  = [r[3] for r in scaffold + harness]
boundary = len(scaffold)

fig, ax = plt.subplots(figsize=(7.0, 1.95))
x = list(range(len(labels)))
bars = ax.bar(x, broken, 0.74, color=GREY, linewidth=0)
for bar, h in zip(bars, hatch):
    if h:
        bar.set_hatch("///"); bar.set_edgecolor("#9a9a9a"); bar.set_linewidth(0.4)
ax.bar(x, left, 0.74, color=SIGNAL, linewidth=0)
top = max(broken)
for i, (b, l) in enumerate(zip(broken, left)):
    ax.text(i, b + top * 0.02, str(b), ha="center", va="bottom", fontsize=7.0, color="#222222")
    if l:
        ax.text(i, l + top * 0.02, str(l), ha="center", va="bottom", fontsize=7.0, color=SIGNAL, fontweight="bold")
ax.axvline(boundary - 0.5, color="#888888", lw=0.6, ls=(0, (3, 3)))
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=42, ha="right")
ax.set_ylabel("test functions")
ax.set_ylim(0, top * 1.16)
ax.text(0.005, 0.94, "unmodified public scaffold", transform=ax.transAxes, fontsize=7.8, color="#222222")
ax.text((boundary + 0.25) / len(labels), 0.94, "our harness, rollback", transform=ax.transAxes, fontsize=7.8, color="#222222")
ax.legend(handles=[Patch(facecolor=GREY, label="broken during the run"),
                   Patch(facecolor=SIGNAL, label="still failing at grade time"),
                   Patch(facecolor="white", edgecolor="#9a9a9a", hatch="///", label="final tree never changed")],
          loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.11))
plt.savefig("/root/taste/paper/latex/fig_undercount.pdf", bbox_inches="tight")
plt.savefig("/root/taste/paper/latex/fig_undercount.png", dpi=200, bbox_inches="tight")
print("scaffold runs:", len(scaffold), "sum broken:", sum(r[1] for r in scaffold), "left:", sum(r[2] for r in scaffold))
print("harness runs:", len(harness), "sum broken:", sum(r[1] for r in harness), "left:", sum(r[2] for r in harness))
