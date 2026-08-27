"""Markdown (paper/workshop.md) -> NeurIPS LaTeX. Deterministic, re-runnable.

Kept deliberately narrow: the paper uses headers, bold/italic, inline code,
pipe tables, blockquotes (the 'what to steal' box), and bracket citations.
Anything else passes through so the diff is inspectable.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "workshop.md"
OUT = Path(__file__).resolve().parent / "main.tex"

PREAMBLE = r"""\documentclass{article}
\PassOptionsToPackage{numbers,sort&compress}{natbib}
\usepackage{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{array}
\usepackage{pifont}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
\title{Twenty-Eight Ways to Measure Nothing:\\A Failure Catalogue from Instrumenting an Agent Harness}
\author{Anonymous Authors\\AgenticOS Workshop Submission}
\begin{document}
\maketitle
"""

def esc(t: str) -> str:
    # order matters: backslash first
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        t = t.replace(a, b)
    t = t.replace("λ̂", r"$\hat\lambda$").replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("×", r"$\times$").replace("→", r"$\rightarrow$").replace("—", "---").replace("–", "--").replace("≈", r"$\approx$")
    t = t.replace("✓", r"\cmark{}").replace("✗", r"\xmark{}").replace("§", r"\S")
    return t

def inline(t: str) -> str:
    # code spans first so their contents are escaped once
    parts = re.split(r"(`[^`]*`)", t)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(r"\texttt{" + esc(p[1:-1]) + "}")
        else:
            e = esc(p)
            e = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", e, flags=re.S)
            e = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", e, flags=re.S)
            e = re.sub(r"\[(\d+)\]", r"\\cite{ref\1}", e)
            out.append(e)
    return "".join(out)

def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], [r for r in cells[2:] if r]
    ncol = len(header)
    spec = "l" + "p{0.40\\linewidth}" + "p{0.36\\linewidth}" + "c" * (ncol - 3) if ncol >= 4 and header[0] == "#" else "l" * ncol
    lines = [r"\begin{center}\small\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(inline(h) for h in header) + r" \\", r"\midrule"]
    for r in body:
        r = r + [""] * (ncol - len(r))
        lines.append(" & ".join(inline(c) for c in r[:ncol]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}\end{center}"]
    return "\n".join(lines)

def convert(md: str) -> str:
    out, i = [], 0
    lines = md.splitlines()
    # drop the H1 title and the italic header line; title is in the preamble
    while i < len(lines) and (lines[i].startswith("# ") or lines[i].startswith("*Agentic") or lines[i].strip() in ("", "---")):
        i += 1
    refs = []
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### References"):
            i += 1
            while i < len(lines):
                m = re.match(r"\[(\d+)\]\s+(.*)", lines[i].strip())
                if m: refs.append((m.group(1), m.group(2)))
                i += 1
            break
        if ln.startswith("## Abstract"):
            i += 1; buf = []
            while i < len(lines) and not lines[i].startswith("## "):
                if lines[i].strip() != "---": buf.append(lines[i])
                i += 1
            out.append(r"\begin{abstract}" + "\n" + inline("\n".join(buf).strip()) + "\n" + r"\end{abstract}")
            continue
        if ln.startswith("## "):
            title = re.sub(r"^\d+\.\s*", "", ln[3:]).strip()
            out.append(r"\section{" + inline(title) + "}"); i += 1; continue
        if ln.startswith("### "):
            title = re.sub(r"^\d+\.\d+\s*", "", ln[4:]).strip()
            out.append(r"\subsection{" + inline(title) + "}"); i += 1; continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            out.append(table(rows)); continue
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i][1:].strip()); i += 1
            body = "\n".join(buf)
            chunks = re.split(r"^\s*\d+\.\s+", body, flags=re.M)
            title, items = chunks[0].strip(), [c.strip() for c in chunks[1:] if c.strip()]
            out.append(r"\begin{quote}\small " + inline(title) + r"\begin{enumerate}" + "\n"
                       + "\n".join(r"\item " + inline(it) for it in items) + "\n" + r"\end{enumerate}\end{quote}")
            continue
        if ln.strip() == "---":
            i += 1; continue
        if ln.startswith("- "):
            buf = []
            while i < len(lines) and lines[i].startswith("- "):
                buf.append(r"\item " + inline(lines[i][2:].strip())); i += 1
            out.append(r"\begin{itemize}" + "\n" + "\n".join(buf) + "\n" + r"\end{itemize}"); continue
        # paragraph
        buf = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", ">", "- ", "---")):
            buf.append(lines[i]); i += 1
        if buf:
            out.append(inline(" ".join(b.strip() for b in buf)))
        else:
            i += 1
    bib = [r"\begin{thebibliography}{9}"] + [r"\bibitem{ref" + n + "} " + inline(t) for n, t in refs] + [r"\end{thebibliography}"]
    return PREAMBLE + "\n\n".join(out) + "\n\n" + "\n".join(bib) + "\n\\end{document}\n"

if __name__ == "__main__":
    OUT.write_text(convert(SRC.read_text()))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
