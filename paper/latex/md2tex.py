"""Markdown (paper/workshop.md) -> NeurIPS LaTeX. Deterministic, re-runnable.

Kept deliberately narrow: the paper uses headers, bold/italic, inline code,
pipe tables, blockquotes (the 'what to steal' box), and bracket citations.
Anything else passes through so the diff is inspectable.
"""
from __future__ import annotations

import re
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
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=black}
\usepackage{array}
\usepackage{pifont}
\usepackage{graphicx}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
\title{TITLE_PLACEHOLDER}
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
            e = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", lambda m: "\\cite{" + ",".join("ref" + x.strip() for x in m.group(1).split(",")) + "}", e)
            out.append(e)
    return "".join(out)

def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], [r for r in cells[2:] if r]
    ncol = len(header)
    spec = "l" + "p{0.42\\linewidth}" + "p{0.36\\linewidth}" + "c" * (ncol - 3) if ncol >= 4 and header[0] == "#" else "l" * ncol
    lines = [r"\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{" + spec + "}", r"\toprule",
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
    title = "Untitled"
    while i < len(lines) and (lines[i].startswith("# ") or lines[i].startswith("*") or lines[i].strip() in ("", "---")):
        if lines[i].startswith("# "):
            title = lines[i][2:].strip()
        i += 1
    title_tex = inline(title).replace(": ", r":\\", 1)
    refs = []
    in_appendix = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### References"):
            # Collect references until the next section heading and keep
            # going: the appendix follows the references in the markdown,
            # and a `break` here silently dropped it from the submission.
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                m = re.match(r"\[(\d+)\]\s+(.*)", lines[i].strip())
                if m: refs.append((m.group(1), m.group(2)))
                i += 1
            continue
        if ln.startswith("## Abstract"):
            i += 1; buf = []
            while i < len(lines) and not lines[i].startswith("## "):
                if lines[i].strip() != "---": buf.append(lines[i])
                i += 1
            out.append(r"\begin{abstract}" + "\n" + inline("\n".join(buf).strip()) + "\n" + r"\end{abstract}")
            continue
        if ln.startswith("## "):
            title = re.sub(r"^\d+\.\s*", "", ln[3:]).strip()
            if title.lower().startswith("appendix"):
                # "Appendix A: Title" -> \section{Title}; LaTeX letters it.
                clean = re.sub(r"^Appendix\s+[A-Z][:.]?\s*", "", title, flags=re.I)
                marker = "" if in_appendix else r"\appendix" + "\n"
                in_appendix = True
                out.append(marker + r"\section{" + inline(clean) + "}")
            else:
                out.append(r"\section{" + inline(title) + "}")
            i += 1; continue
        if ln.startswith("### "):
            title = re.sub(r"^\d+\.\d+\s*", "", ln[4:]).strip()
            if in_appendix:
                title = re.sub(r"^[A-Z]\.\d+\s*", "", title)
            out.append(r"\subsection{" + inline(title) + "}"); i += 1; continue
        m_img = re.match(r"!\[(.*?)\]\((.*?)\)(?:\{width=([0-9.]+)\})?\s*$", ln.strip())
        if m_img:
            cap, path = m_img.group(1), m_img.group(2)
            width = m_img.group(3) or "1"
            label = re.sub(r"[^a-z0-9]+", "-", Path(path).stem.lower())
            out.append(r"\begin{figure}[t]\centering\includegraphics[width=" + width + r"\linewidth]{" + path + "}"
                       + r"\caption{" + inline(cap) + "}\label{fig:" + label + "}\end{figure}")
            i += 1; continue
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
        if re.match(r"^\d+\.\s", ln):
            buf = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i]):
                buf.append(r"\item " + inline(re.sub(r"^\d+\.\s+", "", lines[i]).strip())); i += 1
            out.append(r"\begin{enumerate}" + "\n" + "\n".join(buf) + "\n" + r"\end{enumerate}"); continue
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
    bib = [r"\begin{thebibliography}{99}"] + [r"\bibitem{ref" + n + "} " + inline(t) for n, t in refs] + [r"\end{thebibliography}"]
    labels = re.findall(r"\\label\{(fig:[^}]+)\}", "\n".join(out))
    def figref(m):
        n = int(m.group(1))
        return ("Figure~\\ref{" + labels[n-1] + "}") if 0 < n <= len(labels) else m.group(0)
    body = "\n\n".join(out)
    body = re.sub(r"Figure (\d)(?!\d)", figref, body)
    pre = PREAMBLE.replace("TITLE_PLACEHOLDER", title_tex)
    if "\\appendix" in body:
        head, _, tail = body.partition("\\appendix")
        body = head + "\n".join(bib) + "\n\n\\appendix" + tail
        return pre + body + "\n\\end{document}\n"
    return pre + body + "\n\n" + "\n".join(bib) + "\n\\end{document}\n"

if __name__ == "__main__":
    OUT.write_text(convert(SRC.read_text()))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
