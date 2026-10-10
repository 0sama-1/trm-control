#!/usr/bin/env python3
"""Build the Phase 4 documents (copied from phase3/report/build.py; four glyph mappings added): Markdown -> LaTeX -> PDF (tectonic).

  python3 build.py                 # every *.md in this folder
  python3 build.py 2_phase3_paper.md

Each .md stays the single source; the .tex and .pdf are generated next to it.
Handles the Markdown subset these documents use: headings, paragraphs, pipe
tables, figures, bullet / numbered lists (two levels), blockquotes, fenced code,
and inline bold / italic / code / links. Figures reference figures/*.png; the
PDF build embeds the vector .pdf twin when it exists. Stdlib only.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TECTONIC = Path.home() / ".local" / "bin" / "tectonic"

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=0.95in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,longtable,array}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\usepackage{caption}
\setlist{itemsep=2pt,topsep=4pt}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{5pt}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.15}
\title{%(title)s}
\author{%(author)s}
\date{%(date)s}
\begin{document}
\maketitle
"""

UNICODE = {
    "≥": r"$\geq$", "≤": r"$\leq$", "≈": r"$\approx$", "×": r"$\times$", "−": r"$-$", "→": r"$\rightarrow$",
    "←": r"$\leftarrow$", "±": r"$\pm$", "σ": r"$\sigma$", "Δ": r"$\Delta$", "∈": r"$\in$", "∪": r"$\cup$",
    "²": r"$^2$", "·": r"$\cdot$", "≠": r"$\neq$", "∞": r"$\infty$", "⇒": r"$\Rightarrow$", "√": r"$\surd$",
    "μ": r"$\mu$", "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "τ": r"$\tau$", "θ": r"$\theta$",
    "ω": r"$\omega$", "ℝ": r"$\mathbb{R}$", "≫": r"$\gg$", "≪": r"$\ll$", "∑": r"$\sum$", "′": r"$'$",
    "—": "---", "–": "--", "…": r"\ldots{}", "“": "``", "”": "''", "‘": "`", "’": "'", "✓": r"$\checkmark$",
    "§": r"\S{}", "°": r"$^\circ$", "½": r"$\frac{1}{2}$", "⁻": r"$^-$", "¹": r"$^1$", "₀": r"$_0$",
    # added for the Phase 4 documents
    "ρ": r"$\rho$", "δ": r"$\delta$", "∂": r"$\partial$", "‖": r"$\|$",
}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(text: str) -> str:
    out = []
    for ch in text:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in UNICODE:
            out.append(UNICODE[ch])
        else:
            out.append(ch)
    return "".join(out)


def inline(text: str) -> str:
    """Inline markdown -> LaTeX. Code spans and links are protected before escaping."""
    slots: list[str] = []

    def keep(s: str) -> str:
        slots.append(s)
        return f"\x00{len(slots) - 1}\x00"

    def code(c: str) -> str:
        # allow line breaks inside long paths / identifiers
        e = esc(c)
        if len(c) < 14:
            return r"\texttt{" + e + "}"
        for a, b in (("/", "/\\allowbreak{}"), ("\\_", "\\_\\allowbreak{}"), (",", ",\\allowbreak{}"), (".", ".\\allowbreak{}")):
            e = e.replace(a, b)
        return r"\texttt{" + e + "}"

    text = re.sub(r"`([^`]+)`", lambda m: keep(code(m.group(1))), text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: keep(r"\href{" + m.group(2).replace("%", r"\%").replace("#", r"\#") + "}{" + inline(m.group(1)) + "}"),
                  text)
    text = re.sub(r"(https?://[^\s)]+)", lambda m: keep(r"\url{" + m.group(1) + "}"), text)
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", text)
    return re.sub("\x00(\\d+)\x00", lambda m: slots[int(m.group(1))], text)


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, spec, body = cells[0], cells[1], cells[2:]
    ncol = len(header)
    align = []
    for s in spec + [""] * (ncol - len(spec)):
        align.append("r" if s.endswith(":") and not s.startswith(":") else "c" if s.startswith(":") and s.endswith(":") else "l")
    lens = [max(len(re.sub(r"[*`]", "", (r[i] if i < len(r) else ""))) for r in cells[:1] + body) for i in range(ncol)]
    total = sum(lens)
    if total > 70 or ncol >= 5:
        # wrap: proportional paragraph columns, numeric columns kept right-aligned; leave room for the
        # inter-column padding (2 * tabcolsep per column) so the table fits the text width
        width = [max(0.06, min(0.55, l / total)) for l in lens]
        # never narrower than the column's longest unbreakable token (about 0.0115 linewidth per char at \small)
        longest = [max((len(t) for r in cells[:1] + body if i < len(r)
                        for t in re.sub(r"[*`]", "", r[i]).replace("(", " ").split()), default=4) for i in range(ncol)]
        avail = 0.99 - 0.028 * ncol
        width = [w * avail / sum(width) for w in width]
        floor = [min(0.45, 0.0115 * t) for t in longest]
        for _ in range(20):
            short = [i for i in range(ncol) if width[i] < floor[i]]
            if not short:
                break
            deficit = sum(floor[i] - width[i] for i in short)
            for i in short:
                width[i] = floor[i]
            donors = [i for i in range(ncol) if i not in short and width[i] > floor[i]]
            spare = sum(width[i] - floor[i] for i in donors) or 1.0
            for i in donors:
                width[i] -= deficit * (width[i] - floor[i]) / spare
        scale = 1.0
        cols = []
        for a, w in zip(align, width):
            base = r"p{%.3f\linewidth}" % (w * scale)
            cols.append((r">{\raggedleft\arraybackslash}" if a == "r" else r">{\raggedright\arraybackslash}") + base)
        colspec = "".join(cols)
    else:
        colspec = "".join(align)
    size = r"\small" if total > 60 else ""
    # Short tables (all of these documents' tables) as unbreakable tabulars: longtable interacts badly with
    # pending floats and can push text past the bottom margin. Only very long tables use longtable.
    long = len(body) > 30
    env = "longtable" if long else "tabular"
    out = [(r"{" + size if long else r"\begin{center}" + size) + r"\begin{" + env + "}{" + colspec + "}", r"\toprule",
           " & ".join(inline(h) for h in header) + r" \\", r"\midrule"] + ([r"\endhead"] if long else [])
    for r in body:
        r = (r + [""] * ncol)[:ncol]
        out.append(" & ".join(inline(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{" + env + "}" + ("}" if long else r"\end{center}")]
    return "\n".join(out)


def figure(alt: str, src: str) -> str:
    p = Path(src)
    pdf = (HERE / p).with_suffix(".pdf")
    use = str(p.with_suffix(".pdf")) if pdf.exists() else src
    return (r"\begin{figure}[htbp]\centering" + "\n" + r"\includegraphics[width=\linewidth,height=0.42\textheight,keepaspectratio]{"
            + use + "}\n" + r"\caption{" + inline(alt) + "}\n" + r"\end{figure}")


def convert(md: str) -> tuple[str, dict]:
    lines = md.splitlines()
    meta = {"title": "", "author": "", "date": ""}
    body: list[str] = []
    i = 0
    # title block: "# Title" then optional italic lines
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].startswith("# "):
        meta["title"] = inline(lines[i][2:].strip())
        i += 1
        sub = []
        while i < len(lines) and (not lines[i].strip() or lines[i].startswith("*")):
            if lines[i].strip():
                sub.append(inline(lines[i].strip().strip("*")))
            i += 1
        meta["date"] = r"\\[2pt]".join(sub)
    in_abstract = False
    toc = "<!-- toc -->" in md
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s == "<!-- toc -->":
            body.append(r"\tableofcontents" + "\n" + r"\clearpage")
            i += 1
            continue
        if s.startswith("```"):
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            body.append(r"{\small\begin{verbatim}" + "\n" + "\n".join(code) + "\n" + r"\end{verbatim}}")
            i = j + 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            if in_abstract:
                body.append(r"\end{abstract}")
                in_abstract = False
            level, title = len(m.group(1)), m.group(2).strip()
            if level == 1:   # a part divider (the document title was consumed above)
                t = inline(title)
                body.append(r"\bigskip{\centering\Large\bfseries " + t + r"\par}\medskip"
                            + (f"\\addcontentsline{{toc}}{{section}}{{{t}}}" if toc else ""))
                i += 1
                continue
            if level == 2 and title.lower() == "abstract":
                body.append(r"\begin{abstract}")
                in_abstract = True
            else:
                cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
                t = inline(title)
                body.append(f"\\{cmd}*{{{t}}}" + (f"\\addcontentsline{{toc}}{{{cmd}}}{{{t}}}" if toc and level < 4 else ""))
            i += 1
            continue
        if s in ("---", "***"):
            body.append(r"\medskip")
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-{2,}", lines[i + 1].strip()):
            j = i
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j])
                j += 1
            body.append(table(rows))
            i = j
            continue
        m = re.match(r"^!\[(.*)\]\((.+)\)\s*$", s)
        if m:
            body.append(figure(m.group(1), m.group(2)))
            i += 1
            continue
        if s.startswith(">"):
            j = i
            q = []
            while j < len(lines) and lines[j].strip().startswith(">"):
                q.append(lines[j].strip()[1:].strip())
                j += 1
            body.append(r"\begin{quote}" + inline(" ".join(q)) + r"\end{quote}")
            i = j
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            j = i
            items: list[tuple[int, bool, str]] = []
            while j < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[j])
                if mm:
                    items.append((len(mm.group(1)) // 2, mm.group(2)[0].isdigit(), mm.group(3)))
                elif lines[j].strip() and lines[j].startswith("  ") and items:
                    lvl, num, t = items[-1]
                    items[-1] = (lvl, num, t + " " + lines[j].strip())
                else:
                    break
                j += 1
            out, stack = [], []
            for lvl, num, t in items:
                lvl = min(lvl, 1)
                env = "enumerate" if num else "itemize"
                while len(stack) > lvl + 1:
                    out.append(f"\\end{{{stack.pop()}}}")
                if len(stack) < lvl + 1:
                    stack.append(env)
                    out.append(f"\\begin{{{env}}}")
                out.append(r"\item " + inline(t))
            while stack:
                out.append(f"\\end{{{stack.pop()}}}")
            body.append("\n".join(out))
            i = j
            continue
        if not s:
            body.append("")
            i += 1
            continue
        j = i
        para = []
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#{1,4}\s|\||!\[|>|```|\s*([-*]|\d+\.)\s)", lines[j]):
            para.append(lines[j].strip())
            j += 1
        if not para:
            para, j = [s], i + 1
        body.append(inline(" ".join(para)))
        i = j
    if in_abstract:
        body.append(r"\end{abstract}")
    return "\n\n".join(body), meta


def build(md_path: Path) -> None:
    tex, meta = convert(md_path.read_text())
    doc = PREAMBLE % meta + tex + "\n\\end{document}\n"
    tex_path = md_path.with_suffix(".tex")
    tex_path.write_text(doc)
    r = subprocess.run([str(TECTONIC), "--keep-logs", tex_path.name], cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"tectonic failed on {tex_path.name}")
    warn = [l for l in (HERE / tex_path.with_suffix(".log").name).read_text(errors="replace").splitlines()
            if "Overfull \\hbox" in l and "pt too wide" in l and float(re.search(r"\(([\d.]+)pt", l).group(1)) > 20]
    print(f"built {md_path.with_suffix('.pdf').name}" + (f"  ({len(warn)} overfull boxes > 20pt)" if warn else ""))


if __name__ == "__main__":
    targets = [HERE / a for a in sys.argv[1:]] or sorted(HERE.glob("[0-9]*.md"))
    for t in targets:
        build(t)
