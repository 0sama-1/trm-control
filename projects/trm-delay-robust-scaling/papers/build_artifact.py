#!/usr/bin/env python3
"""Render report.md to a self-contained HTML page for publishing as an Artifact.

Generating the page *from* the markdown guarantees the two never drift, and
makes re-syncing after an edit a single command:

    python3 papers/build_artifact.py     # -> papers/report_artifact.html
    python3 papers/build_artifact.py summary.md summary_artifact.html "Recursion Under Delay: Summary" Summary

Handles the markdown subset the report actually uses: headings, paragraphs,
pipe tables, images (base64-embedded from figures/), bulleted and numbered
lists, blockquotes, horizontal rules, and inline bold/italic/code. Stdlib only.
"""
import base64
import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "report.md"
OUT = HERE / "report_artifact.html"

CSS = """
:root{--paper:#efeee7;--surface:#f8f7f2;--raised:#fff;--ink:#23261f;--ink-soft:#454a3f;
--muted:#6c7264;--rule:#d3d2c5;--rule-strong:#b6b6a5;--accent:#2c6b5e;--accent-soft:#dfe9e2;
--signal:#a8681f;--signal-soft:#f1e3cd;--shadow:0 1px 2px rgba(35,38,31,.05),0 6px 20px rgba(35,38,31,.06)}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--paper:#14170f;--surface:#191c14;
--raised:#1f2318;--ink:#e8e7dc;--ink-soft:#c6c8b9;--muted:#8d9382;--rule:#33372a;--rule-strong:#454a38;
--accent:#6cb5a2;--accent-soft:#1f322b;--signal:#d99a4e;--signal-soft:#332612;
--shadow:0 1px 2px rgba(0,0,0,.35),0 8px 28px rgba(0,0,0,.35)}}
:root[data-theme=dark]{--paper:#14170f;--surface:#191c14;--raised:#1f2318;--ink:#e8e7dc;
--ink-soft:#c6c8b9;--muted:#8d9382;--rule:#33372a;--rule-strong:#454a38;--accent:#6cb5a2;
--accent-soft:#1f322b;--signal:#d99a4e;--signal-soft:#332612;
--shadow:0 1px 2px rgba(0,0,0,.35),0 8px 28px rgba(0,0,0,.35)}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Public Sans",-apple-system,"Segoe UI",sans-serif;
font-size:16px;line-height:1.62}
img{max-width:100%}
.mono,code{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
code{font-size:.86em;background:var(--raised);border:1px solid var(--rule);border-radius:4px;padding:.05em .35em}
.masthead{border-bottom:1px solid var(--rule-strong);background:var(--surface)}
.mast-in{max-width:800px;margin:0 auto;padding:52px 24px 36px}
h1.title{font-family:Fraunces,Georgia,serif;font-weight:600;font-size:clamp(26px,4vw,38px);line-height:1.18;
letter-spacing:-.01em;margin:0 0 18px;text-wrap:balance}
.status{display:flex;gap:10px;align-items:flex-start;font-size:13.5px;color:var(--muted)}
.status .pill{flex:none;display:inline-flex;align-items:center;gap:5px;font-family:"JetBrains Mono",monospace;
font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;padding:3px 9px 3px 8px;
border-radius:3px;background:var(--accent-soft);color:var(--accent)}
.status .pill::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.status .note{max-width:66ch}
.page{max-width:800px;margin:0 auto;padding:0 24px 100px}
section{padding-top:46px;scroll-margin-top:20px}
.sec-head{display:flex;align-items:baseline;gap:13px;margin-bottom:4px}
.sec-num{font-family:"JetBrains Mono",monospace;font-size:13px;color:var(--muted);flex:none}
h2{font-family:Fraunces,Georgia,serif;font-weight:600;font-size:23px;margin:0;text-wrap:balance}
h3{font-family:Fraunces,Georgia,serif;font-weight:600;font-size:17px;margin:26px 0 8px;color:var(--ink)}
.sec-rule{height:1px;background:var(--rule);margin:13px 0 24px}
p{margin:0 0 17px;max-width:68ch}
strong{color:var(--ink);font-weight:650}
ul,ol{margin:0 0 17px;padding-left:22px;max-width:68ch}
li{margin-bottom:10px}
li:last-child{margin-bottom:0}
.tbl{overflow-x:auto;margin:6px 0 22px;border:1px solid var(--rule);border-radius:8px;background:var(--raised)}
table{border-collapse:collapse;width:100%;min-width:520px;font-size:13.5px}
thead th{text-align:left;font-family:"JetBrains Mono",monospace;font-size:10.5px;letter-spacing:.05em;
text-transform:uppercase;color:var(--muted);font-weight:600;padding:11px 13px;
border-bottom:1px solid var(--rule-strong);white-space:nowrap}
tbody td{padding:9px 13px;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}
tbody td:first-child{font-weight:600}
figure{margin:14px 0 26px;background:var(--raised);border:1px solid var(--rule);border-radius:8px;overflow:hidden}
figure img{display:block;width:100%;height:auto;background:#fff}
blockquote{margin:6px 0 22px;padding:14px 20px;border-left:3px solid var(--accent);background:var(--surface);
border-radius:0 8px 8px 0;max-width:68ch}
blockquote p:last-child{margin-bottom:0}
h4{font-family:"JetBrains Mono",monospace;font-size:12px;letter-spacing:.06em;text-transform:uppercase;
color:var(--signal);margin:28px 0 10px}
figcaption{padding:10px 14px 12px;font-size:12.5px;color:var(--muted);border-top:1px solid var(--rule)}
.abstract{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:26px 28px;
box-shadow:var(--shadow)}
.abstract p:last-child{margin-bottom:0}
nav.toc{position:sticky;top:0;z-index:5;background:var(--paper);border-bottom:1px solid var(--rule)}
nav.toc .in{max-width:800px;margin:0 auto;padding:9px 24px;display:flex;gap:3px;overflow-x:auto;scrollbar-width:none}
nav.toc .in::-webkit-scrollbar{display:none}
nav.toc a{flex:none;font-family:"JetBrains Mono",monospace;font-size:12px;color:var(--muted);
text-decoration:none;padding:5px 9px;border-radius:5px;white-space:nowrap}
nav.toc a:hover{background:var(--surface);color:var(--ink)}
a{color:var(--accent)}
"""


def b64_img(rel: str) -> str:
    p = (HERE / rel).resolve()
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def render(md: str, page_title: str = "Recursion Under Delay", pill: str = "Draft") -> tuple[str, str]:
    lines = md.split("\n")
    title, subtitle = "", ""
    body: list[str] = []
    toc: list[tuple[str, str]] = []
    i, sec_open = 0, False
    n_sections = 0

    def close():
        nonlocal sec_open
        if sec_open:
            body.append("</section>")
            sec_open = False

    while i < len(lines):
        ln = lines[i]

        if ln.startswith("# ") and not title:
            title = ln[2:].strip()
            i += 1
            continue

        if not title:
            i += 1
            continue

        # italic status line directly under the title
        if not subtitle and ln.startswith("*") and ln.rstrip().endswith("*") and not ln.startswith("**"):
            subtitle = inline(ln.strip().strip("*"))
            i += 1
            continue

        if ln.startswith("## "):
            close()
            head = ln[3:].strip()
            m = re.match(r"^(\d+)\.\s+(.*)$", head)
            n_sections += 1
            sid = f"s{n_sections}"
            if m:
                num, name = f"{int(m.group(1)):02d}", m.group(2)
            else:
                num, name = "&mdash;", head
            toc.append((sid, head if not m else f"{m.group(1)}\u00b7{m.group(2).split(':')[0]}"))
            body.append(f'<section id="{sid}"><div class="sec-head">'
                        f'<span class="sec-num">{num}</span><h2>{inline(name)}</h2></div>'
                        f'<div class="sec-rule"></div>')
            sec_open = True
            # the Abstract section gets the raised-capsule treatment
            if name.lower().startswith("abstract"):
                body.append('<div class="abstract">')
                i += 1
                while i < len(lines) and not lines[i].startswith("## "):
                    if lines[i].strip():
                        body.append(f"<p>{inline(lines[i].strip())}</p>")
                    i += 1
                body.append("</div>")
                continue
            i += 1
            continue

        if ln.strip() == "---":
            i += 1
            continue

        if ln.startswith("#### "):
            body.append(f"<h4>{inline(ln[5:].strip())}</h4>")
            i += 1
            continue

        if ln.startswith("> ") or ln.strip() == ">":
            quote = []
            while i < len(lines) and (lines[i].startswith("> ") or lines[i].strip() == ">"):
                quote.append(lines[i][2:].strip())
                i += 1
            body.append("<blockquote>" + "".join(f"<p>{inline(q)}</p>" for q in quote if q) + "</blockquote>")
            continue

        if ln.startswith("### "):
            body.append(f"<h3>{inline(ln[4:].strip())}</h3>")
            i += 1
            continue

        # image -> figure
        m = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", ln)
        if m:
            alt, src = m.group(1), m.group(2)
            body.append(f'<figure><img src="{b64_img(src)}" alt="{html.escape(alt)}">'
                        f'<figcaption>{inline(alt)}</figcaption></figure>')
            i += 1
            continue

        # pipe table
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|\s*$", lines[i + 1]):
            def cells(row):
                return [c.strip() for c in row.strip().strip("|").split("|")]
            head = cells(ln)
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tb = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            body.append(f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>')
            continue

        # lists
        if re.match(r"^\s*[-*]\s+", ln) or re.match(r"^\s*\d+\.\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.\s+", ln))
            items = []
            while i < len(lines) and (re.match(r"^\s*[-*]\s+", lines[i]) or re.match(r"^\s*\d+\.\s+", lines[i])):
                items.append(re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", lines[i]))
                i += 1
            tag = "ol" if ordered else "ul"
            body.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue

        if ln.strip():
            body.append(f"<p>{inline(ln.strip())}</p>")
        i += 1

    close()
    nav = "".join(f'<a href="#{sid}">{html.escape(label)}</a>' for sid, label in toc)
    page = f"""<title>{html.escape(page_title)}</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Public+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<nav class="toc"><div class="in">{nav}</div></nav>
<header class="masthead"><div class="mast-in">
<h1 class="title">{inline(title)}</h1>
<div class="status"><span class="pill">{html.escape(pill)}</span><span class="note">{subtitle}</span></div>
</div></header>
<div class="page">
{chr(10).join(body)}
</div>"""
    return page, title


if __name__ == "__main__":
    args = sys.argv[1:]
    src = HERE / args[0] if args else SRC
    out = HERE / args[1] if len(args) > 1 else OUT
    extra = {k: v for k, v in zip(("page_title", "pill"), args[2:4])}
    page, title = render(src.read_text(), **extra)
    out.write_text(page)
    print(f"Wrote {out}  ({len(page)/1024:.0f} KiB)  title: {title}")
