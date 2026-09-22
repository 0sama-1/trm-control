"""Render board_template.html with the latest poll embedded as its at-rest fallback."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dashboard_state" / "out"
fallback = {k: json.loads((OUT / f"{k}.json").read_text()) for k in ("state", "jobs", "log")}
html = (HERE / "board_template.html").read_text()
html = html.replace("/*__FALLBACK__*/null", json.dumps(fallback).replace("</", "<\\/"))
dest = HERE.parents[1] / "dashboard" / "phase3_board.html"
dest.write_text(html)
print(dest, len(html))
