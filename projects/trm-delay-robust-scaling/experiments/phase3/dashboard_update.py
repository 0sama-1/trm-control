"""Build the Phase 3 run-board documents from one Katana status snapshot.

  python dashboard_update.py <status.json> [--event "text" ...] [--next-poll-min 18]

Reads `p3.py status` output, diffs it against the previous snapshot to log
state transitions, merges gate decisions from gates.json, and writes
out/state.json, out/jobs.json, out/log.json -- the three documents the
dashboard subscribes to (collection "p3"). Local state lives next to this
file in dashboard_state/. Stdlib only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "dashboard_state"
OUT = STATE / "out"

# Planned job counts per ledger stage key, and how ledger stages roll up into plan stages.
PARTS = {
    "s0h": {"stage": 0, "label": "S0.1 headroom probe", "planned": 24},
    "s0b": {"stage": 0, "label": "S0.2 checkpoint battery (102 checkpoints, 12 shards)", "planned": 12},
    "s0l": {"stage": 0, "label": "S0.2 latency benchmark", "planned": 1},
    "s1":  {"stage": 1, "label": "Capacity x depth factorial", "planned": 72},
    "s2":  {"stage": 2, "label": "Uncertainty x memory", "planned": 72},
    "s2o": {"stage": 2, "label": "Optional C5: time-varying delay", "planned": 24},
    "s2x": {"stage": 2, "label": "Supplementary: latent per-episode delay (G2s)", "planned": 30},
    "s3t": {"stage": 3, "label": "TD3 cross-check", "planned": 24},
    "s3p": {"stage": 3, "label": "InvertedDoublePendulum budget pilot", "planned": 9},
    "s3i": {"stage": 3, "label": "InvertedDoublePendulum matrix", "planned": 48},
    "s4":  {"stage": 4, "label": "Training-discipline variants", "planned": 36},
    "s5b": {"stage": 5, "label": "Battery over all 168 Phase 3 checkpoints", "planned": 12},
}
STAGES = [
    {"n": 0, "title": "Is there a question?", "sub": "Headroom probe + zero-training evaluation of the 102 banked checkpoints",
     "entry": "none (committed)"},
    {"n": 1, "title": "Capacity or recursion?", "sub": "2x2 architecture x capacity, plus H_cycles in {1, 3, 6} at fixed parameters",
     "entry": "none (not gated)"},
    {"n": 2, "title": "Uncertainty x memory", "sub": "Carry-Z vs Reset-Z vs MLP vs carried LSTM across three delay conditions",
     "entry": "G0 passed and G1 decided"},
    {"n": 3, "title": "Does it generalise?", "sub": "TD3 cross-check + an open-loop-unstable plant",
     "entry": "G2 shows an interaction"},
    {"n": 4, "title": "Restore the training discipline", "sub": "No truncation, per-cycle supervision, fixed-point readout",
     "entry": "G3 passed"},
    {"n": 5, "title": "Synthesis", "sub": "Report revision -- no compute", "entry": "always, after the last stage that runs"},
]
DEFAULT_GATES = [
    {"id": "G0", "name": "Headroom", "after": 0,
     "rule": "Oracle-delay MLP beats augmented MLP at stochastic delay by more than the pooled seed spread, in >=5/6 seeds "
             "(amended: or augmentation beats no-augmentation by the same criterion -- memory headroom)",
     "status": "pending", "detail": ""},
    {"id": "G1", "name": "Attribution", "after": 1,
     "rule": "2x2 architecture x capacity contrast; no kill branch -- sets Stage 2's capacity level",
     "status": "pending", "detail": ""},
    {"id": "G2", "name": "Interaction", "after": 2,
     "rule": "(Carry-Z - MLP) flips sign from fixed-known to stochastic delay, >=5/6 seeds",
     "status": "pending", "detail": ""},
    {"id": "G3", "name": "Continuity", "after": 3,
     "rule": "Carry-Z beats Reset-Z at some condition",
     "status": "pending", "detail": ""},
    {"id": "G4", "name": "Practicality", "after": 0,
     "rule": "Some trunk sits favourably on the return / effort / latency front",
     "status": "pending", "detail": ""},
]
TERMINAL = {"done", "failed"}


def _load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _hms_to_s(t: str | None) -> int | None:
    if not t:
        return None
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _short(name: str) -> str:
    return name.split("_", 1)[1] if "_" in name else name


def build(status: dict, events: list[str], next_poll_min: float) -> None:
    STATE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    prev = {j["name"]: j for j in _load(STATE / "last_status.json", {"jobs": []})["jobs"]}
    log = _load(STATE / "log.json", {"events": []})["events"]
    gates = _load(STATE / "gates.json", {"gates": DEFAULT_GATES})["gates"]
    notes = _load(STATE / "notes.json", {"stage_notes": {}, "headline": ""})
    now = status["polled_at"]

    jobs = [j for j in status["jobs"] if j["stage"] in PARTS]

    # Transition events, grouped so a 72-job submission is one line, not 72.
    groups: dict[tuple, list[str]] = {}
    for j in jobs:
        before = prev.get(j["name"], {}).get("state")
        if before == j["state"]:
            continue
        key = (j["stage"], j["state"], before is None)
        groups.setdefault(key, []).append(j["name"])
    for (stage, state, new), names in sorted(groups.items()):
        label = PARTS[stage]["label"]
        if new and state in ("queued", "running"):
            text = f"{len(names)} job(s) submitted: {label}"
            kind = "submit"
        elif state == "running":
            text = f"{len(names)} started: " + ", ".join(_short(n) for n in names[:6]) + (" ..." if len(names) > 6 else "")
            kind = "start"
        elif state == "done":
            text = f"{len(names)} finished: " + ", ".join(_short(n) for n in names[:6]) + (" ..." if len(names) > 6 else "")
            kind = "done"
        elif state == "failed":
            text = f"{len(names)} FAILED: " + ", ".join(_short(n) for n in names)
            kind = "fail"
        else:
            continue
        log.append({"ts": now, "kind": kind, "text": text})
    for e in events:
        kind, _, text = e.partition(":") if e.split(":", 1)[0] in ("note", "gate", "submit", "fail", "fix") else ("note", "", e)
        log.append({"ts": now, "kind": kind, "text": text.strip()})
    log = log[-200:]

    # Per-part and per-stage rollups.
    parts = {}
    for key, meta in PARTS.items():
        js = [j for j in jobs if j["stage"] == key]
        c = {s: sum(1 for j in js if j["state"] == s) for s in ("queued", "running", "done", "failed", "held")}
        c["submitted"] = len(js)
        c["planned"] = max(meta["planned"], len(js))
        if not js:
            status_ = "not submitted"
        elif c["done"] == len(js) and len(js) >= meta["planned"]:
            status_ = "complete"
        elif c["running"]:
            status_ = "running"
        elif c["queued"] or c["held"]:
            status_ = "queued" if not c["done"] else "running"
        elif c["failed"]:
            status_ = "attention"
        else:
            status_ = "running"
        parts[key] = {**meta, "counts": c, "status": status_}

    stages = []
    for s in STAGES:
        ps = [dict(parts[k], key=k) for k in PARTS if PARTS[k]["stage"] == s["n"]]
        submitted = sum(p["counts"]["submitted"] for p in ps)
        st = "not started"
        if submitted:
            sts = {p["status"] for p in ps if p["counts"]["submitted"]}
            if sts == {"complete"} and all(p["status"] == "complete" for p in ps if p["key"] not in ("s2o",)):
                st = "complete"
            elif "attention" in sts:
                st = "attention"
            elif "running" in sts or "complete" in sts:
                st = "running"
            else:
                st = "queued"
        st = notes.get("stage_status", {}).get(str(s["n"]), st)
        stages.append({**s, "status": st, "parts": ps, "note": notes.get("stage_notes", {}).get(str(s["n"]), "")})

    tot = {s: sum(1 for j in jobs if j["state"] == s) for s in ("queued", "running", "done", "failed", "held")}
    tot["submitted"] = len(jobs)

    # Preliminary per-arm returns for training jobs (final avg100 over the last 100 episodes).
    arms: dict[str, dict] = {}
    for j in jobs:
        if j["kind"] != "train":
            continue
        a = arms.setdefault(f"{j['stage']}|{j['group']}", {
            "stage": j["stage"], "model": j["model"], "cond": j["cond"], "done": 0, "total": 0,
            "finals": {}, "live": {}, "wall_min": []})
        a["total"] += 1
        r = j.get("result") or {}
        if j["state"] == "done" and "final_avg100_return" in r:
            a["done"] += 1
            a["finals"][str(j["seed"])] = round(r["final_avg100_return"], 1)
            if "wall_clock_seconds" in r:
                a["wall_min"].append(r["wall_clock_seconds"] / 60)
        elif j["state"] == "running" and j["progress"].get("avg100") is not None:
            a["live"][str(j["seed"])] = [j["progress"]["episode"], round(j["progress"]["avg100"], 1)]
    arm_rows = []
    for key, a in sorted(arms.items()):
        f = list(a["finals"].values())
        arm_rows.append({
            "stage": a["stage"], "model": a["model"], "cond": a["cond"], "done": a["done"], "total": a["total"],
            "median": round(statistics.median(f), 1) if f else None,
            "spread": round(statistics.pstdev(f), 1) if len(f) > 1 else None,
            "finals": a["finals"], "live": a["live"],
            "wall_min": round(statistics.median(a["wall_min"]), 1) if a["wall_min"] else None,
        })

    running = []
    for j in jobs:
        if j["state"] != "running":
            continue
        p = j["progress"]
        head = p.get("log_head", "")
        gpu = head.split(" gpu ", 1)[1].split(" start ")[0] if " gpu " in head else None
        running.append({"name": _short(j["name"]), "stage": j["stage"], "elapsed_s": _hms_to_s(j["walltime_used"]),
                        "wall_req_s": _hms_to_s(j["walltime_req"]), "episode": p.get("episode"),
                        "total": p.get("total"), "avg100": p.get("avg100"), "host": j.get("host"), "gpu": gpu})
    running.sort(key=lambda r: (r["stage"], r["name"]))

    finished = sorted([j for j in jobs if j["state"] in TERMINAL], key=lambda j: j.get("mtime") or "", reverse=True)
    recent = [{"name": _short(j["name"]), "stage": j["stage"], "state": j["state"],
               "final": (j.get("result") or {}).get("final_avg100_return"),
               "wall_min": round((j.get("result") or {}).get("wall_clock_seconds", 0) / 60, 1) or None,
               "device": (j.get("result") or {}).get("device_name"),
               "error": (j.get("error") or "")[-300:] or None} for j in finished[:12]]

    try:
        polled = dt.datetime.fromisoformat(now)
        next_poll = (polled + dt.timedelta(minutes=next_poll_min)).isoformat(timespec="seconds")
    except ValueError:
        next_poll = None

    state_doc = {"polled_at": now, "next_poll_at": next_poll, "totals": tot, "stages": stages, "gates": gates,
                 "running": running, "recent": recent, "arms": arm_rows, "headline": notes.get("headline", ""),
                 "amendments": notes.get("amendments", [])}
    jobs_doc = {"polled_at": now, "jobs": [
        {"n": _short(j["name"]), "s": j["stage"], "st": j["state"], "g": j["group"], "seed": j.get("seed"),
         "id": j["jobid"], "q": j.get("queue"), "ep": j["progress"].get("episode"), "tot": j["progress"].get("total"),
         "fin": (j.get("result") or {}).get("final_avg100_return"), "w": j.get("walltime_used")} for j in jobs]}
    (OUT / "state.json").write_text(json.dumps(state_doc))
    (OUT / "jobs.json").write_text(json.dumps(jobs_doc))
    (OUT / "log.json").write_text(json.dumps({"events": log}))
    (STATE / "log.json").write_text(json.dumps({"events": log}))
    (STATE / "last_status.json").write_text(json.dumps(status))
    print(json.dumps({"totals": tot, "stages": {s["n"]: s["status"] for s in stages}}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("status")
    ap.add_argument("--event", action="append", default=[])
    ap.add_argument("--next-poll-min", type=float, default=18)
    args = ap.parse_args()
    build(json.loads(Path(args.status).read_text()), args.event, args.next_poll_min)


if __name__ == "__main__":
    main()
