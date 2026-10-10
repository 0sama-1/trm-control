"""Phase 5 job manager -- runs on the Katana login node (stdlib only).

  python p5.py manifest <stage>               print a stage's jobs as JSON
  python p5.py submit <stage> [--dry-run]     generate PBS scripts for a stage and qsub them
  python p5.py status                         one JSON document: every Phase 4 job + live progress
  python p5.py resubmit <job-name> [--walltime HH:MM:SS]

Same scheme as experiments/phase4/p4.py (copied, with Phase 5 paths so the Phase 3-4
ledgers and snapshots are never touched): every stage runs from a frozen code
snapshot (P4_CODE/<stage>, see sync.sh); results, checkpoints and logs go to
fixed absolute paths under ROOT. Stage manifests follow phase5/plan/plan.md.
Stage-specific values that are set by a gate or a pilot (B1's env + step budget,
Stage A's clip norms) are read from gates.json, which is frozen before data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/srv/scratch/z5308482/trm-control")
P5_CODE = ROOT / "p5code"
LOGS = ROOT / "phase5_logs"
JOBS_DIR = Path.home() / "jobs" / "phase5"
LEDGER = JOBS_DIR / "ledger.jsonl"
GATES = ROOT / "phase5" / "gates.json"
VENV = "/srv/scratch/z5308482/venvs/trm-control/bin/activate"
SEEDS6 = [1, 2, 3, 4, 5, 6]
SEEDS3 = [1, 2, 3]

# Condition presets: Hydra overrides + the filename fields the trainer derives.
COND = {
    # No delay (queue width 0, so augment_obs is a no-op and the filename has no aug tag).
    "C0": {"ov": ["delay=none", "augment_obs=true"], "delay": "none", "aug": None, "tag": ""},
    # C1 as in Phase 3 Stage 1 (fixed d=1 + queue, no previous-action input): Stages B and C.
    "C1": {"ov": ["delay=fixed", "delay.value=1", "augment_obs=true"], "delay": "fixed", "aug": 1, "tag": ""},
    # Stage A conditions: previous action also shown, matching Phase 3 Stage 2 (amendment A-6).
    "C1pa": {"ov": ["delay=fixed", "delay.value=1", "augment_obs=true", "prev_action_obs=true"],
             "delay": "fixed", "aug": 1, "tag": "pa"},
    "C4pa": {"ov": ["delay=stochastic_u4", "augment_obs=false", "prev_action_obs=true"],
             "delay": "stochastic", "aug": 0, "tag": "pa"},
}
ENV_IDS = {"pendulum": "Pendulum-v1", "invdoublependulum": "InvertedDoublePendulum-v5", "hopper": "Hopper-v5"}
# Default learning rates (conf/arch/ddpg.yaml) and the Stage C multipliers.
DDPG_LR = {"actor": 1e-4, "critic": 1e-3}
LR_MULT = {"0.3": 0.3, "1": 1.0, "3": 3.0}
# Stage B0 env-step budgets (plan §3).
B0_BUDGETS = [150_000, 300_000, 500_000]


def _train_job(stage, results, model, cond, seed, env="pendulum", arch="ddpg", episodes=200, steps=None,
               walltime="01:59:00", run_tag="", extra=(), group=None):
    c = COND[cond]
    tag = "_".join(t for t in [c["tag"], run_tag] if t)
    name = f"p5{stage}_{model}_{cond}" + (f"_{run_tag}" if run_tag else "") + f"_s{seed}"
    aug_tag = "" if c["delay"] == "none" else f"_aug{c['aug']}"
    result = f"{results}/{ENV_IDS[env]}_{arch}_{model}_{c['delay']}{aug_tag}" + (f"_{tag}" if tag else "") + f"_seed{seed}.json"
    budget = [f"+arch.total_steps={steps}"] if steps else [f"arch.total_episodes={episodes}"]
    if steps:
        budget.append("arch.total_episodes=100000")   # never binds; total_steps ends the run
    ov = [f"arch={arch}", f"model={model}", f"env={env}", *c["ov"], *budget,
          f"seed={seed}", "wandb.enabled=false", "print_stat_every=1", "record_every=100000", "eval_every=10",
          f"results_folder={ROOT / results}", f"model_folder={ROOT / 'models'}",
          f"video_folder={ROOT / 'videos'}", f"track_step_history={'true' if env == 'pendulum' else 'false'}",
          f"hydra.run.dir={LOGS / 'hydra' / name}", *([f"run_tag={tag}"] if tag else []), *extra]
    return {"name": name, "stage": stage, "kind": "train", "group": group or f"{model}|{cond}|{run_tag}",
            "model": model, "cond": cond, "seed": seed, "env": env, "arch": arch, "episodes": episodes,
            "steps": steps, "walltime": walltime, "run_tag": run_tag,
            "cmd": "python3 -u main.py " + " ".join(ov), "result": str(ROOT / result)}


def _gates() -> dict:
    return json.loads(GATES.read_text()) if GATES.exists() else {}


def _clip_overrides() -> list[str]:
    clip = _gates().get("frozen_values", {}).get("stage_a_clip")
    if not clip or clip.get("actor") is None or clip.get("critic") is None:
        raise SystemExit("Stage A clip norms are not frozen in gates.json yet (set them from the a0 pilot)")
    return [f"+arch.actor_grad_clip={clip['actor']}", f"+arch.critic_grad_clip={clip['critic']}"]


def manifest(stage: str) -> list[dict]:
    if stage == "t2":      # Stage T2 (plan §2): TRM 788k under TD3 at three actor LRs; MLP 788k control at actor LR 1e-4
        reuse = _gates().get("frozen_values", {}).get("t2_reuse_stage_t_lr1e-3", False)
        lrs = ["1e-4", "3e-4"] + ([] if reuse else ["1e-3"])
        jobs = [_train_job("t2", "results_p5t2", "trm", c, s, arch="td3", run_tag=f"alr{lr}", group=f"trm|alr{lr}",
                           extra=[f"arch.actor_lr={lr}"])
                for lr in lrs for c in ["C0", "C1"] for s in SEEDS6]
        jobs += [_train_job("t2", "results_p5t2", "dnn_large", c, s, arch="td3", run_tag="alr1e-4", group="dnn_large|alr1e-4",
                            extra=["arch.actor_lr=1e-4"]) for c in ["C0", "C1"] for s in SEEDS6]
        return jobs
    raise SystemExit(f"unknown stage {stage!r}")


# gpu_rp6000=False: the RTX PRO 6000 Blackwell node (sm_120) has no kernels in the venv's
# torch 2.5.1+cu121 build (Phase 3 incident 1).
PBS = """#!/bin/bash
#PBS -N {name}
#PBS -l select=1:ncpus=2:mem={mem}:ngpus=1:gpu_rp6000=False
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {pbs_out}

module purge; module add python/3.11.3; module add cuda/12.1.1
source {venv}
cd {code}
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start $(date -Is)" > {log}
( {cmd} ) >> {log} 2>&1
rc=$?
echo "exit $rc end $(date -Is)" >> {log}
exit $rc
"""


def _ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def _qsub(job: dict, code: Path, dry: bool) -> str | None:
    d = JOBS_DIR / job["stage"]
    d.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    script = d / f"{job['name']}.pbs"
    # Replay buffers on IDP/Hopper hold up to 200k 10-D transitions as Python tuples.
    mem = "8gb" if job.get("env", "pendulum") == "pendulum" else "16gb"
    script.write_text(PBS.format(name=job["name"], walltime=job["walltime"], mem=mem,
                                 pbs_out=d / f"{job['name']}.o", venv=VENV, code=code,
                                 log=LOGS / f"{job['name']}.log", cmd=job["cmd"]))
    if dry:
        return None
    out = subprocess.run(["qsub", str(script)], capture_output=True, text=True)
    if out.returncode != 0:
        print(f"qsub failed for {job['name']}: {out.stderr.strip()}", file=sys.stderr)
        return None
    return out.stdout.strip()


def cmd_submit(stage: str, dry: bool) -> None:
    code = P5_CODE / stage
    if not (code / "main.py").exists():
        raise SystemExit(f"no code snapshot at {code}: sync one first")
    done = {e["name"] for e in _ledger()}
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for job in manifest(stage):
        if job["name"] in done:
            continue
        jid = _qsub(job, code, dry)
        if dry:
            print("would submit", job["name"], job["walltime"])
            continue
        if jid:
            rec = {**job, "jobid": jid, "code": str(code), "submitted": dt.datetime.now().isoformat(timespec="seconds")}
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n += 1
            print(f"{job['name']} {jid}")
    if n and not dry and stage != "smoke":
        (code / ".frozen").write_text(dt.datetime.now().isoformat(timespec="seconds") + "\n")
    print(f"submitted {n} jobs for stage {stage}")


def cmd_resubmit(name: str, walltime: str | None = None) -> None:
    recs = [e for e in _ledger() if e["name"] == name]
    if not recs:
        raise SystemExit(f"{name} not in ledger")
    job = dict(recs[-1])
    if walltime:
        job["walltime"] = walltime
    jid = _qsub(job, Path(job["code"]), False)
    if jid:
        rec = {**job, "jobid": jid, "submitted": dt.datetime.now().isoformat(timespec="seconds"),
               "resubmit_of": job["jobid"]}
        with open(LEDGER, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name} {jid}")


_EP = re.compile(r"Episode\s+(\d+)/(\d+) \| Return:\s+(-?[\d.]+) \| Avg100:\s+(-?[\d.]+)")


def _progress(name: str) -> dict:
    log = LOGS / f"{name}.log"
    if not log.exists():
        return {}
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode(errors="replace")
        with open(log) as f:
            head = f.readline().strip()
    except OSError:
        return {}
    out = {"log_head": head[:160]}
    m = list(_EP.finditer(tail))
    if m:
        ep, tot, ret, avg = m[-1].groups()
        out.update(episode=int(ep), ret=float(ret), avg100=float(avg))
    for line in reversed(tail.splitlines()):
        if line.startswith("exit "):
            out["exit_line"] = line.strip()
            break
    if "Traceback" in tail:
        out["error"] = tail[tail.rindex("Traceback"):][-600:]
    return out


def _qstat(ids: list[str]) -> dict:
    info = {}
    for i in range(0, len(ids), 200):
        r = subprocess.run(["qstat", "-x", "-f", "-F", "json", *ids[i:i + 200]], capture_output=True, text=True)
        try:
            info.update(json.loads(r.stdout).get("Jobs", {}))
        except json.JSONDecodeError:
            pass
    return info


def _result_summary(path: str | None, kind: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    if kind != "train":
        return {"present": True}
    try:
        with open(path) as f:
            txt = f.read(4096)   # the scalar fields come first; skip parsing the histories
        grab = lambda k: re.search(rf'"{k}": (-?[\d.eE+-]+|"[^"]*")', txt)
        out = {"present": True}
        for k in ["final_avg100_return", "wall_clock_seconds", "param_count", "total_steps", "episode_count"]:
            m = grab(k)
            if m:
                out[k] = float(m.group(1))
        m = grab("device_name")
        if m:
            out["device_name"] = m.group(1).strip('"')
        return out
    except OSError:
        return {"present": True}


def cmd_status() -> None:
    latest = {}
    for e in _ledger():
        latest[e["name"]] = e          # a resubmission supersedes the earlier record
    q = _qstat([e["jobid"] for e in latest.values()])
    jobs = []
    for name, e in latest.items():
        qi = q.get(e["jobid"], {})
        st, exit_status = qi.get("job_state"), qi.get("Exit_status")
        res = _result_summary(e.get("result"), e["kind"])
        prog = _progress(name)
        if st in ("Q", "H", "W", "T"):
            state = "queued" if st != "H" else "held"
        elif st in ("R", "E", "B"):
            state = "running"
        elif st == "F" or (not qi and "exit_line" in prog):
            # A script that logged "exit 0" and wrote its result is done even if PBS then
            # reports a nonzero code: exit -19 has come from node cleanup after the job ended.
            clean = prog.get("exit_line", "").startswith("exit 0")
            has_out = res is not None or e["kind"] != "train"   # analysis jobs write folders, not one result
            state = "done" if has_out and (exit_status in (0, None) or clean) else "failed"
        elif res is not None:
            state = "done"
        else:
            state = "unknown"
        ru = qi.get("resources_used", {})
        jobs.append({
            "name": name, "stage": e["stage"], "kind": e["kind"], "group": e["group"],
            "model": e.get("model"), "cond": e.get("cond"), "seed": e.get("seed"), "env": e.get("env"),
            "steps": e.get("steps"), "jobid": e["jobid"].split(".")[0], "state": state, "pbs_state": st,
            "exit": exit_status, "submitted": e["submitted"], "walltime_req": e["walltime"],
            "walltime_used": ru.get("walltime"), "host": (qi.get("exec_host") or "").split("/")[0] or None,
            "stime": qi.get("stime"), "queue": qi.get("queue"), "comment": qi.get("comment"),
            "progress": {k: v for k, v in prog.items() if k != "error"},
            "error": prog.get("error") if state == "failed" else None, "result": res,
        })
    print(json.dumps({"polled_at": dt.datetime.now().isoformat(timespec="seconds"), "jobs": jobs}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("stage"); s.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    r = sub.add_parser("resubmit"); r.add_argument("name"); r.add_argument("--walltime")
    m = sub.add_parser("manifest"); m.add_argument("stage")
    a = ap.parse_args()
    if a.cmd == "submit":
        cmd_submit(a.stage, a.dry_run)
    elif a.cmd == "status":
        cmd_status()
    elif a.cmd == "resubmit":
        cmd_resubmit(a.name, a.walltime)
    else:
        print(json.dumps(manifest(a.stage), indent=1))


if __name__ == "__main__":
    main()
