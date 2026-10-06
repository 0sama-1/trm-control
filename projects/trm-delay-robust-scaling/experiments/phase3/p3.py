"""Phase 3 job manager -- runs on the Katana login node (stdlib only).

  python p3.py submit <stage> [--dry-run]   generate PBS scripts for a stage and qsub them
  python p3.py status                       one JSON document: every Phase 3 job + live progress
  python p3.py resubmit <job-name>          qsub a job again (e.g. after a node failure)

Every stage runs from a frozen code snapshot (P3_CODE/<stage>), so editing the
live tree for a later stage never changes the code a queued job will import.
Results, checkpoints and logs go to fixed absolute paths under P3_ROOT.
Stage manifests follow projects/trm-delay-robust-scaling/phase3/plan/plan.md §4
plus the §10 amendment (headroom arm A4).
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

P3_ROOT = Path("/srv/scratch/z5308482/trm-control")
P3_CODE = P3_ROOT / "p3code"
LOGS = P3_ROOT / "phase3_logs"
JOBS_DIR = Path.home() / "jobs" / "phase3"
LEDGER = JOBS_DIR / "ledger.jsonl"
VENV = "/srv/scratch/z5308482/venvs/trm-control/bin/activate"
SEEDS = [1, 2, 3, 4, 5, 6]

# Condition presets: Hydra overrides + the filename fields the trainer derives.
COND = {
    "none":          {"ov": ["delay=none", "augment_obs=true"], "delay": "none", "aug": None, "tag": ""},
    "fixed1-aug":    {"ov": ["delay=fixed", "delay.value=1", "augment_obs=true"], "delay": "fixed", "aug": 1, "tag": ""},
    "stoch4-aug":    {"ov": ["delay=stochastic_u4", "augment_obs=true"], "delay": "stochastic", "aug": 1, "tag": ""},
    "stoch4-oracle": {"ov": ["delay=stochastic_u4", "augment_obs=true", "delay.expose_realised=true", "run_tag=oracle"],
                      "delay": "stochastic", "aug": 1, "tag": "oracle"},
    "stoch4-noaug":  {"ov": ["delay=stochastic_u4", "augment_obs=false"], "delay": "stochastic", "aug": 0, "tag": ""},
    # Stage 2 conditions: every trunk also sees its previous action (plan §10, A-6).
    "C1": {"ov": ["delay=fixed", "delay.value=1", "augment_obs=true", "prev_action_obs=true", "run_tag=pa"],
           "delay": "fixed", "aug": 1, "tag": "pa"},
    "C3": {"ov": ["delay=stochastic_u4", "augment_obs=true", "prev_action_obs=true", "run_tag=pa"],
           "delay": "stochastic", "aug": 1, "tag": "pa"},
    "C4": {"ov": ["delay=stochastic_u4", "augment_obs=false", "prev_action_obs=true", "run_tag=pa"],
           "delay": "stochastic", "aug": 0, "tag": "pa"},
    # Supplementary latent per-episode delay (G2s): d ~ U{0..4} fixed within an episode.
    "C6": {"ov": ["delay=episode_u4", "augment_obs=true", "prev_action_obs=true", "run_tag=pa"],
           "delay": "episode", "aug": 1, "tag": "pa"},
    "C6o": {"ov": ["delay=episode_u4", "augment_obs=true", "prev_action_obs=true", "delay.expose_realised=true",
                   "run_tag=paoracle"], "delay": "episode", "aug": 1, "tag": "paoracle"},
}
ENV_IDS = {"pendulum": "Pendulum-v1"}
RECURSIVE = {"trm", "trm_small", "trm_h1", "trm_h6", "trm_hier6", "trm_singlez", "trm_carryz"}
STATEFUL = {"trm_carryz", "lstm_carry"}


def _train_job(stage, results, model, cond, seed, env="pendulum", arch="ddpg", episodes=200, walltime=None, extra=()):
    c = COND[cond]
    name = f"p3{stage}_{model}_{cond}_s{seed}"
    aug_tag = "" if c["delay"] == "none" else f"_aug{c['aug']}"
    tag = f"_{c['tag']}" if c["tag"] else ""
    result = f"{results}/{ENV_IDS[env]}_{arch}_{model}_{c['delay']}{aug_tag}{tag}_seed{seed}.json"
    if walltime is None:
        # Phase 2 GPU runs: <=82 min for 92k steps on any TRM variant; Pendulum is 40k steps.
        # Stateful trunks add a critic forward per env step.
        # Measured on Stage 1 (A100): trm_small ~31 min; all Pendulum arms fit the 2 h tier, which
        # reaches every GPU node (4-6 h requests queue for group-owned nodes only).
        walltime = "01:59:00"
    ov = [f"arch={arch}", f"model={model}", f"env={env}", *c["ov"], f"arch.total_episodes={episodes}",
          f"seed={seed}", "wandb.enabled=false", "print_stat_every=1", "record_every=1000", "eval_every=10",
          f"results_folder={P3_ROOT / results}", f"model_folder={P3_ROOT / 'models'}",
          f"video_folder={P3_ROOT / 'videos'}", "track_step_history=true",
          f"hydra.run.dir={LOGS / 'hydra' / name}", *extra]
    return {"name": name, "stage": stage, "kind": "train", "group": f"{model}|{cond}", "model": model,
            "cond": cond, "seed": seed, "episodes": episodes, "walltime": walltime,
            "cmd": "python3 -u main.py " + " ".join(ov), "result": str(P3_ROOT / result)}


def manifest(stage: str, mlp: str = "dnn") -> list[dict]:
    if stage == "s0h":   # S0.1 headroom probe (+ amendment arm A4 = stoch4-noaug)
        return [_train_job("s0h", "results_p3s0", "dnn", c, s)
                for c in ["fixed1-aug", "stoch4-aug", "stoch4-oracle", "stoch4-noaug"] for s in SEEDS]
    if stage == "s0b":   # S0.2 checkpoint battery over the 102 banked Phase 1-2 runs
        n = 12
        return [{"name": f"p3s0b_battery_{i:02d}", "stage": "s0b", "kind": "battery", "group": "battery",
                 "shard": i, "nshards": n, "walltime": "04:00:00",
                 "cmd": (f"python3 -u phase3/w0_eval.py --list {P3_ROOT / 'phase3' / 'battery_list.txt'} "
                         f"--shard {i} --nshards {n} --models {P3_ROOT / 'models'} "
                         f"--out {P3_ROOT / 'results_p3s0_battery'}"),
                 "result": None} for i in range(n)]
    if stage == "s2x":   # supplementary latent-delay test (G2s)
        return [_train_job("s2x", "results_p3s2x", m, c, s)
                for m, c in [("dnn", "C6"), ("dnn", "C6o"), ("trm", "C6"), ("trm_carryz", "C6"), ("lstm_carry", "C6")]
                for s in SEEDS]
    if stage == "s5b":   # battery over every Phase 3 checkpoint (Stages 0.1, 1, 2), for the synthesis
        n = 12
        return [{"name": f"p3s5b_battery_{i:02d}", "stage": "s5b", "kind": "battery", "group": "battery",
                 "shard": i, "nshards": n, "walltime": "01:59:00",
                 "cmd": (f"python3 -u phase3/w0_eval.py --list {P3_ROOT / 'phase3' / 'battery2_list.txt'} "
                         f"--shard {i} --nshards {n} --models {P3_ROOT / 'models'} "
                         f"--out {P3_ROOT / 'results_p3_battery2'}"),
                 "result": None} for i in range(n)]
    if stage in ("s5c", "s5d"):   # battery top-ups: s5c = the 6 Stage 1 dnn checkpoints whose results basenames
        # collide with Stage 0's arm A1 (same filename, so s5b skipped them); s5d = the latent-delay checkpoints
        lst = {"s5c": "battery3_list.txt", "s5d": "battery4_list.txt"}[stage]
        out = {"s5c": "results_p3_battery3", "s5d": "results_p3_battery4"}[stage]
        n = 1 if stage == "s5c" else 5
        return [{"name": f"p3{stage}_battery" + (f"_{i:02d}" if n > 1 else ""), "stage": stage, "kind": "battery",
                 "group": "battery", "walltime": "01:59:00",
                 "cmd": (f"python3 -u phase3/w0_eval.py --list {P3_ROOT / 'phase3' / lst} --shard {i} --nshards {n} "
                         f"--models {P3_ROOT / 'models'} --out {P3_ROOT / out}"), "result": None} for i in range(n)]
    if stage == "s0l":   # S0.2 latency benchmark
        return [{"name": "p3s0l_latency", "stage": "s0l", "kind": "latency", "group": "latency",
                 "walltime": "03:00:00",
                 "cmd": f"OMP_NUM_THREADS=1 python3 -u phase3/bench_latency.py --out {P3_ROOT / 'results_p3s0_latency' / 'latency.json'}",
                 "result": str(P3_ROOT / "results_p3s0_latency" / "latency.json")}]
    if stage == "s1":    # Stage 1 capacity x depth factorial
        return [_train_job("s1", "results_p3s1", m, c, s)
                for m in ["dnn", "trm_small", "dnn_large", "trm", "trm_h1", "trm_h6"]
                for c in ["none", "fixed1-aug"] for s in SEEDS]
    if stage == "s2":    # Stage 2 uncertainty x memory; `mlp` is the capacity level G1 picks
        return [_train_job("s2", "results_p3s2", m, c, s)
                for m in [mlp, "trm", "trm_carryz", "lstm_carry"] for c in ["C1", "C3", "C4"] for s in SEEDS]
    if stage == "smoke2":  # the post-smoke code (Carry-Z, seeding) on GPU
        jobs = [_train_job("smoke2", "results_p3smoke2", m, c, 1, episodes=3, walltime="01:00:00",
                           extra=["arch.warmup_steps=200"])
                for m, c in [("dnn", "fixed1-aug"), ("trm", "fixed1-aug"), ("trm_carryz", "C4"), ("lstm_carry", "C3")]]
        res = P3_ROOT / "results_p3smoke2"
        cmd = " && ".join(j["cmd"] for j in jobs) + (
            f" && python3 -u phase3/w0_eval.py --results {res}/Pendulum-v1_ddpg_trm_carryz_stochastic_aug0_pa_seed1.json "
            f"{res}/Pendulum-v1_ddpg_lstm_carry_stochastic_aug1_pa_seed1.json --quick --force "
            f"--models {P3_ROOT / 'models'} --out {res / 'battery'}"
            f" && OMP_NUM_THREADS=1 python3 -u phase3/bench_latency.py --models trm_carryz lstm_carry --n 50 "
            f"--out {res / 'latency.json'}")
        return [{"name": "p3smoke2_all", "stage": "smoke2", "kind": "smoke", "group": "smoke", "walltime": "01:30:00",
                 "cmd": cmd, "result": str(res / "latency.json")}]
    if stage == "smoke":
        jobs = [_train_job("smoke", "results_p3smoke", m, c, 1, episodes=3, walltime="01:00:00",
                           extra=["arch.warmup_steps=200"])
                for m, c in [("dnn", "fixed1-aug"), ("dnn", "stoch4-aug"), ("dnn", "stoch4-oracle"),
                             ("dnn", "stoch4-noaug"), ("trm_small", "none"), ("dnn_large", "fixed1-aug"),
                             ("trm_h1", "fixed1-aug"), ("trm_h6", "none")]]
        # One GPU job runs every smoke command in sequence.
        cmd = " && ".join(j["cmd"] for j in jobs) + (
            f" && python3 -u phase3/w0_eval.py --list {P3_ROOT / 'phase3' / 'smoke_list.txt'} --quick --force "
            f"--models {P3_ROOT / 'models'} --out {P3_ROOT / 'results_p3smoke' / 'battery'}"
            f" && OMP_NUM_THREADS=1 python3 -u phase3/bench_latency.py --models dnn trm --n 50 "
            f"--out {P3_ROOT / 'results_p3smoke' / 'latency.json'}")
        return [{"name": "p3smoke_all", "stage": "smoke", "kind": "smoke", "group": "smoke", "walltime": "01:30:00",
                 "cmd": cmd, "result": str(P3_ROOT / "results_p3smoke" / "latency.json")}]
    raise SystemExit(f"unknown stage {stage!r}")


# gpu_rp6000=False: the RTX PRO 6000 Blackwell node (k207, sm_120) has no kernels in the
# venv's torch 2.5.1+cu121 build; every other Katana GPU model (V100/A100/L40S/H100/H200) does.
PBS = """#!/bin/bash
#PBS -N {name}
#PBS -l select=1:ncpus=2:mem=8gb:ngpus=1:gpu_rp6000=False
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
    script.write_text(PBS.format(name=job["name"], walltime=job["walltime"], pbs_out=d / f"{job['name']}.o",
                                 venv=VENV, code=code, log=LOGS / f"{job['name']}.log", cmd=job["cmd"]))
    if dry:
        return None
    out = subprocess.run(["qsub", str(script)], capture_output=True, text=True)
    if out.returncode != 0:
        print(f"qsub failed for {job['name']}: {out.stderr.strip()}", file=sys.stderr)
        return None
    return out.stdout.strip()


def cmd_submit(stage: str, dry: bool, mlp: str = "dnn") -> None:
    code = P3_CODE / stage
    if not (code / "main.py").exists():
        raise SystemExit(f"no code snapshot at {code}: sync one first")
    done = {e["name"] for e in _ledger()}
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for job in manifest(stage, mlp):
        if job["name"] in done:
            continue
        jid = _qsub(job, code, dry)
        if dry:
            print("would submit", job["name"])
            continue
        if jid:
            rec = {**job, "jobid": jid, "code": str(code), "submitted": dt.datetime.now().isoformat(timespec="seconds")}
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n += 1
            print(f"{job['name']} {jid}")
    if n and not dry and not stage.startswith("smoke"):
        (code / ".frozen").write_text(dt.datetime.now().isoformat(timespec="seconds") + "\n")
    print(f"submitted {n} jobs for stage {stage}")


def cmd_resubmit(name: str, walltime: str | None = None) -> None:
    recs = [e for e in _ledger() if e["name"] == name]
    if not recs:
        raise SystemExit(f"{name} not in ledger")
    job = dict(recs[-1])
    if walltime:
        # <=2 h routes to the mechgpu2 tier, which reaches every GPU node rather than
        # only group-owned ones (the site hook forbids qalter-ing a queued job's walltime).
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
        out.update(episode=int(ep), total=int(tot), ret=float(ret), avg100=float(avg))
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
        for k in ["final_avg100_return", "wall_clock_seconds", "param_count"]:
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
    ledger = _ledger()
    latest = {}
    for e in ledger:
        latest[e["name"]] = e          # a resubmission supersedes the earlier record
    q = _qstat([e["jobid"] for e in latest.values()])
    jobs = []
    for name, e in latest.items():
        qi = q.get(e["jobid"], {})
        st = qi.get("job_state")
        exit_status = qi.get("Exit_status")
        res = _result_summary(e.get("result"), e["kind"])
        prog = _progress(name)
        if st in ("Q", "H", "W", "T"):
            state = "queued" if st != "H" else "held"
        elif st in ("R", "E", "B"):
            state = "running"
        elif st == "F" or (not qi and ("exit_line" in prog)):
            ok = (exit_status in (0, None)) and (res is not None or e["kind"] == "battery")
            if e["kind"] == "battery":
                ok = exit_status == 0 or prog.get("exit_line", "").startswith("exit 0")
            state = "done" if ok else "failed"
        elif res is not None:
            state = "done"
        else:
            state = "unknown"
        ru = qi.get("resources_used", {})
        jobs.append({
            "name": name, "stage": e["stage"], "kind": e["kind"], "group": e["group"],
            "model": e.get("model"), "cond": e.get("cond"), "seed": e.get("seed"),
            "jobid": e["jobid"].split(".")[0], "state": state, "pbs_state": st, "exit": exit_status,
            "submitted": e["submitted"], "walltime_req": e["walltime"],
            "walltime_used": ru.get("walltime"), "host": (qi.get("exec_host") or "").split("/")[0] or None,
            "stime": qi.get("stime"), "qtime": qi.get("qtime"), "mtime": qi.get("mtime"),
            "queue": qi.get("queue"), "comment": qi.get("comment"),
            "progress": {k: v for k, v in prog.items() if k != "error"},
            "error": prog.get("error") if state == "failed" else None,
            "result": res,
        })
    print(json.dumps({"polled_at": dt.datetime.now().isoformat(timespec="seconds"), "jobs": jobs}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("stage"); s.add_argument("--dry-run", action="store_true")
    s.add_argument("--mlp", default="dnn", help="Stage 2 MLP capacity level (dnn or dnn_large), set by G1")
    sub.add_parser("status")
    r = sub.add_parser("resubmit"); r.add_argument("name"); r.add_argument("--walltime")
    m = sub.add_parser("manifest"); m.add_argument("stage"); m.add_argument("--mlp", default="dnn")
    args = ap.parse_args()
    if args.cmd == "submit":
        cmd_submit(args.stage, args.dry_run, args.mlp)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "resubmit":
        cmd_resubmit(args.name, args.walltime)
    elif args.cmd == "manifest":
        print(json.dumps(manifest(args.stage, args.mlp), indent=1))


if __name__ == "__main__":
    main()
