"""Phase 4 code verification (plan §5): run before any Stage A / B0 job.

  python verify_code.py grad                       gradient-hook test: cell applications receiving gradient
  python verify_code.py run --src DIR --out F      short training runs of every existing config under the code in DIR
  python verify_code.py compare A B                bit-identity of two `run` outputs

`run` trains each config for 2 Pendulum episodes (warmup 100 steps, so ~300 updates)
on CPU with fixed seeds and records the episode history and a hash of every final
weight tensor. Running it on the pre-Phase-4 snapshot and on the current tree and
comparing the two shows the new flags leave every existing code path unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

# Every trunk and algorithm combination Phases 1-3 used, plus the Phase 4
# step-budget flag on an existing config (it changes the episode count, so it is
# checked separately: the default path must ignore it).
EXISTING = [
    (algo, model, cond)
    for algo in ["ddpg", "td3"]
    for model in ["dnn", "dnn_large", "trm", "trm_small", "trm_h1", "trm_h6", "trm_hier6", "trm_singlez",
                  "transformer", "lstm", "trm_carryz", "lstm_carry"]
    for cond in ["fixed1"]
] + [("ddpg", m, c) for m in ["dnn", "trm", "trm_carryz"] for c in ["none", "C4"]]
COND = {
    "none": ["delay=none", "augment_obs=true"],
    "fixed1": ["delay=fixed", "delay.value=1", "augment_obs=true"],
    "C4": ["delay=stochastic_u4", "augment_obs=false", "prev_action_obs=true"],
    "C1pa": ["delay=fixed", "delay.value=1", "augment_obs=true", "prev_action_obs=true"],
}


def _cfg(conf_dir: Path, algo: str, model: str, cond: str, extra=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(str(conf_dir), version_base=None):
        return compose("config", overrides=[
            f"arch={algo}", f"model={model}", "env=pendulum", *COND[cond], "arch.total_episodes=2",
            "arch.warmup_steps=100", "seed=3", "wandb.enabled=false", "print_stat_every=1000",
            "record_every=1000", "eval_every=1", "track_step_history=false", *extra])


def _train(src: Path, cfg, tmp: Path) -> dict:
    import torch
    torch.set_num_threads(1)
    cfg.model_folder, cfg.video_folder, cfg.results_folder = str(tmp / "m"), str(tmp / "v"), str(tmp / "r")
    if cfg.arch.name == "ddpg":
        from algos.ddpg import DDPGTrainer as T
    else:
        from algos.td3 import TD3Trainer as T
    tr = T(cfg)
    tr.train()
    res = json.loads(next((tmp / "r").glob("*.json")).read_text())
    sd = torch.load(tr.model_path, map_location="cpu")
    h = hashlib.sha256()
    for net in sorted(k for k in sd if not k.endswith("_opt") and k != "episode_count"):
        for name, t in sorted(sd[net].items()):
            h.update(f"{net}.{name}".encode()); h.update(t.cpu().numpy().tobytes())
    hist = [{k: v for k, v in row.items() if not k.endswith("_grad_norms")} for row in res["episode_history"]]
    return {"weights_sha256": h.hexdigest(), "episode_history": hist, "param_count": res["param_count"]}


def cmd_run(src: Path, out: Path, which: str, shard: int = 0, nshards: int = 1) -> None:
    sys.path.insert(0, str(src / "src"))
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    results = {}
    jobs = list(EXISTING) if which == "existing" else []
    if which == "mlp_pa":  # Stage A MLP reuse: the exact Phase 3 Stage 2 dnn configurations
        jobs = [("ddpg", "dnn", "C1pa"), ("ddpg", "dnn", "C4")]
    if which == "flags":   # current tree only: the new paths at least run, and inf clipping is a no-op
        jobs = [("ddpg", "trm", "fixed1", ("+arch.actor_grad_clip=inf", "+arch.critic_grad_clip=inf")),
                ("ddpg", "trm_fullgrad", "fixed1", ()), ("ddpg", "trm_fullgrad_ds", "fixed1", ()),
                ("ddpg", "trm_fullgrad_ds", "C4", ("+arch.actor_grad_clip=1.0", "+arch.critic_grad_clip=1.0")),
                ("td3", "dnn", "fixed1", ("+arch.total_steps=500",))]
    for job in jobs[shard::nshards]:
        algo, model, cond, *extra = job
        extra = extra[0] if extra else ()
        key = f"{algo}|{model}|{cond}" + ("|" + ",".join(extra) if extra else "")
        with tempfile.TemporaryDirectory() as t:
            results[key] = _train(src, _cfg(src / "conf", algo, model, cond, extra), Path(t))
        print(key, results[key]["weights_sha256"][:12], flush=True)
    out.write_text(json.dumps(results, indent=1))


def _load(p: Path) -> dict:
    """A `run` output, or the merge of its shards (<stem>.shard*.json)."""
    if p.exists():
        return json.loads(p.read_text())
    out = {}
    for f in sorted(p.parent.glob(f"{p.stem}.shard*.json")):
        out.update(json.loads(f.read_text()))
    return out


def cmd_compare(a: Path, b: Path) -> None:
    A, B = _load(a), _load(b)
    bad = [k for k in A if A[k] != B.get(k)]
    for k in A:
        print(("IDENTICAL " if k not in bad else "DIFFERENT ") + k)
    print(f"{len(A) - len(bad)}/{len(A)} configs bit-identical")
    sys.exit(1 if bad or set(A) != set(B) else 0)


def cmd_grad() -> None:
    """Count cell applications whose output receives gradient from a loss on z_H."""
    sys.path.insert(0, str(REPO / "src"))
    import torch
    from models.trm_control import TRMTrunk
    out = {}
    for label, kw in [("truncated (trm)", {}), ("full_grad", {"full_grad": True}),
                      ("full_grad + deep_sup", {"full_grad": True, "deep_sup": True})]:
        for H in (1, 3, 6):
            torch.manual_seed(0)
            trunk = TRMTrunk(in_dim=5, hidden_size=32, H_cycles=H, L_cycles=6, **kw)
            calls, got = [], []

            def hook(_m, _inp, output):
                idx = len(calls)
                calls.append(idx)
                if output.requires_grad:
                    output.register_hook(lambda g, i=idx: got.append(i) if g.abs().sum() > 0 else None)

            hd = trunk.reasoning.register_forward_hook(hook)
            x = torch.randn(4, 5)
            if kw.get("deep_sup"):
                loss = torch.stack([z.square().mean() for z in trunk.forward_cycles(x)]).sum()
            else:
                loss = trunk(x).square().mean()
            loss.backward()
            hd.remove()
            out[f"{label}, H={H}"] = {"applications": len(calls), "with_gradient": len(set(got))}
            print(f"{label:22s} H={H}: {len(set(got)):2d} of {len(calls):2d} cell applications receive gradient")
    exp = {"truncated (trm), H=3": 7, "full_grad, H=3": 21, "full_grad + deep_sup, H=3": 21}
    ok = all(out[k]["with_gradient"] == v and out[k]["applications"] == 21 for k, v in exp.items())
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("grad")
    r = sub.add_parser("run"); r.add_argument("--src", type=Path, default=REPO); r.add_argument("--out", type=Path)
    r.add_argument("--which", default="existing", choices=["existing", "flags", "mlp_pa"])
    r.add_argument("--shard", type=int, default=0); r.add_argument("--nshards", type=int, default=1)
    c = sub.add_parser("compare"); c.add_argument("a", type=Path); c.add_argument("b", type=Path)
    a = ap.parse_args()
    {"grad": cmd_grad, "run": lambda: cmd_run(a.src, a.out, a.which, a.shard, a.nshards), "compare": lambda: cmd_compare(a.a, a.b)}[a.cmd]()
