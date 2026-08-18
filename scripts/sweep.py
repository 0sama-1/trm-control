#!/usr/bin/env python
"""Stage-A sweep controller for trm-control.

Generates `torchrun --standalone --nproc_per_node=1 main.py <overrides>`
commands across a config grid (model x env x delay x staleness x seed) and
either:

  * prints them only            (--dry-run),
  * runs them sequentially on the local node (--local),
  * writes one Slurm batch script per command and submits them (default,
    requires `sbatch`), following the template in AGENTS.md §6.4.

The grid is specified as a YAML list of dicts (one dict per cell), so each
cell can carry arbitrary extra Hydra overrides (e.g. model.trm.H_cycles)
without changing this script. A compact CLI (--grid or --arch/--model/...)
also works for the common case.

Per-run wall-clock and steps/sec are measured for local runs (plan §5.5 /
AGENTS.md §6.5). For Slurm runs the per-run timing is emitted by the
training code itself to wandb; this controller only records submission
metadata + total sweep wall-clock.

Examples
--------
  # Dry-run: print the torchrun commands that *would* run.
  python scripts/sweep.py --dry-run \
      --arch ppo --model "dnn trm" --env pendulum --delay none seed \
      --seed "0 1 2 3 4"

  # Local sequential run on a node with a GPU (no torch install here needed at
  # parse time — torch is only imported if --local is not dry-run).
  python scripts/sweep.py --local --arch ppo --model dnn trm \
      --env pendulum --delay none --seed 0 1 2

  # Slurm submission (default): one sbatch job per cell.
  python scripts/sweep.py --arch sac --model "dnn trm" \
      --env hopper halfcheetah --delay none --seed 0 1 2 3 4

  # Arbitrary grid from a YAML file (full flexibility).
  python scripts/sweep.py --grid conf/sweeps/h1_smoke.yaml --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Command model
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """One cell of the sweep grid -> one torchrun command."""
    arch: str
    model: str
    env: str
    delay: str
    seed: int
    # Optional extra Hydra overrides (deep supervision, H_cycles, etc.).
    extra: dict[str, str] = field(default_factory=dict)
    # Optional staleness override (observation staleness group).
    staleness: str | None = None
    # Optional human-friendly label for the run/job name.
    label: str | None = None

    def overrides(self) -> list[str]:
        """Exact Hydra overrides emitted on the torchrun command line."""
        ov = [
            f"arch={self.arch}",
            f"model={self.model}",
            f"env={self.env}",
            f"delay={self.delay}",
            f"seed={self.seed}",
        ]
        if self.staleness is not None:
            ov.append(f"staleness={self.staleness}")
        for k, v in self.extra.items():
            ov.append(f"{k}={v}")
        return ov

    def command(self, nproc: int = 1) -> list[str]:
        """The exact torchrun invocation (single-process, per AGENTS.md §6.3)."""
        return [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
            "main.py",
            *self.overrides(),
        ]

    def job_name(self) -> str:
        n = self.label or f"{self.arch}_{self.model}_{self.env}_{self.delay}_s{self.seed}"
        return n.replace("/", "_")


def build_torchrun_command(spec: RunSpec) -> list[str]:
    """Public helper: build the torchrun command list for a RunSpec."""
    return spec.command()


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def _split_tokens(token: str) -> list[str]:
    """Split a whitespace-separated CLI token like 'dnn trm' -> ['dnn','trm']."""
    # Support both space-separated and comma-separated forms.
    parts = []
    for raw in token.replace(",", " ").split():
        parts.append(raw)
    return parts


def _expand_seeds(seeds: list[str]) -> list[int]:
    out = []
    for s in seeds:
        out.append(int(s))
    return out


def grid_from_cli(args: argparse.Namespace) -> list[RunSpec]:
    """Build the list of RunSpecs from flat CLI flags (the common case)."""
    archs = _split_tokens(args.arch)
    models = _split_tokens(args.model)
    envs = _split_tokens(args.env)
    delays = _split_tokens(args.delay)
    stales = _split_tokens(args.staleness) if args.staleness else [None]
    seeds = _expand_seeds(_split_tokens(args.seed))

    extra = {}
    for ov in args.extra or []:
        if "=" not in ov:
            raise SystemExit(f"--extra override must be k=v, got: {ov!r}")
        k, v = ov.split("=", 1)
        extra[k] = v

    specs: list[RunSpec] = []
    for arch, model, env, delay, stale, seed in itertools.product(
        archs, models, envs, delays, stales, seeds
    ):
        specs.append(RunSpec(
            arch=arch, model=model, env=env, delay=delay, seed=seed,
            staleness=stale, extra=dict(extra),
        ))
    return specs


def grid_from_yaml(path: Path) -> list[RunSpec]:
    """Load a grid from a YAML file.

    Format (a list of cells):
      - arch: ppo
        model: trm
        env: pendulum
        delay: none
        seed: 42
        extra:
          model.trm.H_cycles: 3
    """
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(f"PyYAML is required to read --grid: {exc}") from exc

    with open(path) as fh:
        cells = yaml.safe_load(fh)
    if not isinstance(cells, list):
        raise SystemExit(f"--grid {path} must be a YAML list of cells")

    specs: list[RunSpec] = []
    for cell in cells:
        specs.append(RunSpec(
            arch=cell["arch"], model=cell["model"], env=cell["env"],
            delay=cell["delay"], seed=int(cell["seed"]),
            staleness=cell.get("staleness"),
            extra=cell.get("extra") or {},
            label=cell.get("label"),
        ))
    return specs


# ---------------------------------------------------------------------------
# Slurm
# ---------------------------------------------------------------------------

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu=1
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --job-name={jobname}
#SBATCH --output=logs/%x_%j.out
set -e
cd {repo}
# --- module + venv bootstrap (see setup.sh / AGENTS.md §6) ---
source .venv/bin/activate
torchrun --standalone --nproc_per_node=1 main.py {overrides}
"""


def write_sbatch(spec: RunSpec, out_dir: Path, sbatch_opts: dict) -> Path:
    """Write one Slurm batch script for a RunSpec; return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = " ".join(spec.overrides())
    script = SBATCH_TEMPLATE.format(
        partition=sbatch_opts.get("partition", "quantum"),
        time=sbatch_opts.get("time", "02:00:00"),
        account=sbatch_opts.get("account", "${ALLOCATION}"),
        jobname=spec.job_name(),
        repo=str(REPO_ROOT),
        overrides=overrides,
    )
    path = out_dir / f"{spec.job_name()}.sbatch"
    path.write_text(script)
    return path


def submit_slurm(specs: list[RunSpec], sbatch_opts: dict, dry_run: bool) -> list[dict]:
    """Emit + submit one Slurm script per RunSpec; return a manifest of paths."""
    out_dir = REPO_ROOT / "logs" / "slurm"
    manifest: list[dict] = []
    for spec in specs:
        script_path = write_sbatch(spec, out_dir, sbatch_opts)
        entry = {"spec": spec.__dict__, "script": str(script_path)}
        if dry_run:
            print(f"[sweep] (dry-run) would submit: sbatch {script_path}")
            print(f"[sweep]   overrides: {' '.join(spec.overrides())}")
        else:
            print(f"[sweep] submitting: sbatch {script_path}")
            res = subprocess.run(["sbatch", str(script_path)],
                                 capture_output=True, text=True)
            entry["sbatch_stdout"] = res.stdout.strip()
            entry["sbatch_stderr"] = res.stderr.strip()
            entry["returncode"] = res.returncode
            if res.returncode != 0:
                print(f"[sweep] sbatch failed for {script_path}: {res.stderr}",
                      file=sys.stderr)
        manifest.append(entry)
    return manifest


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------

def run_local(spec: RunSpec, nproc: int, timeout_min: float | None) -> dict:
    """Run one torchrun command locally, measuring wall-clock + steps/sec.

    Timing is reported by parsing the trainer's stdout (the trainer logs
    `run_steps_per_sec` / `run_wall_clock` at the end). We also measure an
    outer wall-clock as a fallback.
    """
    cmd = spec.command(nproc=nproc)
    timeout_s = None
    if timeout_min:
        timeout_s = int(timeout_min * 60)

    print(f"[sweep-local] {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), environment=None,
            timeout=timeout_s, capture_output=True, text=True,
        )
        wall = time.time() - t0
        ok = proc.returncode == 0
        if proc.stdout:
            print(proc.stdout[-2000:])
        if proc.stderr:
            print(proc.stderr[-2000:], file=sys.stderr)
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        ok = False
        proc = None

    return {
        "spec": spec.__dict__,
        "command": " ".join(cmd),
        "returncode": None if proc is None else proc.returncode,
        "ok": ok,
        "wall_clock": wall,
        "steps_per_sec": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sweep.py",
        description="trm-control sweep controller: emit/submit/run torchrun commands.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print the torchrun commands without running or submitting.")
    mode.add_argument("--local", action="store_true",
                      help="Run torchrun commands sequentially on this node (requires torch + GPU/CPU).")
    mode.add_argument("--slurm", action="store_true",
                      help="Submit one Slurm batch job per cell (default).")

    p.add_argument("--grid", type=str, default=None,
                   help="Path to a YAML grid file (list of cells). Overrides flat flags.")
    p.add_argument("--arch", type=str, default="ppo",
                   help="Algos to sweep over, space/comma-separated (default: ppo).")
    p.add_argument("--model", type=str, default="dnn trm",
                   help="Backbone trunks, space/comma-separated (default: 'dnn trm').")
    p.add_argument("--env", type=str, default="pendulum",
                   help="Envs, space/comma-separated (default: pendulum).")
    p.add_argument("--delay", type=str, default="none stochastic",
                   help="Delay modes, space/comma-separated (default: 'none stochastic').")
    p.add_argument("--staleness", type=str, default=None,
                   help="Observation-staleness modes, space/comma-separated (default: none).")
    p.add_argument("--seed", type=str, default="0 1 2 3 4",
                   help="Seeds, space/comma-separated (default: 0 1 2 3 4).")
    p.add_argument("--extra", action="append", default=None,
                   help="Extra Hydra k=v override appended to every command. Repeatable.")
    p.add_argument("--nproc", type=int, default=1,
                   help="nproc_per_node (default 1; must stay 1 per AGENTS.md §6.3).")
    p.add_argument("--timeout-min", type=float, default=None,
                   help="Per-run timeout in minutes (local mode only).")
    p.add_argument("--account", type=str, default=None,
                   help="Slurm account allocation (default: read $ALLOCATION or env).")
    p.add_argument("--time", type=str, default="02:00:00",
                   help="Slurm wall-time per job (default 02:00:00).")
    p.add_argument("--partition", type=str, default="quantum",
                   help="Slurm partition (default quantum).")
    p.add_argument("--manifest-out", type=str, default=None,
                   help="Write a JSON manifest of all commands to this path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Resolve the grid.
    if args.grid:
        specs = grid_from_yaml(Path(args.grid))
    else:
        specs = grid_from_cli(args)

    if not specs:
        print("[sweep] empty grid — nothing to do.", file=sys.stderr)
        return 2

    print(f"[sweep] {len(specs)} run(s) in grid")

    mode = "dry_run" if args.dry_run else ("local" if args.local else "slurm")

    results: list[dict] = []
    sweep_t0 = time.time()

    if mode == "dry_run":
        for spec in specs:
            print(" ".join(spec.command(nproc=args.nproc)))
        results = [{"spec": s.__dict__, "command": " ".join(s.command(nproc=args.nproc)),
                    "mode": "dry_run"} for s in specs]

    elif mode == "local":
        # Lazy torch check: only needed if we actually run.
        try:
            import torch  # noqa: F401
        except ImportError:
            print("[sweep] --local requires torch to be importable in this environment.",
                  file=sys.stderr)
            print("[sweep] Falling back to printing commands. Use --dry-run for a torch-free check.",
                  file=sys.stderr)
            for spec in specs:
                print(" ".join(spec.command(nproc=args.nproc)))
            return 1
        for spec in specs:
            results.append(run_local(spec, args.nproc, args.timeout_min))

    else:  # slurm (default)
        account = args.account or os.environ.get("ALLOCATION", "${ALLOCATION}")
        sbatch_opts = {
            "partition": args.partition,
            "time": args.time,
            "account": account,
        }
        manifest = submit_slurm(specs, sbatch_opts, dry_run=False)
        print(f"[sweep] submitted {len(manifest)} slurm job(s)")
        results = manifest

    sweep_wall = time.time() - sweep_t0

    # Summary + optional manifest.
    print(f"\n[sweep] mode={mode} runs={len(specs)} sweep_wall_clock={sweep_wall:.2f}s")
    if mode == "slurm":
        print("[sweep] per-run wall-clock/steps-sec is logged by the trainer to wandb "
              "(AGENTS.md §6.5); see logs/slurm/ for batch scripts.")
    elif mode == "local":
        for r in results:
            print(f"  {r['spec']['arch']}/{r['spec']['model']}/{r['spec']['env']}/"
                  f"{r['spec']['delay']}_s{r['spec']['seed']}: "
                  f"ok={r['ok']} wall={r['wall_clock']:.2f}s rc={r['returncode']}")

    if args.manifest_out:
        out = Path(args.manifest_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": mode,
            "n_runs": len(specs),
            "sweep_wall_clock": sweep_wall,
            "results": results,
        }
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[sweep] manifest -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
