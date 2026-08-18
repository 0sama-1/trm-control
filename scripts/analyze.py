#!/usr/bin/env python
"""Stage-A analysis script for trm-control.

Reads a results CSV (one row per run) and produces:

  * a per-condition summary (mean, std, 95% bootstrap CI, n),
  * effect sizes (Cohen's d) and paired/2-sample t-tests between conditions,
  * a markdown report + a bar/figure under docs/plan/reports/.

The stats are **torch-free** (numpy + scipy only) so this runs anywhere,
including the login node. Per-run wall-clock/steps-per-sec columns are
forwarded into the summary when present (plan §5.5 / AGENTS.md §6.5).

CSV contract
------------
The CSV must contain at least a ``condition`` column and a ``return`` column.
Optionally it may contain ``arch``, ``model``, ``env``, ``delay``, ``seed``,
``wall_clock``, ``steps_per_sec`` columns. Extra override columns (e.g.
``H_cycles``) are folded into the condition label automatically.

Usage
-----
  # Analyse a CSV (wandb-exported or hand-written).
  python scripts/analyze.py results.csv

  # Custom output dir + figure.
  python scripts/analyze.py results.csv --out reports/my_run.md --fig reports/my_run.png

  # Generate a tiny synthetic CSV and analyse it (smoke test, torch-free).
  python scripts/analyze.py --synthetic

Columns expected in the CSV:
  condition, return, [arch, model, env, delay, seed, wall_clock, steps_per_sec, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "docs" / "plan" / "reports"

# Columns that describe a condition vs. metrics to aggregate.
CONDITION_COLS = ["arch", "model", "env", "delay", "staleness", "seed",
                  "H_cycles", "L_cycles", "h_backprop"]
METRIC_COLS = ["return", "wall_clock", "steps_per_sec", "avg100_return"]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, n_boot: int, conf: float,
                 rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `data`."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(data[0]), float(data[0]))
    means = np.mean(rng.choice(data, size=(n_boot, n), replace=True), axis=1)
    alpha = (1.0 - conf) / 2.0
    lo = float(np.percentile(means, 100.0 * alpha))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha)))
    return (lo, hi)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled-SD, two-sample). >0.2 small, >0.5 med, >0.8 large."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    """Paired t-test if `a` and `b` share the same seed set; else None.

    Returns (t_stat, p_value). Requires matched seeds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return None
    res = stats.ttest_rel(a, b)
    return (float(res.statistic), float(res.pvalue))


def condition_label(row: pd.Series) -> str:
    """Build a human-readable condition label from a row's metadata columns."""
    parts = []
    for col in CONDITION_COLS:
        if col in row.index and pd.notna(row[col]) and row[col] != "":
            parts.append(f"{col}={row[col]}")
    meta = "+".join(parts) if parts else "default"
    env = row.get("env", "")
    model = row.get("model", "")
    arch = row.get("arch", "")
    delay = row.get("delay", "")
    base = f"arch={arch},model={model},env={env},delay={delay}"
    if meta and meta != "default":
        base += f",{meta}"
    return base


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame, n_boot: int, conf: float, seed: int) -> pd.DataFrame:
    """Per-condition summary table."""
    rng = np.random.default_rng(seed)

    # Derive a stable condition column if 'condition' is missing.
    if "condition" not in df.columns:
        df = df.copy()
        df["condition"] = df.apply(condition_label, axis=1)

    rows = []
    for cond, grp in df.groupby("condition", sort=True):
        returns = grp["return"].to_numpy(dtype=float)
        n = len(returns)
        mean = float(np.mean(returns)) if n else float("nan")
        std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        lo, hi = bootstrap_ci(returns, n_boot, conf, rng)
        row = {
            "condition": cond,
            "n": n,
            "mean_return": mean,
            "std_return": std,
            "sem": sem,
            "ci95_lo": lo,
            "ci95_hi": hi,
        }
        # Forward timing metrics if present.
        for m in METRIC_COLS:
            if m in grp.columns:
                row[f"mean_{m}"] = float(np.mean(grp[m].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_contrasts(df: pd.DataFrame, n_boot: int, conf: float,
                       seed: int) -> pd.DataFrame:
    """Cohen's d + paired t-test between every pair of conditions.

    Paired tests require aligned seeds within each condition; if seeds don't
    match, a 2-sample t-test is reported instead (with method noted).
    """
    if "condition" not in df.columns:
        df = df.copy()
        df["condition"] = df.apply(condition_label, axis=1)

    conds = sorted(df["condition"].unique())
    seed_col = "seed" if "seed" in df.columns else None
    rows = []
    for a, b in itertools_pairs(conds):
        ga = df[df["condition"] == a]
        gb = df[df["condition"] == b]
        ra = ga["return"].to_numpy(dtype=float)
        rb = gb["return"].to_numpy(dtype=float)
        d = cohen_d(ra, rb)
        method = "none"
        tstat = float("nan")
        pval = float("nan")
        if seed_col and len(ra) == len(rb) and len(ra) >= 2:
            sa = set(ga[seed_col].tolist())
            sb = set(gb[seed_col].tolist())
            if sa == sb:
                res = stats.ttest_rel(ra, rb)
                tstat = float(res.statistic)
                pval = float(res.pvalue)
                method = "paired"
            else:
                res = stats.ttest_ind(ra, rb, equal_var=False)
                tstat = float(res.statistic)
                pval = float(res.pvalue)
                method = "welch"
        elif len(ra) >= 2 and len(rb) >= 2:
            res = stats.ttest_ind(ra, rb, equal_var=False)
            tstat = float(res.statistic)
            pval = float(res.pvalue)
            method = "welch"
        rows.append({
            "condition_a": a,
            "condition_b": b,
            "cohens_d": d,
            "t_stat": tstat,
            "p_value": pval,
            "test": method,
        })
    return pd.DataFrame(rows)


def itertools_pairs(items: list) -> list[tuple[str, str]]:
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append((items[i], items[j]))
    return pairs


# ---------------------------------------------------------------------------
# Reporting / figures
# ---------------------------------------------------------------------------

def fmt_val(v, floatfmt: str = ".4f") -> str:
    """Format a scalar for markdown (no tabulate dependency)."""
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return format(v, floatfmt)
    if isinstance(v, (np.floating,)):
        f = float(v)
        if np.isnan(f):
            return "nan"
        return format(f, floatfmt)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v).replace("|", "\\|")


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a github markdown table (stdlib + numpy only)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = [fmt_val(row[c], floatfmt) for c in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_markdown(summary: pd.DataFrame, contrasts: pd.DataFrame,
                   out_path: Path, meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# trm-control analysis report\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    if meta:
        lines.append("## Metadata\n\n")
        for k, v in meta.items():
            lines.append(f"- **{k}**: {v}\n")
        lines.append("\n")

    lines.append("## Per-condition summary\n\n")
    lines.append(df_to_markdown(summary, ".4f"))
    lines.append("\n\n")

    if not contrasts.empty:
        lines.append("## Pairwise contrasts\n\n")
        lines.append("Cohen's d (positive = condition_a > condition_b); "
                     "paired t-test where seeds match, Welch otherwise.\n\n")
        lines.append(df_to_markdown(contrasts, ".4f"))
        lines.append("\n\n")

    lines.append("## Per-run wall-clock / throughput (plan §5.5)\n\n")
    if "mean_wall_clock" in summary.columns and "mean_steps_per_sec" in summary.columns:
        tc = summary[["condition", "n", "mean_wall_clock", "mean_steps_per_sec"]].copy()
        tc.columns = ["condition", "n", "mean_wall_clock_s", "mean_steps_per_sec"]
        lines.append(df_to_markdown(tc, ".4f"))
        lines.append("\n\n")
    else:
        lines.append("_No per-run timing columns found in the input data._\n\n")

    # Headline conditions if return column present.
    if "mean_return" in summary.columns and len(summary) > 0:
        top = summary.sort_values("mean_return", ascending=False).iloc[0]
        lines.append("## Headline\n\n")
        lines.append(f"Best mean return: **{top['mean_return']:.4f}** "
                     f"({top['condition']}, n={int(top['n'])}, "
                     f"95% CI [{top['ci95_lo']:.4f}, {top['ci95_hi']:.4f}]).\n\n")

    out_path.write_text("".join(lines))


def write_figure(summary: pd.DataFrame, fig_path: Path) -> None:
    """Bar chart of mean return with 95% CI error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.empty or "mean_return" not in summary.columns:
        return
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(summary) * 0.9), 4.5))
    labels = [c.replace(",model=", "\nmodel=").replace(",arch=", "\narch=")
              for c in summary["condition"]]
    means = summary["mean_return"].to_numpy(dtype=float)
    lo = summary["ci95_lo"].to_numpy(dtype=float)
    hi = summary["ci95_hi"].to_numpy(dtype=float)
    yerr = np.vstack([np.nan_to_num(means - lo), np.nan_to_num(hi - means)])
    ax.bar(range(len(means)), means, yerr=yerr, capsize=5,
           color="#4C72B0", alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean return (±95% bootstrap CI)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("trm-control: mean return per condition")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic CSV (torch-free smoke fixture)
# ---------------------------------------------------------------------------

def make_synthetic_csv(path: Path) -> None:
    """Generate a tiny synthetic CSV with two conditions x 10 seeds."""
    rng = np.random.default_rng(0)
    rows = []
    for cond, mu, sigma in [("arch=ppo,model=trm,env=pendulum,delay=none",
                             -180.0, 20.0),
                            ("arch=ppo,model=dnn,env=pendulum,delay=none",
                             -200.0, 25.0),
                            ("arch=ppo,model=trm,env=pendulum,delay=none,H_cycles=3",
                             -160.0, 18.0)]:
        for seed in range(10):
            rows.append({
                "condition": cond,
                "arch": cond.split(",arch=")[0] if "arch=" in cond else "ppo",
                "return": round(float(mu + rng.normal(0, sigma)), 2),
                "wall_clock": round(float(rng.uniform(20, 40)), 2),
                "steps_per_sec": round(float(rng.uniform(300, 500)), 2),
                "seed": seed,
            })
    df = pd.DataFrame(rows)
    # Re-derive arch/model/env/delay from condition for cleanliness.
    df["arch"] = [c.split("arch=")[1].split(",")[0] for c in df["condition"]]
    df["model"] = [c.split("model=")[1].split(",")[0] for c in df["condition"]]
    df["env"] = [c.split("env=")[1].split(",")[0] for c in df["condition"]]
    df["delay"] = [c.split("delay=")[1].split(",")[0] for c in df["condition"]]
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="analyze.py",
        description="Analyse a trm-control results CSV: per-condition mean/std, "
                    "95%% bootstrap CI, Cohen's d, t-tests, markdown + figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv", type=str, nargs="?", default=None,
                   help="Input CSV (one row per run).")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate a tiny synthetic CSV and analyse it (smoke test).")
    p.add_argument("--synthetic-out", type=str,
                   default=str(REPO_ROOT / "results" / "synthetic.csv"),
                   help="Where to write the synthetic CSV.")
    p.add_argument("--out", type=str, default=None,
                   help="Output markdown report path (default: docs/plan/reports/analyze_<csv>.md).")
    p.add_argument("--fig", type=str, default=None,
                   help="Output figure path (default: docs/plan/reports/analyze_<csv>.png).")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="Bootstrap resamples (default 2000).")
    p.add_argument("--conf", type=float, default=0.95,
                   help="CI confidence level (default 0.95).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for bootstrap (default 42).")
    p.add_argument("--json-summary", type=str, default=None,
                   help="Optional: also dump the summary table as JSON.")
    args = p.parse_args(argv)

    if args.synthetic:
        csv_path = Path(args.synthetic_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_csv(csv_path)
        print(f"[analyze] wrote synthetic CSV -> {csv_path} ({len(pd.read_csv(csv_path))} rows)")
        csv_arg = str(csv_path)
    else:
        csv_arg = args.csv

    if not csv_arg:
        p.error("provide a CSV path or use --synthetic")

    df = pd.read_csv(Path(csv_arg))
    if "return" not in df.columns and "mean_return" in df.columns:
        df = df.rename(columns={"mean_return": "return"})
    if "return" not in df.columns:
        p.error(f"CSV must contain a 'return' column (have: {list(df.columns)})")

    print(f"[analyze] read {len(df)} rows, {df['condition'].nunique() if 'condition' in df.columns else df.apply(condition_label, axis=1).nunique()} conditions")

    summary = summarize(df, args.n_boot, args.conf, args.seed)
    contrasts = pairwise_contrasts(df, args.n_boot, args.conf, args.seed)

    # Output paths.
    stem = Path(csv_arg).stem
    if args.synthetic:
        stem = "synthetic"
    md_path = Path(args.out) if args.out else (REPORTS_DIR / f"analyze_{stem}.md")
    fig_path = Path(args.fig) if args.fig else (REPORTS_DIR / f"analyze_{stem}.png")

    write_markdown(summary, contrasts, md_path, meta={
        "csv": csv_arg, "n_runs": int(len(df)),
        "n_conditions": int(summary.shape[0]),
        "n_boot": args.n_boot, "conf": args.conf,
    })
    write_figure(summary, fig_path)

    print(f"[analyze] markdown -> {md_path}")
    print(f"[analyze] figure  -> {fig_path}")
    print("\n" + summary.to_string(index=False))
    if not contrasts.empty:
        print("\n" + contrasts.to_string(index=False))

    if args.json_summary:
        jp = Path(args.json_summary)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps({
            "summary": summary.to_dict(orient="records"),
            "contrasts": contrasts.to_dict(orient="records"),
        }, indent=2, default=str))
        print(f"[analyze] json    -> {jp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
