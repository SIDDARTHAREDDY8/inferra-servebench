"""Sweep runner: evaluates the config grid on every SKU, writes results.

Outputs (all numbers MOCK - from the analytical model, not hardware):
  results/results.csv  - one row per (SKU, config)
  results/REPORT.md    - per-SKU top-5, fungibility ranking, transfer tax

Usage: python -m servebench.sweep [--skus H100-SXM,A100-80GB] [--out results]
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timezone

from .configs import ServingConfig, default_grid
from .gpu_catalog import CATALOG
from .serving_model import evaluate, fungibility_score, LLAMA_8B


def run(skus, outdir):
    os.makedirs(outdir, exist_ok=True)
    grid = default_grid()
    rows = []
    for gpu in skus:
        for cfg in grid:
            m = evaluate(cfg, gpu, LLAMA_8B)
            rows.append({
                "sku": gpu.name, "config": cfg.short(),
                "vllm_flags": cfg.vllm_flags(),
                "feasible": m.feasible,
                "ttft_s": round(m.ttft_s, 6),
                "itl_ms": round(m.itl_s * 1000, 4),
                "tok_per_s": round(m.tok_per_s, 2),
                "tok_per_s_gpu": round(m.tok_per_s_gpu, 2),
                "usd_per_mtok": round(m.usd_per_mtok, 4),
                "eff_batch": round(m.eff_batch, 1),
            })
    csv_path = os.path.join(outdir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Fungibility ranking over the grid.
    ranked = []
    for cfg in grid:
        score, per_sku = fungibility_score(cfg, skus, LLAMA_8B)
        ranked.append((score, cfg, per_sku))
    ranked.sort(key=lambda r: r[0], reverse=True)

    # Per-SKU top-5 by tok/s/GPU.
    top = {}
    for gpu in skus:
        cands = [(r["tok_per_s_gpu"], r["config"], r["vllm_flags"])
                 for r in rows if r["sku"] == gpu.name and r["feasible"]]
        cands.sort(reverse=True)
        top[gpu.name] = cands[:5]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = [f"# ServeBench sweep report (MOCK numbers - analytical model, no GPU)",
          f"\nGenerated {ts}. Model: {LLAMA_8B.name}. "
          "Every figure below comes from the deterministic mock model in "
          "`servebench/serving_model.py`. Do not quote as measured hardware data.",
          "\n## Most fungible configs (one config, whole fleet)",
          "\n| rank | fungibility score | config | " +
          " | ".join(g.name for g in skus) + " tok/s/GPU |",
          "|---|---|---|" + "|".join("---" for _ in skus) + "|"]
    for i, (score, cfg, per_sku) in enumerate(ranked[:10], 1):
        vals = " | ".join(f"{per_sku[g.name]:.0f}" for g in skus)
        md.append(f"| {i} | {score:.3f} | `{cfg.short()}` | {vals} |")
    md.append("\nFungibility score = mean over SKUs of "
              "(config tok/s/GPU / best tok/s/GPU on that SKU). "
              "1.000 = optimal on every SKU.")
    for gpu in skus:
        md.append(f"\n## {gpu.name}: top-5 by tok/s/GPU (mock)")
        for tps, short, flags in top[gpu.name]:
            md.append(f"- {tps:.0f} tok/s/GPU `{short}`\n  `{flags}`")
    md.append("\n## Transfer tax of the naive default")
    default = ServingConfig()
    dscore, _ = fungibility_score(default, skus, LLAMA_8B)
    best_score = ranked[0][0]
    md.append(f"vLLM-ish defaults score {dscore:.3f} vs best-grid "
              f"{best_score:.3f}: running defaults fleet-wide leaves "
              f"~{(1 - dscore / max(best_score, 1e-9)) * 100:.1f}% of "
              "per-GPU throughput on the table (mock estimate).")
    md_path = os.path.join(outdir, "REPORT.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {csv_path} ({len(rows)} rows)")
    print(f"wrote {md_path}")
    print(f"best fungible config: {ranked[0][1].short()} score={ranked[0][0]:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skus", default=",".join(g.name for g in CATALOG))
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    wanted = {s.strip() for s in a.skus.split(",")}
    skus = [g for g in CATALOG if g.name in wanted]
    if not skus:
        sys.exit(f"no matching SKUs in {sorted(wanted)}")
    run(skus, a.out)


if __name__ == "__main__":
    main()
