# GPU runbook — producing REAL numbers with servebench

The mock sweep proves the machinery. These steps replace mock data with
measured data on real GPUs. Estimated cost: a few GPU-hours on one node
per SKU (spot is fine; this is throughput benchmarking, not training).

## 0. Prereqs

- One node per SKU you want to characterize (H100, A100, L40S, 4090, ...),
  with Docker + NVIDIA Container Toolkit.
- A HuggingFace token with access to the model you serve (Llama-3.1-8B
  used below; gated repo — accept the license first).

## 1. Serve the model with a candidate config

Each `ServingConfig.short()` string maps to flags 1:1. Example — the
config `tp1_seq32_tok8192_util0.90_chunk1_prefix1`:

```bash
docker run --gpus all -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill True \
  --enable-prefix-caching True
```

Wait for `/health` → 200 before benchmarking.

## 2. Benchmark it with the servebench client

From this repo (no extra deps):

```bash
python3 - <<'EOF'
from servebench import bench_client
r = bench_client.benchmark("http://<node-ip>:8000",
                           n_requests=200, concurrency=32,
                           max_tokens=256)
bench_client.print_report(r)
EOF
```

Tune `n_requests`/`concurrency` until aggregate tok/s plateaus across two
consecutive runs (server saturated, not client-bound). Record: TTFT p50/p95,
mean ITL, aggregate tok/s.

Note: the client counts streamed SSE chunks as tokens. vLLM may emit
partial-token deltas, so treat client tok/s as approximate; for rigorous
per-config numbers also run vLLM's official
`benchmark_serving.py --backend vllm` and use its output tokens/s as the
source of truth. The client's TTFT is exact in both cases.

## 3. Sweep the grid

Repeat steps 1–2 for each config in the grid (script the docker flags from
`ServingConfig.vllm_flags()`), or start with the report's top-10 most
fungible mock configs plus your current production defaults as a baseline.

## 4. Score fungibility on measured data

Replace the mock `tok_per_s_gpu` column in `results/results.csv` with your
measured values and re-run the ranking logic:

```bash
python3 - <<'EOF'
# pip install pandas  (only needed for this analysis step)
import pandas as pd
df = pd.read_csv("results/results_measured.csv")  # your measured data
best = df.groupby("sku")["tok_per_s_gpu"].transform("max")
df["ratio"] = df["tok_per_s_gpu"] / best
print(df.groupby("config")["ratio"].mean().sort_values(ascending=False).head(10))
EOF
```

The config with the highest mean ratio is your fleet-wide default: the one
that degrades least as new SKUs come online.

## 5. Operationalize

- Bake the winning config into your node bring-up image / IaC as the
  default serving profile per model family.
- Re-run the sweep when a new SKU enters the fleet; the fungibility score
  tells you whether the default still holds or needs a per-SKU override.
- Track tok/s/GPU in your fleet dashboard — it is the margin KPI this
  whole exercise optimizes.
