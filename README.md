# servebench — inference serving-config benchmark for heterogeneous GPU fleets

A neocloud's margin lives in throughput-per-GPU, and its growth lives in
**hardware fungibility**: each new GPU SKU should not mean another round of
manual serving-config tuning during bring-up. servebench sweeps vLLM-style
serving configs across GPU SKUs, ranks them by throughput-per-GPU, and
scores how well **one** config transfers across the whole fleet.

## Quickstart (CPU only, no GPU needed)

```bash
python3 -m unittest discover -s tests        # 11 tests, ~2s
python3 -m servebench.sweep --out results    # 72 configs x 5 SKUs -> results/
```

`results/REPORT.md` gives the fungibility ranking and per-SKU top-5 with
ready-to-paste `vllm serve` flags. `results/results.csv` has the full grid.

Benchmark any OpenAI-compatible endpoint (mock or real vLLM):

```bash
python3 - <<'EOF'
from servebench import bench_client
r = bench_client.benchmark("http://localhost:8000", n_requests=16,
                           concurrency=4, max_tokens=64)
bench_client.print_report(r)
EOF
```

Spin up the mock server to validate the client path end-to-end:

```bash
python3 - <<'EOF'
from servebench import mock_server, bench_client
from servebench.configs import ServingConfig
from servebench.gpu_catalog import get
srv = mock_server.run_server(ServingConfig(), get("L40S"))
r = bench_client.benchmark(f"http://127.0.0.1:{srv.server_port}",
                           n_requests=8, concurrency=2)
bench_client.print_report(r)
srv.shutdown()
EOF
```

## What it does

- `servebench/configs.py` — serving config dataclass + 72-config grid.
  Every field maps 1:1 to a `vllm serve` flag.
- `servebench/gpu_catalog.py` — 5 fleet SKUs (H100-SXM, MI300X, A100-80GB,
  L40S, RTX-4090) with approximate public specs.
- `servebench/serving_model.py` — **MOCK** deterministic analytical model
  (prefill compute-bound, decode KV-bandwidth-bound, preempt/recompute
  thrash when a batch oversubscribes KV capacity). All assumptions A1–A8
  are documented in the module docstring.
- `servebench/sweep.py` — grid sweep → CSV + REPORT.md, fungibility
  ranking, "transfer tax" of naive defaults.
- `servebench/mock_server.py` — OpenAI-compatible SSE streaming server
  (stdlib only) with mock timing; every response carries `X-Mock: 1`.
- `servebench/bench_client.py` — live benchmark client: TTFT p50/p95,
  mean ITL, aggregate tok/s against any `/v1/completions` endpoint.

## Honest verification notes

- **No GPU was available on this machine.** Nothing here has touched real
  hardware. Every tok/s figure from the sweep is labeled MOCK.
- **What actually ran and passed:** 11/11 unit tests on CPU, including an
  end-to-end test where the live client benchmarks the mock server and the
  measured TTFT matches the mock's simulated TTFT within overhead bounds —
  i.e. the *measurement path* is validated, not the hardware numbers.
- **What did not run:** any real vLLM/SGLang server, any GPU kernel, any
  cloud instance. The mock model is a simplification (see A1–A8); absolute
  tok/s values are illustrative, not quotable hardware facts.
- **Mock disclosures:** `results/`, `REPORT.md`, and the mock server's
  `X-Mock` headers all say MOCK explicitly. The one quantitative claim in
  the report ("naive defaults leave ~12% of per-GPU throughput on the
  table fleet-wide") is a mock-model estimate of the *method's* output,
  not a measured result.
- **For real numbers:** follow `GPU_RUNBOOK.md` — it gives the exact
  protocol to run the same sweep against `vllm serve` on real GPUs and
  swap the mock CSV for measured data.

## Why this matters for a GPU neocloud

Bring-up tunes serving configs per SKU by hand; the tuned values rarely
transfer (an 80GB-tuned `max_num_seqs` thrashes a 24GB card's KV cache —
the model reproduces this: oversized batches collapse on small SKUs).
A repeatable sweep that scores *config transferability*, not just peak
throughput, turns bring-up tuning from folklore into a checklist.
