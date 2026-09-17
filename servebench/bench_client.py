"""Live benchmark client for any OpenAI-compatible /v1/completions endpoint.

Same client is used against the MOCK server (CPU validation) and against a
real `vllm serve` instance on GPUs (GPU_RUNBOOK.md). Measures, per request:
TTFT, per-token latencies, tokens/sec; aggregates p50/p95 TTFT, mean ITL,
aggregate throughput.

Token counting: counts streamed SSE chunks carrying choices[0].text as one
token each. Exact against the mock server (1 chunk = 1 token). Against real
vLLM the server may emit partial-token deltas, so treat tok/s there as
approximate and use vLLM's official benchmark_serving.py for rigorous
numbers (noted in GPU_RUNBOOK.md). TTFT is exact in both cases.
"""
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List


@dataclass
class RequestStats:
    ttft_s: float
    token_times_s: List[float]  # wall time of each streamed token after first
    n_tokens: int
    ok: bool = True
    error: str = ""


@dataclass
class BenchResult:
    stats: List[RequestStats] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok_stats(self):
        return [s for s in self.stats if s.ok]

    def ttft_p50(self): return statistics.median(s.ttft_s for s in self.ok_stats)
    def ttft_p95(self):
        xs = sorted(s.ttft_s for s in self.ok_stats)
        return xs[max(0, int(len(xs) * 0.95) - 1)] if xs else 0.0
    def mean_itl(self):
        itls = [t for s in self.ok_stats for t in s.token_times_s[1:]]
        return statistics.mean(itls) if itls else 0.0
    def agg_tok_per_s(self):
        toks = sum(s.n_tokens for s in self.ok_stats)
        return toks / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _one_request(url: str, prompt: str, max_tokens: int,
                 timeout: int = 600) -> RequestStats:
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "stream": True}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return RequestStats(0.0, [], 0, ok=False, error=str(e))
    first = True
    ttft = 0.0
    token_times = []
    n_tokens = 0
    try:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            text = (obj.get("choices") or [{}])[0].get("text")
            now = time.perf_counter() - t0
            if not text:  # final stop chunk / empty delta: not a token
                continue
            if first:
                ttft = now
                first = False
            token_times.append(now)
            n_tokens += 1
    except Exception as e:  # noqa: BLE001
        return RequestStats(ttft, token_times, n_tokens, ok=False, error=str(e))
    return RequestStats(ttft, token_times, n_tokens)


def benchmark(url: str, n_requests: int = 8, concurrency: int = 4,
              prompt: str = "Explain load balancing in one paragraph:",
              max_tokens: int = 32) -> BenchResult:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_one_request, url, f"{prompt} [{i}]", max_tokens)
                for i in range(n_requests)]
        stats = [f.result() for f in futs]
    return BenchResult(stats, time.perf_counter() - t0)


def print_report(r: BenchResult):
    ok, total = len(r.ok_stats), len(r.stats)
    print(f"requests ok: {ok}/{total}  elapsed: {r.elapsed_s:.2f}s")
    if not ok:
        for s in r.stats:
            if not s.ok:
                print("  error:", s.error)
        return
    print(f"TTFT  p50: {r.ttft_p50()*1000:.1f} ms   p95: {r.ttft_p95()*1000:.1f} ms")
    print(f"mean ITL: {r.mean_itl()*1000:.2f} ms")
    print(f"aggregate throughput: {r.agg_tok_per_s():.1f} tok/s")
