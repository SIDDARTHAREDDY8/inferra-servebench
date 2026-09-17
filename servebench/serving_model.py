"""MOCK analytical serving model. Deterministic. CPU-only. NOT measured data.

What it estimates for a (model, GPU SKU, serving config, workload) tuple:
  * TTFT           - time to first token (prefill), seconds
  * ITL            - inter-token latency (decode step), seconds
  * tok_per_s      - aggregate output tokens/sec across the GPU(s)
  * tok_per_s_gpu  - output tokens/sec per GPU (the fungibility KPI)
  * feasible       - whether the config fits in memory at all

Documented assumptions (all simplifications are deliberate and labeled):
  A1. Prefill is compute-bound: 2*P*prompt_len FLOPs / effective TFLOPS.
      Attention quadratic term ignored (fine for prompt_len <= 4k on 8B).
  A2. Decode is KV-cache-bandwidth-bound: each step re-reads every live
      request's KV cache. step_time = B * kv_bytes/token * avg_seq_len / bw.
      Compute term 2*P*B/FLOPs is also evaluated; the max wins.
  A3. KV bytes/token = 2 * n_kv_heads * head_dim * n_layers * bytes/elem.
      (Llama-3.1-8B: 2*8*128*32*2 = 131,072 B = 128 KiB/token. Real value.)
  A4. The server is assumed saturated: requested batch B = max_num_seqs.
      If B * avg_total_tokens exceeds KV capacity, vLLM preempts/recomputes;
      modeled as a linear thrash penalty tok/s *= kv_cap / kv_needed. This is
      THE cross-SKU effect: a config tuned for an 80GB card thrashes on 24GB.
  A5. Prefix caching: halves prefill time (assumes ~50% of the prompt is a
      shared, cacheable prefix). Rough but directionally right.
  A6. Chunked prefill: when OFF and the workload mixes prefills with decodes,
      each decode step stalls; modeled as +50% of one prefill time spread
      over the batch. When ON there is no stall.
  A7. max_num_batched_tokens: if B * prompt_len exceeds it, the prefill is
      chunked over ceil(...) passes and prefill time scales up.
  A8. Tensor parallelism splits FLOPs and weights evenly; no comm overhead
      is modeled (optimistic for tp>1, noted in README).

Use this to RANK configs and study config transfer across SKUs. Do not
quote the absolute tok/s numbers as hardware facts.
"""
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Tuple

from .configs import ServingConfig
from .gpu_catalog import GPU


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: float          # parameter count
    n_layers: int
    n_kv_heads: int
    head_dim: int
    bytes_per_elem: float = 2.0  # fp16

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.n_kv_heads * self.head_dim * self.n_layers * self.bytes_per_elem

    @property
    def weights_bytes(self) -> float:
        return self.params * self.bytes_per_elem


LLAMA_8B = ModelSpec("Llama-3.1-8B", params=8.03e9, n_layers=32,
                     n_kv_heads=8, head_dim=128)

# (prompt_tokens, output_tokens, share) - a chat-serving-ish mix
DEFAULT_WORKLOAD: List[Tuple[int, int, float]] = [
    (256, 128, 0.60),
    (1024, 256, 0.30),
    (4096, 512, 0.10),
]

SHARED_PREFIX_FRACTION = 0.50  # A5


@dataclass
class Metrics:
    feasible: bool
    ttft_s: float = 0.0
    itl_s: float = 0.0
    tok_per_s: float = 0.0
    tok_per_s_gpu: float = 0.0
    eff_batch: float = 0.0
    usd_per_mtok: float = float("inf")


def evaluate(cfg: ServingConfig, gpu: GPU, model: ModelSpec = LLAMA_8B,
             workload: List[Tuple[int, int, float]] = DEFAULT_WORKLOAD) -> Metrics:
    gb = 1024 ** 3
    mem_bytes = gpu.mem_gb * gb
    kv_bpt = model.kv_bytes_per_token
    weights = model.weights_bytes

    # Feasibility: weights must fit on one shard with headroom.
    if weights / cfg.tensor_parallel_size > mem_bytes * cfg.gpu_memory_utilization * 0.9:
        return Metrics(feasible=False)

    usable_kv = mem_bytes * cfg.gpu_memory_utilization - weights / cfg.tensor_parallel_size
    if usable_kv <= 0:
        return Metrics(feasible=False)
    kv_cap_tokens = usable_kv / kv_bpt

    avg_prompt = sum(p * s for p, _, s in workload)
    avg_out = sum(o * s for _, o, s in workload)
    avg_total = avg_prompt + avg_out

    # Saturated batch; oversubscribing KV capacity causes preempt/recompute
    # thrash (A4): the scheduler still attempts the full batch, but a
    # (1 - thrash) fraction of every step's cycles is wasted on recompute,
    # so useful throughput scales by thrash. This is THE cross-SKU effect:
    # a config tuned for an 80GB card thrashes on 24GB.
    requested = float(cfg.max_num_seqs)
    kv_needed = requested * avg_total
    thrash = min(1.0, kv_cap_tokens / kv_needed) if kv_needed > 0 else 1.0
    eff_batch = requested * thrash
    if eff_batch < 1:
        return Metrics(feasible=False)

    flops = gpu.fp16_tflops * 1e12 * cfg.tensor_parallel_size  # A8: no comm cost
    bw = gpu.mem_bw_gbs * 1e9

    # Prefill (TTFT), weighted over the mix - A1, A5, A7.
    ttft = 0.0
    for p, _, share in workload:
        t = 2 * model.params * p / flops
        if cfg.enable_prefix_caching:
            t *= (1 - SHARED_PREFIX_FRACTION)
        passes = ceil(requested * p / cfg.max_num_batched_tokens)
        t *= max(1, passes)
        ttft += share * t

    # Decode step (ITL) - A2, A6. Computed on the requested batch because
    # the scheduler attempts it; useful throughput is then scaled by thrash.
    kv_term = requested * kv_bpt * avg_total / bw
    compute_term = 2 * model.params * requested / flops
    step = max(kv_term, compute_term)
    if not cfg.enable_chunked_prefill:
        step *= 1 + 0.5 * ttft / max(step * requested, 1e-9)

    tok_per_s = requested / step * thrash
    tok_per_s_gpu = tok_per_s / cfg.tensor_parallel_size
    usd_per_mtok = (gpu.usd_per_hr * cfg.tensor_parallel_size) / max(tok_per_s, 1e-9) * 1e6 / 3600

    return Metrics(feasible=True, ttft_s=ttft, itl_s=step,
                   tok_per_s=tok_per_s, tok_per_s_gpu=tok_per_s_gpu,
                   eff_batch=eff_batch, usd_per_mtok=usd_per_mtok)


def fungibility_score(cfg: ServingConfig, gpus: List[GPU],
                      model: ModelSpec = LLAMA_8B) -> Tuple[float, Dict[str, float]]:
    """How well ONE config transfers across a heterogeneous fleet.

    score = mean over SKUs of (this config's tok/s/GPU / best tok/s/GPU on
    that SKU). 1.0 = optimal everywhere; lower = you pay a "transfer tax"
    for running one config fleet-wide. Infeasible on any SKU -> 0.0.
    """
    per_sku = {}
    for g in gpus:
        m = evaluate(cfg, g, model)
        per_sku[g.name] = m.tok_per_s_gpu if m.feasible else 0.0
    if any(v == 0.0 for v in per_sku.values()):
        return 0.0, per_sku
    best = {g.name: max(evaluate(c, g, model).tok_per_s_gpu
                        for c in _best_candidates(g, model))
            for g in gpus}
    ratios = [per_sku[g.name] / best[g.name] for g in gpus]
    return sum(ratios) / len(ratios), per_sku


def _best_candidates(gpu: GPU, model: ModelSpec):
    # best-per-SKU is computed over the same grid sweep() uses
    from .configs import default_grid
    return [c for c in default_grid() if evaluate(c, gpu, model).feasible]
