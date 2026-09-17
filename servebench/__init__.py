"""servebench: inference serving-config benchmark harness for GPU fleets.

Compares vLLM/SGLang-style serving configs across GPU SKUs by
throughput-per-GPU and a cross-SKU "fungibility" score, so one config can
survive new hardware instead of being retuned at every bring-up.

Two backends:
  * Analytical model (mock): deterministic, CPU-only. All numbers from it
    are labeled MOCK.
  * Live client: benchmarks any OpenAI-compatible endpoint (e.g. a real
    `vllm serve` instance) on real GPUs. See GPU_RUNBOOK.md.
"""
