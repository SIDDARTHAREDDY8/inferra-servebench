"""Serving configs: the knobs a neocloud tunes per model + GPU SKU.

Every field maps 1:1 to a `vllm serve` flag, so a winning config from the
mock sweep can be pasted straight into the GPU runbook (GPU_RUNBOOK.md).
"""
from dataclasses import dataclass, field
from itertools import product
from typing import Iterator, List


@dataclass(frozen=True)
class ServingConfig:
    tensor_parallel_size: int = 1          # --tensor-parallel-size
    max_num_seqs: int = 128                # --max-num-seqs
    max_num_batched_tokens: int = 8192     # --max-num-batched-tokens
    gpu_memory_utilization: float = 0.90   # --gpu-memory-utilization
    enable_chunked_prefill: bool = True    # --enable-chunked-prefill
    enable_prefix_caching: bool = True     # --enable-prefix-caching

    def vllm_flags(self) -> str:
        return (
            f"--tensor-parallel-size {self.tensor_parallel_size} "
            f"--max-num-seqs {self.max_num_seqs} "
            f"--max-num-batched-tokens {self.max_num_batched_tokens} "
            f"--gpu-memory-utilization {self.gpu_memory_utilization} "
            f"--enable-chunked-prefill {str(self.enable_chunked_prefill)} "
            f"--enable-prefix-caching {str(self.enable_prefix_caching)}"
        )

    def short(self) -> str:
        return (
            f"tp{self.tensor_parallel_size}_seq{self.max_num_seqs}_"
            f"tok{self.max_num_batched_tokens}_util{self.gpu_memory_utilization}_"
            f"chunk{'1' if self.enable_chunked_prefill else '0'}_"
            f"prefix{'1' if self.enable_prefix_caching else '0'}"
        )


def default_grid() -> List[ServingConfig]:
    """A practical sweep: 72 configs. Pure math in mock mode, so it is instant."""
    grid = []
    for seqs, toks, util, chunk, prefix in product(
        [32, 128, 256],
        [2048, 8192, 16384],
        [0.85, 0.95],
        [True, False],
        [True, False],
    ):
        grid.append(
            ServingConfig(
                max_num_seqs=seqs,
                max_num_batched_tokens=toks,
                gpu_memory_utilization=util,
                enable_chunked_prefill=chunk,
                enable_prefix_caching=prefix,
            )
        )
    return grid
