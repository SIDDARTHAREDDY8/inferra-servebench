"""GPU catalog: approximate vendor-published specs for common fleet SKUs.

Numbers are rounded public specs (FP16 dense tensor TFLOPS, HBM capacity,
memory bandwidth, ballpark on-demand $/GPU/hr). They feed only the MOCK
analytical model; never treat mock output as measured hardware data.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    fp16_tflops: float   # dense FP16 tensor throughput, TFLOP/s
    mem_gb: float        # HBM capacity, GiB
    mem_bw_gbs: float    # memory bandwidth, GB/s
    usd_per_hr: float    # ballpark on-demand price, USD per GPU-hour


CATALOG = [
    GPU("H100-SXM",  fp16_tflops=989,  mem_gb=80,  mem_bw_gbs=3350, usd_per_hr=2.50),
    GPU("MI300X",    fp16_tflops=1307, mem_gb=192, mem_bw_gbs=5300, usd_per_hr=2.40),
    GPU("A100-80GB", fp16_tflops=624,  mem_gb=80,  mem_bw_gbs=2039, usd_per_hr=1.50),
    GPU("L40S",      fp16_tflops=362,  mem_gb=48,  mem_bw_gbs=864,  usd_per_hr=0.90),
    GPU("RTX-4090",  fp16_tflops=330,  mem_gb=24,  mem_bw_gbs=1008, usd_per_hr=0.60),
]


def get(name: str) -> GPU:
    for g in CATALOG:
        if g.name == name:
            return g
    raise KeyError(f"unknown GPU SKU: {name}")
