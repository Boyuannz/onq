# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantSpec:
    bits: int
    group_size: int
    quant_type: str
    qmin: int
    qmax: int


def build_quant_spec(bits: int, group_size: int = 128) -> QuantSpec:
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4; got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive; got {group_size}")
    return QuantSpec(
        bits=bits,
        group_size=group_size,
        quant_type=f"efficientqat-int{bits}",
        qmin=0,
        qmax=(1 << bits) - 1,
    )
