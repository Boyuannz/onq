# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a unified training contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class TrainConfig:
    stage: str
    model: str
    bits: int
    group_size: int
    calibration_recipe: str
    model_path: str
    calibration_path: Path | None
    run_dir: Path
    gpus: str
    python_bin: str
    runtime: str
    dry_run: bool
    profile: str | None = None
    typed_overrides: tuple[str, ...] = ()
    backend_args: tuple[str, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def gpu_ids(self) -> tuple[str, ...]:
        ids = tuple(part.strip() for part in self.gpus.split(",") if part.strip())
        if not ids:
            raise ConfigError("at least one GPU id is required")
        if len(set(ids)) != len(ids):
            raise ConfigError(f"GPU ids must be unique, got {self.gpus!r}")
        return ids

    @property
    def n_gpus(self) -> int:
        return len(self.gpu_ids)
