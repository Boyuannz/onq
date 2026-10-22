# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import shlex


_PASSTHROUGH_ENV_NAMES = {
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LIBRARY_PATH",
    "LOGNAME",
    "NO_PROXY",
    "PATH",
    "SHELL",
    "TMPDIR",
    "USER",
    "XDG_CACHE_HOME",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
_PASSTHROUGH_ENV_PREFIXES = (
    "CONDA_",
    "CUBLAS_",
    "CUDA_",
    "FI_",
    "FLASH_",
    "GLOO_",
    "HF_",
    "HIP_",
    "HUGGINGFACE_",
    "LC_",
    "MKL_",
    "NCCL_",
    "OMP_",
    "OMPI_",
    "OPENAI_",
    "PYTORCH_",
    "PYTHON_",
    "PMI_",
    "PMIX_",
    "RAY_",
    "ROCM_",
    "SLURM_",
    "TORCH_",
    "TOKENIZERS_",
    "TRANSFORMERS_",
    "TRITON_",
    "UCX_",
    "VIRTUAL_ENV",
    "VLLM_",
    "WANDB_",
)


def _runtime_base_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key in _PASSTHROUGH_ENV_NAMES
        or key.startswith(_PASSTHROUGH_ENV_PREFIXES)
    }


@dataclass(frozen=True)
class BackendCommand:
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "cwd": str(self.cwd),
            "argv": list(self.argv),
            "env": dict(sorted(self.env.items())),
            "shell": " ".join(shlex.quote(part) for part in self.argv),
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def process_env(self) -> dict[str, str]:
        result = _runtime_base_environment()
        result.update(self.env)
        return result
