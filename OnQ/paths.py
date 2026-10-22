# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]
