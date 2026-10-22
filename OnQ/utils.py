# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
"""Project-owned file hashing."""
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
