# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from OnQ.config.schema import ConfigError


def run_checkpoint_audit(
    *,
    python_bin: str,
    checkpoint: Path,
    bits: int,
    group_size: int,
    output_manifest: Path,
) -> dict[str, Any]:
    """Run the canonical tensor-level Calibration audit and return its manifest."""
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    command = (
        python_bin, "-c",
        "import json,sys; from OnQ.quantization.checkpoint_checks import audit_checkpoint; "
        "result=audit_checkpoint(sys.argv[1], 'official_fakequant', int(sys.argv[2]), int(sys.argv[3])); "
        "print(json.dumps(result)); sys.exit(0 if result['status']=='PASS' else 1)",
        str(checkpoint), str(bits), str(group_size),
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output_manifest.with_suffix(output_manifest.suffix + ".stdout.log").write_text(
        result.stdout,
        encoding="utf-8",
    )
    output_manifest.with_suffix(output_manifest.suffix + ".stderr.log").write_text(
        result.stderr,
        encoding="utf-8",
    )
    if result.stdout.strip():
        output_manifest.write_text(result.stdout, encoding="utf-8")
    if not output_manifest.is_file():
        raise ConfigError(
            "Calibration checkpoint audit did not produce its manifest; "
            f"exit_code={result.returncode}"
        )
    try:
        manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Calibration checkpoint audit emitted invalid JSON: {output_manifest}"
        ) from exc
    if result.returncode != 0 or manifest.get("status") != "PASS":
        errors = manifest.get("errors") or [
            f"audit process exited with code {result.returncode}"
        ]
        raise ConfigError(
            "Calibration checkpoint audit failed: " + "; ".join(map(str, errors[:8]))
        )
    if manifest.get("wbits") != bits or manifest.get("group_size") != group_size:
        raise ConfigError("Calibration checkpoint audit returned inconsistent quant metadata")
    return manifest
