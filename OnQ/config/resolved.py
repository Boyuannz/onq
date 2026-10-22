# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from OnQ.config.schema import TrainConfig


RESOLVED_CONFIG_NAME = "resolved_config.json"
RESUME_MUTABLE_SETTINGS = {"total_steps", "save_freq", "test_freq"}


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def resolved_config_payload(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "stage": cfg.stage,
        "model": cfg.model,
        "bits": cfg.bits,
        "group_size": cfg.group_size,
        "calibration_recipe": cfg.calibration_recipe,
        "profile": cfg.profile,
        "model_path": cfg.model_path,
        "calibration_path": (
            str(cfg.calibration_path.resolve()) if cfg.calibration_path else None
        ),
        "run_dir": str(cfg.run_dir),
        "gpus": list(cfg.gpu_ids),
        "python_bin": cfg.python_bin,
        "runtime": cfg.runtime,
        "dry_run": cfg.dry_run,
        "typed_overrides": list(cfg.typed_overrides),
        "backend_args": list(cfg.backend_args),
        "settings": dict(sorted(cfg.settings.items())),
    }


def resume_contract_payload(resolved: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(resolved))
    payload.pop("dry_run", None)
    payload.pop("resume_mode", None)
    payload.pop("typed_overrides", None)
    settings = payload.get("settings", {})
    for key in RESUME_MUTABLE_SETTINGS:
        settings.pop(key, None)
    return payload


def resolved_config_record(cfg: TrainConfig) -> dict[str, Any]:
    resolved = resolved_config_payload(cfg)
    return {
        "schema_version": 1,
        "resolved_config": resolved,
        "resolved_config_sha256": _sha256(resolved),
        "resume_contract_sha256": _sha256(resume_contract_payload(resolved)),
        "resume_mutable_settings": sorted(RESUME_MUTABLE_SETTINGS),
    }


def write_resolved_config(cfg: TrainConfig, path: Path) -> dict[str, Any]:
    record = resolved_config_record(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def resolved_config_sha256(cfg: TrainConfig) -> str:
    return resolved_config_record(cfg)["resolved_config_sha256"]
