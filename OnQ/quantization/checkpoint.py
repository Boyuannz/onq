# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from OnQ.config.schema import ConfigError


HANDOFF_MANIFEST_NAME = "unified_calibration_manifest.json"
HANDOFF_SCHEMA_VERSION = 2
LEGACY_HANDOFF_SCHEMA_VERSION = 1
STAGE2_START_MANIFEST_NAME = "unified_stage2_start_manifest.json"
STAGE2_START_SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _write_immutable_manifest(
    destination: Path,
    payload: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"existing {label} manifest is invalid: {destination}") from exc
        if existing != payload:
            raise ConfigError(
                f"refusing to replace immutable {label} manifest: {destination}"
            )
        return existing
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def write_calibration_handoff_manifest(
    *,
    checkpoint: Path,
    model: str,
    model_path: str,
    bits: int,
    group_size: int,
    recipe: str,
    upstream_commit: str,
    audit_manifest: dict[str, Any],
    training_metadata: dict[str, Any] | None = None,
    provenance_type: str = "unified_calibration",
    provenance_note: str | None = None,
) -> dict[str, Any]:
    if audit_manifest.get("status") != "PASS":
        raise ConfigError("refusing to register a Calibration checkpoint that failed audit")
    if training_metadata is None and provenance_type != "legacy_registered":
        raise ConfigError(
            "new unified Calibration checkpoints require schema v2 training metadata"
        )
    checkpoint_hash = audit_manifest.get("checkpoint_tree_sha256")
    if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
        raise ConfigError("Calibration audit did not produce a checkpoint tree hash")
    payload = {
        "schema_version": (
            HANDOFF_SCHEMA_VERSION
            if training_metadata is not None
            else LEGACY_HANDOFF_SCHEMA_VERSION
        ),
        "format": "official_fakequant",
        "producer": "qat-unified",
        "provenance_type": provenance_type,
        "provenance_note": provenance_note,
        "model": model,
        "model_path": model_path,
        "bits": bits,
        "group_size": group_size,
        "calibration_recipe": recipe,
        "reasoningqat_upstream_commit": upstream_commit,
        "checkpoint_tree_sha256": checkpoint_hash,
        "validated_projection_count": audit_manifest.get(
            "tensor_validation", {}
        ).get("validated_projection_count"),
    }
    if training_metadata is not None:
        payload["calibration_training"] = training_metadata
    destination = checkpoint / HANDOFF_MANIFEST_NAME
    return _write_immutable_manifest(
        destination,
        payload,
        label="Calibration handoff",
    )


def validate_calibration_handoff_manifest(
    *,
    checkpoint: Path,
    model: str,
    bits: int,
    group_size: int,
    recipe: str,
    audit_manifest: dict[str, Any],
) -> dict[str, Any]:
    path = checkpoint / HANDOFF_MANIFEST_NAME
    if not path.is_file():
        raise ConfigError(
            f"Calibration checkpoint is missing {HANDOFF_MANIFEST_NAME}: {checkpoint}. "
            "Use the current calibration latent output; historical checkpoints "
            "require their original source revision."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid Calibration handoff manifest: {path}") from exc
    schema_version = payload.get("schema_version")
    if schema_version not in {LEGACY_HANDOFF_SCHEMA_VERSION, HANDOFF_SCHEMA_VERSION}:
        raise ConfigError(
            f"unsupported Calibration handoff schema version: {schema_version!r}"
        )
    expected = {
        "format": "official_fakequant",
        "model": model,
        "bits": bits,
        "group_size": group_size,
        "calibration_recipe": recipe,
        "checkpoint_tree_sha256": audit_manifest.get("checkpoint_tree_sha256"),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ConfigError(f"Calibration handoff manifest mismatch: {mismatches}")
    if payload.get("producer") != "qat-unified":
        raise ConfigError("Calibration handoff manifest has an unknown producer")
    if schema_version == HANDOFF_SCHEMA_VERSION:
        training = payload.get("calibration_training")
        required = {
            "resolved_config_sha256",
            "seed",
            "train_size",
            "val_size",
            "training_seqlen",
            "batch_size",
            "epochs_per_block",
            "weight_lr",
            "quant_lr",
            "calibration_sha256",
            "split",
            "runtime_source_overlay_sha256",
        }
        if not isinstance(training, dict) or not required.issubset(training):
            actual_keys = set(training) if isinstance(training, dict) else set()
            missing = sorted(required - actual_keys)
            raise ConfigError(
                f"Calibration handoff schema v2 is missing training metadata: {missing}"
            )
        if not SHA256_PATTERN.fullmatch(str(training["resolved_config_sha256"])):
            raise ConfigError("Calibration handoff has an invalid resolved config hash")
        for key in (
            "train_size",
            "val_size",
            "training_seqlen",
            "batch_size",
            "epochs_per_block",
        ):
            if type(training[key]) is not int or training[key] <= 0:
                raise ConfigError(f"Calibration handoff field {key!r} must be positive")
        if type(training["seed"]) is not int:
            raise ConfigError("Calibration handoff seed must be an integer")
        for key in ("weight_lr", "quant_lr"):
            if type(training[key]) not in {int, float} or training[key] < 0:
                raise ConfigError(f"Calibration handoff field {key!r} must be nonnegative")
        for key in ("calibration_sha256", "runtime_source_overlay_sha256"):
            value = training[key]
            if value is not None and not SHA256_PATTERN.fullmatch(str(value)):
                raise ConfigError(f"Calibration handoff field {key!r} is not a SHA256")
        split = training["split"]
        split_keys = {
            "reasoning_train_size",
            "general_train_size",
            "reasoning_val_size",
            "general_val_size",
        }
        if not isinstance(split, dict) or set(split) != split_keys:
            raise ConfigError("Calibration handoff split metadata is incomplete")
        if any(type(split[key]) is not int or split[key] < 0 for key in split_keys):
            raise ConfigError("Calibration handoff split sizes must be nonnegative integers")
        if (
            split["reasoning_train_size"] + split["general_train_size"]
            != training["train_size"]
            or split["reasoning_val_size"] + split["general_val_size"]
            != training["val_size"]
        ):
            raise ConfigError("Calibration handoff split sizes do not match train/val totals")
    return payload


def validate_stage2_start_manifest(
    *,
    checkpoint: Path,
    model: str,
    bits: int,
    group_size: int,
    recipe: str,
    audit_manifest: dict[str, Any],
) -> dict[str, Any]:
    path = checkpoint / STAGE2_START_MANIFEST_NAME
    if not path.is_file():
        raise ConfigError(
            f"Stage2 start checkpoint is missing {STAGE2_START_MANIFEST_NAME}: "
            f"{checkpoint}. Historical checkpoint registration is not provided "
            "in this release; use the original source revision."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid Stage2 start manifest: {path}") from exc
    expected = {
        "schema_version": STAGE2_START_SCHEMA_VERSION,
        "format": "official_fakequant",
        "producer": "qat-unified",
        "provenance_type": "legacy_registered_stage2_start",
        "source_stage": "stage2",
        "model": model,
        "bits": bits,
        "group_size": group_size,
        "calibration_recipe": recipe,
        "checkpoint_tree_sha256": audit_manifest.get("checkpoint_tree_sha256"),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ConfigError(f"Stage2 start manifest mismatch: {mismatches}")
    for key in ("stage2_method", "source_run", "provenance_note"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"Stage2 start manifest field {key!r} is empty")
    return payload


def validate_stage2_start_checkpoint(
    *,
    checkpoint: Path,
    model: str,
    bits: int,
    group_size: int,
    recipe: str,
    audit_manifest: dict[str, Any],
) -> dict[str, Any]:
    calibration_path = checkpoint / HANDOFF_MANIFEST_NAME
    stage2_path = checkpoint / STAGE2_START_MANIFEST_NAME
    if calibration_path.is_file() and stage2_path.is_file():
        raise ConfigError(
            "checkpoint contains both Calibration and Stage2-start manifests; "
            "the start provenance is ambiguous"
        )
    if stage2_path.is_file():
        return validate_stage2_start_manifest(
            checkpoint=checkpoint,
            model=model,
            bits=bits,
            group_size=group_size,
            recipe=recipe,
            audit_manifest=audit_manifest,
        )
    if calibration_path.is_file():
        return validate_calibration_handoff_manifest(
            checkpoint=checkpoint,
            model=model,
            bits=bits,
            group_size=group_size,
            recipe=recipe,
            audit_manifest=audit_manifest,
        )
    raise ConfigError(
        f"checkpoint has no reviewed start manifest: {checkpoint}. "
        "Use the current calibration latent output; historical checkpoints "
        "require their original source revision."
    )
