# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from OnQ.config.schema import ConfigError, TrainConfig
from OnQ.config.resolved import RESOLVED_CONFIG_NAME, write_resolved_config
from OnQ.quantization.checkpoint import write_calibration_handoff_manifest
from OnQ.quantization.bit_router import build_quant_spec
from OnQ.calibration.audit import run_checkpoint_audit
from OnQ.calibration.block_ap import (
    UPSTREAM_COMMIT,
    build_block_ap_command,
    prepare_runtime_source,
)
from OnQ.calibration.recipe_router import build_calibration_recipe
from OnQ.utils import sha256_file as _sha256




def run_calibration(cfg: TrainConfig) -> None:
    recipe = build_calibration_recipe(cfg.calibration_recipe)
    quant = build_quant_spec(cfg.bits, cfg.group_size)
    if recipe.requires_calibration_path:
        assert cfg.calibration_path is not None
        if not cfg.calibration_path.is_file():
            raise ConfigError(f"calibration file does not exist: {cfg.calibration_path}")
    if not cfg.dry_run:
        if (cfg.run_dir / "calibration_latent").exists():
            raise ConfigError(f"Calibration latent output already exists: {cfg.run_dir / 'calibration_latent'}")
        if (cfg.run_dir / "calibration_fakequant").exists():
            raise ConfigError(
                f"Calibration output already exists: {cfg.run_dir / 'calibration_fakequant'}"
            )
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        resolved = write_resolved_config(cfg, cfg.run_dir / RESOLVED_CONFIG_NAME)
    else:
        resolved = None
    expected_source = cfg.run_dir / "_reasoningqat_runtime"
    source = (
        expected_source
        if cfg.dry_run and (recipe.requires_calibration_path or Path(cfg.model_path).exists())
        else prepare_runtime_source(cfg, recipe)
    )
    command = build_block_ap_command(cfg, recipe, quant, source)
    if cfg.dry_run:
        print(command.as_json())
        return

    (cfg.run_dir / "cache").mkdir(exist_ok=True)
    (cfg.run_dir / "logs").mkdir(exist_ok=True)
    calibration_sha256 = (
        _sha256(cfg.calibration_path) if cfg.calibration_path else None
    )
    runtime_manifest = source / "UNIFIED_RUNTIME_MANIFEST.json"
    runtime_source_overlay_sha256 = (
        _sha256(runtime_manifest) if runtime_manifest.is_file() else None
    )
    assert resolved is not None
    preflight = {
        "status": "PASS",
        "stage": "calibration",
        "algorithm": "ReasoningQAT Block-AP",
        "upstream_commit": "c36060fefb05aa8c29dcad47b52feacf95201d37",
        "model": cfg.model,
        "model_path": cfg.model_path,
        "bits": quant.bits,
        "group_size": quant.group_size,
        "recipe": recipe.name,
        "calibration_path": str(cfg.calibration_path) if cfg.calibration_path else None,
        "calibration_sha256": calibration_sha256,
        "train_size": cfg.settings["train_size"],
        "val_size": cfg.settings["val_size"],
        "training_seqlen": cfg.settings["training_seqlen"],
        "batch_size": cfg.settings["batch_size"],
        "epochs_per_block": cfg.settings["epochs"],
        "seed": cfg.settings["seed"],
        "weight_lr": cfg.settings["weight_lr"],
        "quant_lr": cfg.settings["quant_lr"],
        "gpus": list(cfg.gpu_ids),
        "resolved_config_sha256": resolved["resolved_config_sha256"],
        "runtime_source_overlay_sha256": runtime_source_overlay_sha256,
        "command": command.as_dict(),
    }
    (cfg.run_dir / "preflight_review.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        command.argv,
        cwd=command.cwd,
        env=command.process_env(),
        check=True,
    )
    checkpoint = cfg.run_dir / "calibration_latent"
    from OnQ.quantization.latent_checkpoint import MANIFEST
    latent_manifest = json.loads((checkpoint / MANIFEST).read_text())
    if (latent_manifest.get("status") != "COMPLETE" or
            latent_manifest.get("latent_weights_preserved") is not True):
        raise ConfigError("Calibration did not complete its pre-round latent checkpoint")
    audit = run_checkpoint_audit(
        python_bin=cfg.python_bin,
        checkpoint=checkpoint,
        bits=quant.bits,
        group_size=quant.group_size,
        output_manifest=cfg.run_dir / "calibration_checkpoint_audit.json",
    )
    handoff = write_calibration_handoff_manifest(
        checkpoint=checkpoint,
        model=cfg.model,
        model_path=cfg.model_path,
        bits=quant.bits,
        group_size=quant.group_size,
        recipe=recipe.name,
        upstream_commit=UPSTREAM_COMMIT,
        audit_manifest=audit,
        training_metadata={
            "resolved_config_sha256": resolved["resolved_config_sha256"],
            "seed": cfg.settings["seed"],
            "train_size": cfg.settings["train_size"],
            "val_size": cfg.settings["val_size"],
            "training_seqlen": cfg.settings["training_seqlen"],
            "batch_size": cfg.settings["batch_size"],
            "epochs_per_block": cfg.settings["epochs"],
            "weight_lr": cfg.settings["weight_lr"],
            "quant_lr": cfg.settings["quant_lr"],
            "calibration_sha256": calibration_sha256,
            "gpus": list(cfg.gpu_ids),
            "split": {
                key: cfg.settings.get(key)
                for key in (
                    "reasoning_train_size",
                    "general_train_size",
                    "reasoning_val_size",
                    "general_val_size",
                )
            },
            "runtime_source_overlay_sha256": runtime_source_overlay_sha256,
        },
    )
    (cfg.run_dir / "postflight_review.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "stage": "calibration",
                "checkpoint": str(checkpoint.resolve()),
                "inference_export": str((cfg.run_dir / "calibration_fakequant").resolve()),
                "full_training_resume": False,
                "checkpoint_tree_sha256": audit["checkpoint_tree_sha256"],
                "validated_projection_count": audit["tensor_validation"][
                    "validated_projection_count"
                ],
                "handoff_manifest": handoff,
                "resolved_config_sha256": resolved["resolved_config_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
