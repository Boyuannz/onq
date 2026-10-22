# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from OnQ.config.schema import ConfigError, TrainConfig
from OnQ.paths import repository_root
from OnQ.quantization.bit_router import QuantSpec
from OnQ.runtime import BackendCommand
from OnQ.calibration.recipe_router import CalibrationRecipe
from OnQ.config.resolved import resolved_config_sha256


UPSTREAM_COMMIT = "c36060fefb05aa8c29dcad47b52feacf95201d37"
OFFICIAL_IMPORT = "from datautils_block import get_loaders, test_ppl"
CUSTOM_IMPORT = (
    "from OnQ.calibration.data import get_loaders\n"
    "from datautils_block import test_ppl"
)
OFFICIAL_CHOICES = (
    'choices=["sweep_0.0", "sweep_0.2", "sweep_0.4", "sweep_0.6", '
    '"sweep_0.8", "sweep_0.9", "sweep_0.95", "sweep_1.0"],'
)
CUSTOM_CHOICES = OFFICIAL_CHOICES[:-2] + (
    ', "paper_tokens", "filtered_reasoning", "math_train", "math_ot_mix"],'
)
OFFICIAL_LOCAL_BRANCH = "        if os.path.exists(args.model):"
COMPAT_LOCAL_BRANCH = (
    '        if os.path.exists(args.model) and '
    'os.environ.get("REASONINGQAT_DENSE_LOCAL", "0") != "1":'
)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ConfigError(f"{label}: expected one upstream match, found {count}")
    return text.replace(old, new, 1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_runtime_source(cfg: TrainConfig, recipe: CalibrationRecipe) -> Path:
    source = (repository_root() / 'OnQ/calibration/upstream')
    local_dense = Path(cfg.model_path).expanduser().exists()
    needs_overlay = recipe.requires_calibration_path or local_dense
    if not needs_overlay:
        return source

    output = cfg.run_dir / "_reasoningqat_runtime"
    if output.exists():
        raise ConfigError(f"run-scoped ReasoningQAT source already exists: {output}")
    shutil.copytree(source, output)
    main_path = output / "main_block_qat.py"
    text = main_path.read_text(encoding="utf-8")
    changes = []
    if recipe.requires_calibration_path:
        text = _replace_once(text, OFFICIAL_IMPORT, CUSTOM_IMPORT, "data import")
        text = _replace_once(text, OFFICIAL_CHOICES, CUSTOM_CHOICES, "recipe choices")
        changes.extend(["custom_data_import", "custom_recipe_choices"])
    if local_dense:
        text = _replace_once(
            text,
            OFFICIAL_LOCAL_BRANCH,
            COMPAT_LOCAL_BRANCH,
            "local dense checkpoint branch",
        )
        changes.append("local_dense_checkpoint")
    main_path.write_text(text, encoding="utf-8")
    manifest = {
        "upstream_commit": UPSTREAM_COMMIT,
        "source": str(source),
        "recipe": recipe.name,
        "changes": changes,
        "main_block_qat_sha256": _sha256(main_path),
        "official_main_block_qat_sha256": _sha256(source / "main_block_qat.py"),
    }
    (output / "UNIFIED_RUNTIME_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def build_block_ap_command(
    cfg: TrainConfig,
    recipe: CalibrationRecipe,
    quant: QuantSpec,
    source: Path,
) -> BackendCommand:
    settings = cfg.settings
    save_dir = cfg.run_dir / "calibration_fakequant"
    argv = (
        cfg.python_bin,
        str(source / "main_block_qat.py"),
        "--model",
        cfg.model_path,
        "--wbits",
        str(quant.bits),
        "--group_size",
        str(quant.group_size),
        "--calib_dataset",
        recipe.backend_dataset,
        "--train_size",
        str(settings["train_size"]),
        "--val_size",
        str(settings["val_size"]),
        "--training_seqlen",
        str(settings["training_seqlen"]),
        "--epochs",
        str(settings["epochs"]),
        "--batch_size",
        str(settings["batch_size"]),
        "--weight_lr",
        str(settings["weight_lr"]),
        "--quant_lr",
        str(settings["quant_lr"]),
        "--seed",
        str(settings["seed"]),
        "--cache_dir",
        str(cfg.run_dir / "cache"),
        "--output_dir",
        str(cfg.run_dir / "logs"),
        "--save_quant_dir",
        str(save_dir),
        "--save_latent_dir",
        str(cfg.run_dir / "calibration_latent"),
        *cfg.backend_args,
    )
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(cfg.gpu_ids),
        "ONQ_CALIBRATION_GPUS": str(cfg.n_gpus),
        "PYTHONPATH": str((repository_root())),
        "TOKENIZERS_PARALLELISM": "false",
        "UNIFIED_STAGE": "calibration",
        "UNIFIED_CALIBRATION_RECIPE": recipe.name,
        "UNIFIED_QUANT_BITS": str(quant.bits),
        "UNIFIED_RESOLVED_CONFIG_SHA256": resolved_config_sha256(cfg),
    }
    if Path(cfg.model_path).expanduser().exists():
        env["REASONINGQAT_DENSE_LOCAL"] = "1"
    if cfg.calibration_path is not None:
        env["CALIBRATION_CALIBRATION_PATH"] = str(cfg.calibration_path.resolve())
    split_environment = {
        "reasoning_train_size": "CALIBRATION_REASONING_TRAIN_SIZE",
        "general_train_size": "CALIBRATION_GENERAL_TRAIN_SIZE",
        "reasoning_val_size": "CALIBRATION_REASONING_VAL_SIZE",
        "general_val_size": "CALIBRATION_GENERAL_VAL_SIZE",
    }
    for setting, environment_name in split_environment.items():
        if setting in settings:
            env[environment_name] = str(settings[setting])
    return BackendCommand(argv=argv, cwd=source, env=env)
