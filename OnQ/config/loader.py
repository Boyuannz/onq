# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from OnQ.config.schema import ConfigError, TrainConfig
from OnQ.paths import repository_root


CALIBRATION_RECIPES = {
    "paper_tokens",
    "official",
    "filtered_reasoning",
    "math_train",
    "math_ot_mix",
}

SETTING_TYPES: dict[str, type] = {
    "batch_size": int,
    "epochs": int,
    "general_train_size": int,
    "general_val_size": int,
    "quant_lr": float,
    "reasoning_train_size": int,
    "reasoning_val_size": int,
    "seed": int,
    "train_size": int,
    "training_seqlen": int,
    "val_size": int,
    "weight_lr": float,
}

# One schema for the Python CLI and its compatibility shell entry point.
SETTING_ENV_NAMES = {key: key.upper() for key in SETTING_TYPES}
SETTING_ENV_NAMES.update({
    "batch_size": "CALIBRATION_BATCH_SIZE",
    "epochs": "CALIBRATION_EPOCHS",
})

PROFILE_REQUIREMENT_KEYS = {
    "stage",
    "model",
    "bits",
    "group_size",
    "calibration_recipe",
}

CRITICAL_BACKEND_KEYS = {
    "algorithm.adv_estimator",
    "algorithm.rollout_source",
    "algorithm.teacher_url",
    "data.train_files",
    "data.val_files",
    "data.train_batch_size",
    "data.val_batch_size",
    "data.max_prompt_length",
    "data.max_response_length",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.model.quant_type",
    "actor_rollout_ref.model.q_group_size",
    "actor_rollout_ref.model.train_mode",
    "actor_rollout_ref.actor.optim.lr",
    "actor_rollout_ref.actor.optim.weight_lr",
    "actor_rollout_ref.actor.optim.quant_lr",
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.temperature",
    "actor_rollout_ref.rollout.top_p",
    "trainer.default_local_dir",
    "trainer.save_freq",
    "trainer.test_freq",
    "trainer.total_training_steps",
}


CRITICAL_CALIBRATION_FLAGS = {
    "--train_size",
    "--val_size",
    "--training_seqlen",
    "--epochs",
    "--batch_size",
    "--weight_lr",
    "--quant_lr",
    "--seed",
    "--wbits",
    "--group_size",
    "--calib_dataset",
    "--save_quant_dir",
}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate config key: {key}")
        payload[key] = value
    return payload


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ConfigError(
            f"{path} must use the JSON-compatible subset of YAML: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"config must contain a mapping: {path}")
    return payload


def _config_path(*parts: str) -> Path:
    return repository_root().joinpath("OnQ", "configs", *parts)


def _profile_path(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError(f"invalid experiment profile path: {name!r}")
    if relative.suffix:
        raise ConfigError("experiment profile names must omit the file suffix")
    return _config_path("experiments", *relative.parts).with_suffix(".yaml")


def _parse_setting_value(key: str, raw: str) -> Any:
    expected = SETTING_TYPES.get(key)
    if expected is None:
        raise ConfigError(
            f"unsupported typed setting {key!r}; use a named profile or add it "
            "to the validated setting schema"
        )
    try:
        if expected is bool:
            normalized = raw.lower()
            if normalized not in {"true", "false"}:
                raise ValueError("expected true or false")
            return normalized == "true"
        return expected(raw)
    except ValueError as exc:
        raise ConfigError(
            f"invalid value for {key}: expected {expected.__name__}, got {raw!r}"
        ) from exc


def _parse_typed_overrides(items: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key or key in result:
            reason = "empty key" if not key else "duplicate key"
            raise ConfigError(f"invalid --set {item!r}: {reason}")
        result[key] = _parse_setting_value(key, raw.strip())
    return result


def _validate_setting_stage(
    stage: str,
    settings: dict[str, Any],
    *,
    source: str,
) -> None:
    invalid = set(settings) - set(SETTING_TYPES)
    if invalid:
        raise ConfigError(
            f"{source} contains settings that do not apply to {stage}: "
            f"{sorted(invalid)}"
        )


def _load_profile(name: str | None, selectors: dict[str, Any]) -> dict[str, Any]:
    if name is None:
        return {}
    path = _profile_path(name)
    payload = _read_config(path)
    unexpected = set(payload) - {"requires", "settings"}
    if unexpected:
        raise ConfigError(f"profile {name!r} has unknown keys: {sorted(unexpected)}")
    requirements = payload.get("requires", {})
    settings = payload.get("settings", {})
    if not isinstance(requirements, dict) or not isinstance(settings, dict):
        raise ConfigError(f"profile {name!r} requires mapping fields")
    unknown_requirements = set(requirements) - PROFILE_REQUIREMENT_KEYS
    if unknown_requirements:
        raise ConfigError(
            f"profile {name!r} has unknown requirements: {sorted(unknown_requirements)}"
        )
    mismatches = {
        key: {"expected": expected, "actual": selectors.get(key)}
        for key, expected in requirements.items()
        if selectors.get(key) != expected
    }
    if mismatches:
        raise ConfigError(f"profile {name!r} selector mismatch: {mismatches}")
    parsed: dict[str, Any] = {}
    for key, value in settings.items():
        expected = SETTING_TYPES.get(key)
        if expected is None:
            raise ConfigError(f"profile {name!r} has unsupported setting {key!r}")
        if expected is float and type(value) in {int, float}:
            parsed[key] = float(value)
        elif type(value) is expected:
            parsed[key] = value
        else:
            raise ConfigError(
                f"profile {name!r} setting {key!r} must be {expected.__name__}"
            )
    return parsed


def _validate_backend_args(stage: str, backend_args: tuple[str, ...]) -> None:
    for argument in backend_args:
        normalized = argument.strip().strip("'\"")
        key = normalized.split("=", 1)[0].lstrip("+")
        if key in CRITICAL_BACKEND_KEYS or (
            stage == "calibration" and key.startswith('--') and len(key) > 2
            and any(flag.startswith(key) for flag in CRITICAL_CALIBRATION_FLAGS)
        ):
            raise ConfigError(
                f"contract-critical backend override {key!r} is forbidden; "
                f"use --set with a validated setting"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibration checkpoint tooling; online training uses integrations.verl_v080")
    parser.add_argument("--stage", choices=("calibration",), default=_env("STAGE", "calibration"))
    parser.add_argument("--model", choices=("qwen3_1p7b", "qwen3_4b", "qwen3_8b"), default=_env("MODEL", "qwen3_4b"))
    parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=int(_env("BITS", "3")))
    parser.add_argument("--group-size", type=int, default=int(_env("GROUP_SIZE", "128")))
    parser.add_argument("--calibration-recipe", default=_env("CALIBRATION_RECIPE", "paper_tokens"))
    parser.add_argument("--model-path", default=_env("MODEL_PATH"))
    parser.add_argument("--calibration-path", type=Path, default=_env("CALIBRATION_PATH"))
    parser.add_argument("--run-dir", type=Path, default=_env("RUN_DIR", "runs/unified"))
    parser.add_argument("--gpus", default=_env("GPUS", _env("CUDA_VISIBLE_DEVICES", "0")))
    parser.add_argument("--python-bin", default=_env("PYTHON_BIN", sys.executable))
    parser.add_argument("--runtime", choices=("local", "h100", "gpu"), default=_env("RUNTIME", "local"))
    parser.add_argument("--profile", default=_env("PROFILE"))
    parser.add_argument("--set", dest="typed_overrides", action="append", default=[])
    parser.add_argument("--backend-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", default=_env("DRY_RUN", "false").lower() == "true")
    return parser


def load_train_config(argv: list[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    if args.stage != 'calibration':
        raise ConfigError('Legacy online training was archived; use OnQ.integrations.verl_v080.opd.prepare_paper')
    if args.calibration_recipe not in CALIBRATION_RECIPES:
        raise ConfigError(f"unsupported Calibration recipe: {args.calibration_recipe}")

    model_cfg = _read_config(_config_path("model", f"{args.model}.yaml"))
    quant_cfg = _read_config(_config_path("quant", f"w{args.bits}g{args.group_size}.yaml"))
    runtime_name = "gpu" if args.runtime == "h100" else args.runtime
    runtime_cfg = _read_config(_config_path("runtime", f"{runtime_name}.yaml"))
    if quant_cfg.get("bits") != args.bits or quant_cfg.get("group_size") != args.group_size:
        raise ConfigError(
            "quantization config selector mismatch: "
            f"requested=w{args.bits}g{args.group_size}, "
            f"file_bits={quant_cfg.get('bits')}, "
            f"file_group_size={quant_cfg.get('group_size')}"
        )
    # These recipes share numerical defaults, but keep distinct data selectors.
    recipe_defaults = "official" if args.calibration_recipe == "math_ot_mix" else args.calibration_recipe
    stage_cfg = _read_config(_config_path("calibration", f"{recipe_defaults}.yaml"))

    selectors = {
        "stage": args.stage,
        "model": args.model,
        "bits": args.bits,
        "group_size": args.group_size,
        "calibration_recipe": args.calibration_recipe,
    }
    profile_settings = _load_profile(args.profile, selectors)
    # Explicit CLI settings win over environment settings; profiles are lower
    # priority. Record the effective overrides, not only the command-line ones.
    cli_settings = _parse_typed_overrides(args.typed_overrides)
    env_overrides = [f"{key}={value}" for key, name in SETTING_ENV_NAMES.items()
                     if (value := _env(name)) is not None and key not in cli_settings]
    effective_overrides = [*env_overrides, *args.typed_overrides]
    typed_settings = _parse_typed_overrides(effective_overrides)
    _validate_setting_stage(args.stage, profile_settings, source="experiment profile")
    _validate_setting_stage(args.stage, typed_settings, source="typed overrides")
    settings: dict[str, Any] = {}
    for source in (model_cfg, quant_cfg, stage_cfg, runtime_cfg):
        settings.update(source)
    settings.update(profile_settings)
    settings.update(typed_settings)
    model_path = args.model_path or str(model_cfg["model_id"])
    cfg = TrainConfig(
        stage=args.stage,
        model=args.model,
        bits=args.bits,
        group_size=args.group_size,
        calibration_recipe=args.calibration_recipe,
        model_path=model_path,
        calibration_path=args.calibration_path,
        run_dir=args.run_dir.expanduser().resolve(),
        gpus=args.gpus,
        python_bin=args.python_bin,
        runtime=args.runtime,
        dry_run=args.dry_run,
        profile=args.profile,
        typed_overrides=tuple(effective_overrides),
        backend_args=tuple(args.backend_arg),
        settings=settings,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: TrainConfig) -> None:
    _ = cfg.gpu_ids
    # A data-sampling seed may remain exported from preparation; it does not
    # configure calibration, whose seed is resolved independently above.
    stale = sorted(key for key in ("ROLLOUT_N", "QUANT_MATH_DTYPE", "OPTIMIZER_OFFLOAD") if _env(key) is not None)
    if stale:
        raise ConfigError(f"settings {stale} do not apply to calibration")
    _validate_backend_args(cfg.stage, cfg.backend_args)
    for key, expected_type in SETTING_TYPES.items():
        if expected_type is float and key in cfg.settings:
            value = float(cfg.settings[key])
            if not math.isfinite(value):
                raise ConfigError(f"{key} must be finite")
    if cfg.group_size <= 0:
        raise ConfigError("group_size must be positive")
    if cfg.n_gpus not in (1, 2):
        raise ConfigError("Calibration supports one or two GPUs")
    if cfg.n_gpus == 2 and cfg.settings["batch_size"] != 2:
        raise ConfigError("Two-GPU calibration requires global batch size 2")
    if cfg.calibration_recipe != "official" and cfg.calibration_path is None:
        raise ConfigError(
            f"Calibration recipe {cfg.calibration_recipe!r} requires --calibration-path"
        )
    for key in ("train_size", "training_seqlen", "batch_size", "epochs"):
        if int(cfg.settings[key]) <= 0:
            raise ConfigError(f"{key} must be positive")
    if cfg.settings["val_size"] < 0 or (cfg.calibration_recipe != "paper_tokens" and cfg.settings["val_size"] == 0):
        raise ConfigError("val_size must be positive except for paper_tokens")
    if cfg.calibration_recipe == "paper_tokens" and (cfg.settings["val_size"] != 0 or cfg.settings["train_size"] != 4096 or cfg.settings["training_seqlen"] != 2048 or cfg.settings["epochs"] != 2):
        raise ConfigError("paper_tokens requires 4096 windows of 2048 tokens, two epochs, no validation windows")
    for key in ("weight_lr", "quant_lr"):
        if float(cfg.settings[key]) <= 0:
            raise ConfigError(f"{key} must be positive")
    split_keys = (
        "reasoning_train_size",
        "general_train_size",
        "reasoning_val_size",
        "general_val_size",
    )
    if cfg.calibration_recipe == "official" and any(
        item.split("=", 1)[0].strip() in split_keys
        for item in cfg.typed_overrides
    ):
        raise ConfigError(
            "official Calibration uses the upstream sweep_0.8 split; explicit "
            "split overrides would be recorded without changing that loader"
        )
    if cfg.calibration_recipe == "official":
        expected_reasoning_train = int(cfg.settings["train_size"] * 0.8)
        expected_reasoning_val = int(cfg.settings["val_size"] * 0.8)
        expected_split = {
            "reasoning_train_size": expected_reasoning_train,
            "general_train_size": (
                cfg.settings["train_size"] - expected_reasoning_train
            ),
            "reasoning_val_size": expected_reasoning_val,
            "general_val_size": (
                cfg.settings["val_size"] - expected_reasoning_val
            ),
        }
        actual_split = {
            key: cfg.settings.get(key) for key in split_keys
        }
        if actual_split != expected_split:
            raise ConfigError(
                "official Calibration split must match the upstream sweep_0.8 "
                f"loader: expected={expected_split}, actual={actual_split}"
            )
    if any(key in cfg.settings for key in split_keys):
        if not all(key in cfg.settings for key in split_keys):
            raise ConfigError("explicit Calibration split requires all four split sizes")
        if (
            cfg.settings["reasoning_train_size"]
            + cfg.settings["general_train_size"]
            != cfg.settings["train_size"]
            or cfg.settings["reasoning_val_size"]
            + cfg.settings["general_val_size"]
            != cfg.settings["val_size"]
        ):
            raise ConfigError("explicit Calibration split does not match train/val size")
