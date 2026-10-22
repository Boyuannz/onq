# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
"""Identity and numerical contract shared by training launches and exports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from OnQ.config.schema import ConfigError


CONTRACT_NAME = "quantization_export_contract.json"
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    "generation_config.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(source: Path) -> dict[str, str]:
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (source / name).is_file():
            raise ConfigError(f"missing tokenizer source file: {source / name}")
    return {name: file_sha256(source / name) for name in TOKENIZER_FILES
            if (source / name).is_file()}


def read_export_contract(checkpoint: Path, explicit_path: Path | None = None) -> dict:
    embedded = checkpoint / "actor" / CONTRACT_NAME
    path = explicit_path or (embedded if embedded.is_file() else checkpoint.parent / CONTRACT_NAME)
    if not path.is_file():
        raise ConfigError(
            f"missing quantization/export contract: {path}; legacy checkpoints "
            "require an explicitly reviewed --contract, never inferred precision"
        )
    contract = json.loads(path.read_text())
    required = {"schema_version", "bits", "group_size", "parameter_dtype", "quant_math_dtype",
                "materialized_dtype", "scale_floor", "scale_ceil", "tokenizer_files",
                "tokenizer_source", "artifact_kind"}
    if not isinstance(contract, dict) or not required <= contract.keys():
        raise ConfigError("incomplete quantization/export contract")
    if (contract["schema_version"] != 1
            or contract["artifact_kind"] != "fsdp_training_checkpoint"
            or type(contract["bits"]) is not int or type(contract["group_size"]) is not int
            or contract["bits"] not in (2, 3, 4) or contract["group_size"] <= 0
            or contract["parameter_dtype"] != "fp32"
            or contract["quant_math_dtype"] not in ("fp32", "bf16")
            or contract["materialized_dtype"] not in ("fp32", "fp16", "bf16")
            or not 0 < contract["scale_floor"] <= contract["scale_ceil"] < float("inf")):
        raise ConfigError("invalid quantization/export contract")
    contract = dict(contract)
    contract["contract_file"] = str(path.resolve())
    contract["contract_sha256"] = file_sha256(path)
    return contract


def native_export_contract(checkpoint: Path, explicit_path: Path | None = None) -> dict:
    """Read the native adapter's saved numerical identity, never infer from weights.

    Native veRL saves its HF config/tokenizer alongside rank shards, without
    the legacy training_state.json or quantization_export_contract.json.
    """
    from OnQ.integrations.verl_v080.utils.model_adapter import adapter_from_config
    from OnQ.quantization.qlinear import resolve_quantizer_scale_floor
    from OnQ.quantization.quantizer import DEFAULT_EQAT_SCALE_CEIL, DEFAULT_EQAT_SCALE_FLOOR

    hf = checkpoint / 'actor' / 'huggingface'
    path = hf / 'config.json'
    saved = json.loads(path.read_text())
    try:
        adapter = adapter_from_config(saved)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    bits, group = adapter.get('bits'), adapter.get('group_size', 128)
    floor = adapter.get('scale_floor', DEFAULT_EQAT_SCALE_FLOOR)
    if (saved.get('model_type') != 'qwen3'
            or saved.get('qat_quant_input_dtype') != 'float32'
            or saved.get('qat_activation_dtype') != 'bfloat16'
            or adapter.get('initial_weight_dtype') not in (None, 'float32')
            or type(bits) is not int or bits not in (2, 3, 4)
            or type(group) is not int or group <= 0
            or type(floor) not in (int, float) or not math.isfinite(floor)
            or not 0 < floor <= DEFAULT_EQAT_SCALE_CEIL):
        raise ConfigError('Unsupported native Qwen3 FP32/BF16 quantization identity')
    # install_qlinears uses the same legacy-default translation as conversion.
    floor = resolve_quantizer_scale_floor(f'efficientqat-int{bits}', floor)
    result = dict(schema_version=1, artifact_kind='fsdp_training_checkpoint',
                  checkpoint_format='native_verl', bits=bits, group_size=group,
                  parameter_dtype='fp32', quant_math_dtype='fp32', materialized_dtype='bf16',
                  scale_floor=floor, scale_ceil=DEFAULT_EQAT_SCALE_CEIL,
                  tokenizer_source=str(hf), tokenizer_files=tokenizer_identity(hf),
                  contract_file=str(path), contract_sha256=file_sha256(path))
    if explicit_path is not None:
        declared = read_export_contract(checkpoint, explicit_path)
        for key in ('bits', 'group_size', 'parameter_dtype', 'quant_math_dtype',
                    'materialized_dtype', 'scale_floor', 'scale_ceil', 'tokenizer_files'):
            if declared[key] != result[key]:
                raise ConfigError('Explicit export contract conflicts with native metadata: ' + key)
        result['reviewed_contract_sha256'] = declared['contract_sha256']
    return result
