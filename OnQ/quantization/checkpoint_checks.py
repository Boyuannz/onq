"""Internal checks for the calibration-to-OPD checkpoint handoff."""
from pathlib import Path
from typing import Any
import hashlib
import json

import torch
from safetensors import safe_open

CHECKPOINT_IDENTITY_FILES = {
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


def _append_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_checkpoint_identity(
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    errors: list[str] = []
    candidates: set[Path] = set()
    candidates.update(
        checkpoint / name
        for name in CHECKPOINT_IDENTITY_FILES
        if (checkpoint / name).is_file()
    )
    for pattern in ("*.safetensors", "*.safetensors.index.json"):
        candidates.update(path for path in checkpoint.glob(pattern) if path.is_file())

    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(checkpoint).as_posix()):
        relative = path.relative_to(checkpoint).as_posix()
        try:
            digest = _sha256_file(path)
            size = path.stat().st_size
        except OSError as exc:
            _append_error(errors, f"cannot hash {relative}: {exc}")
            continue
        entries.append({"path": relative, "sha256": digest, "size_bytes": size})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))

    if errors or not entries:
        return entries, None, errors
    return entries, aggregate.hexdigest(), errors


def audit_checkpoint(checkpoint, checkpoint_format, wbits, group_size):
    """Inspect the training checkpoint; retain the handoff manifest interface."""
    root = Path(checkpoint).expanduser().resolve()
    result = dict(status="FAIL", wbits=wbits, group_size=group_size,
                  checkpoint_tree_sha256=None,
                  tensor_validation={"validated_projection_count": 0}, errors=[])
    try:
        if checkpoint_format != "official_fakequant":
            raise ValueError("Expected calibration training weights")
        if type(wbits) is not int or not 2 <= wbits <= 16:
            raise ValueError("Invalid bit width")
        if type(group_size) is not int or group_size <= 0:
            raise ValueError("Invalid group size")
        config = _load_json(root / "config.json")
        layers = config["num_hidden_layers"]
        if type(layers) is not int or layers <= 0:
            raise ValueError("Invalid layer count")
        for name in ("qat_blockwise", "quantization_config"):
            spec = config.get(name, {})
            for key in ("bits", "wbits", "weight_bits"):
                if key in spec and spec[key] != wbits:
                    raise ValueError("Checkpoint bit width mismatch")
            if "group_size" in spec and spec["group_size"] != group_size:
                raise ValueError("Checkpoint group size mismatch")
        locations = {}
        for shard in sorted(root.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in locations:
                        raise ValueError("Duplicate tensor: " + key)
                    locations[key] = shard.name
        index = root / "model.safetensors.index.json"
        if index.exists() and _load_json(index)["weight_map"] != locations:
            raise ValueError("Checkpoint index and shard contents differ")
        projections = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                       "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
        def load(key):
            with safe_open(root / locations[key], framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)
        for layer in range(layers):
            for projection in projections:
                prefix = f"model.layers.{layer}.{projection}"
                shapes = []
                for suffix in ("weight", "weight_quantizer.scale", "weight_quantizer.zero_point"):
                    key = prefix + "." + suffix
                    tensor = load(key)
                    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                        raise ValueError("Expected floating tensor: " + key)
                    flat = tensor.reshape(-1)
                    if any(not torch.isfinite(part).all().item() for part in flat.split(8 << 20)):
                        raise ValueError("Nonfinite tensor: " + key)
                    shapes.append(tuple(tensor.shape))
                    del tensor, flat
                weight, scale, zero = shapes
                if len(weight) != 2 or min(weight) <= 0 or weight[1] % group_size:
                    raise ValueError("Invalid weight shape: " + prefix)
                expected = (weight[0] * (weight[1] // group_size), 1)
                if scale != expected or zero != expected:
                    raise ValueError("Quantizer shape mismatch: " + prefix)
                bit_key = prefix + ".weight_quantizer.wbits_buf"
                if bit_key in locations and load(bit_key).item() != wbits:
                    raise ValueError("Quantizer bit width mismatch: " + prefix)
                result["tensor_validation"]["validated_projection_count"] += 1
        files, tree_hash, errors = _build_checkpoint_identity(root)
        if errors or not tree_hash:
            raise ValueError("Cannot identify checkpoint: " + str(errors))
        result.update(status="PASS", checkpoint_tree_sha256=tree_hash)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        result["errors"].append(str(exc))
    return result
