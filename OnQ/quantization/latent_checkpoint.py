"""Lossless model-state handoff, separate from materialized inference exports.

Block-AP discards each block's optimizer. These artifacts preserve model state,
not the optimizer/data position needed to resume an interrupted Block-AP run.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


MANIFEST = "latent_checkpoint.json"


def require_fp32_latent_source(path, *, bits, group_size, check_provenance=True):
    """Header-only admission for model-state training handoff, before HF casts.

    This is a precision/format gate, not checksum or full-resume acceptance.
    Integer quantizer metadata is allowed; materialized inference is not.
    """
    from safetensors import safe_open

    path = Path(path)
    if check_provenance:
        config = json.loads((path / 'config.json').read_text())
        records = [config.get('qat_latent_checkpoint', {}), config.get('qat_blockwise', {})]
        for name in (MANIFEST, 'checkpoint_kind.json'):
            if (path / name).is_file():
                record = json.loads((path / name).read_text())
                if name == MANIFEST and record.get('status') != 'COMPLETE':
                    raise ValueError('Incomplete latent checkpoint')
                records.append(record)
        if any(r.get('latent_weights_preserved') is False or
               r.get('artifact_kind') == 'materialized_inference' for r in records):
            raise ValueError('Inference exports cannot be training handoffs')
        declarations = [r for r in records if r.get('latent_weights_preserved') is True]
        if not declarations or any(r.get('bits') != bits or r.get('group_size') != group_size
                                   for r in declarations):
            raise ValueError('Training requires declared FP32 latent provenance and matching quantization')
    index = path / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map'] if index.exists() else None
    if index.exists() and (not isinstance(mapping, dict) or not mapping):
        raise ValueError('Latent index requires a nonempty weight_map')
    names = sorted(set(mapping.values())) if mapping else [p.name for p in path.glob('*.safetensors')]
    if not names:
        raise ValueError('Training handoff requires FP32 latent safetensors')
    dtypes = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError('Invalid latent shard path')
        with safe_open(path / name, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key in dtypes or (mapping is not None and mapping.get(key) != name):
                    raise ValueError('Duplicate or misindexed latent tensor: ' + key)
                dtype = handle.get_slice(key).get_dtype()
                if dtype.startswith(('F', 'BF')) and dtype != 'F32':
                    raise ValueError('Training handoff must be FP32: ' + key)
                dtypes[key] = dtype
                if key.endswith('.weight_quantizer.fixed_codes_buf'):
                    if handle.get_tensor(key).item() != 0:
                        raise ValueError('Fixed integer codes are not continuous latent weights')
                if key.endswith('.weight_quantizer.wbits_buf'):
                    if handle.get_tensor(key).item() != bits:
                        raise ValueError('Quantizer bit width mismatch')
    if mapping is not None and set(mapping) != set(dtypes):
        raise ValueError('Missing indexed latent tensors')
    scales = [k for k in dtypes if k.endswith('.weight_quantizer.scale')]
    if not scales:
        raise ValueError('Training handoff lacks quantizer state')
    for key in scales:
        prefix = key.removesuffix('.weight_quantizer.scale')
        for suffix in ('weight', 'weight_quantizer.scale', 'weight_quantizer.zero_point'):
            if dtypes.get(prefix + '.' + suffix) != 'F32':
                raise ValueError('Incomplete FP32 latent quantizer: ' + prefix)
    return dict(status='PASS', floating_dtype='float32', tensor_count=len(dtypes),
                quantizer_count=len(scales), scope='precision and format only')


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(state):
    import torch

    result = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.is_meta:
            raise ValueError(f"Not a saved tensor: {key}")
        tensor = value.detach().to("cpu").contiguous().clone()
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"Latent training state must be FP32: {key}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint tensor: {key}")
        result[key] = tensor
    return result


def _validate_quantizers(state):
    scales = [key for key in state if key.endswith(".weight_quantizer.scale")]
    if not scales:
        raise ValueError("No quantizer state to preserve")
    for key in scales:
        prefix = key.removesuffix(".weight_quantizer.scale")
        for suffix in ("weight", "weight_quantizer.zero_point"):
            if prefix + "." + suffix not in state:
                raise ValueError(f"Incomplete quantizer: {prefix}")
    return len(scales)


def read_latent_metadata(path, *, bits, group_size):
    """Legacy inputs have no claim; new incomplete/mismatched inputs fail closed."""
    path = Path(path)
    manifest = path / MANIFEST
    if not manifest.is_file():
        return None
    record = json.loads(manifest.read_text())
    if (record.get("status") != "COMPLETE" or
            record.get("latent_weights_preserved") is not True or
            record.get("bits") != bits or record.get("group_size") != group_size):
        raise ValueError("Incomplete or mismatched latent checkpoint")
    if not record.get("files"):
        raise ValueError("Latent checkpoint has no recorded shards")
    for name, expected in record["files"].items():
        if Path(name).name != name or _hash(path / name) != expected:
            raise ValueError(f"Latent shard changed: {name}")
    return record


class BlockwiseLatentWriter:
    """Persist one pre-cast, pre-materialization block at a time on CPU."""

    def __init__(self, output, *, num_blocks, bits, group_size):
        if num_blocks < 1:
            raise ValueError("Expected at least one block")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.record = dict(schema_version=1, status="INCOMPLETE",
                           artifact_kind="qat_latent_model", full_training_resume=False,
                           optimizer_state_preserved=False, num_blocks=num_blocks,
                           bits=bits, group_size=group_size,
                           capture_boundary="before_dtype_cast_and_quant_inplace",
                           blocks=[], files={})
        self.weight_map = {}
        self.total_size = 0
        _json(self.output / MANIFEST, self.record)

    def _save_shard(self, name, state):
        from safetensors.torch import save_file

        if set(state) & self.weight_map.keys():
            raise ValueError("Duplicate latent checkpoint tensor")
        path = self.output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file(state, str(temporary), metadata={"format": "pt"})
        temporary.replace(path)
        self.record["files"][name] = _hash(path)
        self.weight_map.update(dict.fromkeys(state, name))
        self.total_size += sum(t.numel() * t.element_size() for t in state.values())

    def save_block(self, index, module):
        if self.record["status"] != "INCOMPLETE":
            raise ValueError("Latent checkpoint is already complete")
        if index in self.record["blocks"] or not 0 <= index < self.record["num_blocks"]:
            raise ValueError(f"Duplicate or invalid block: {index}")
        state = _snapshot(module.state_dict())
        _validate_quantizers(state)
        state = {f"model.layers.{index}.{key}": value for key, value in state.items()}
        self._save_shard(f"model-block-{index:05d}.safetensors", state)
        self.record["blocks"].append(index)
        _json(self.output / MANIFEST, self.record)

    def finalize(self, model, tokenizer):
        if (self.record["status"] != "INCOMPLETE" or
                sorted(self.record["blocks"]) != list(range(self.record["num_blocks"]))):
            raise ValueError("Incomplete or already finalized latent checkpoint")
        # Decoder states come only from pre-materialization snapshots. The live
        # decoder may now be fake-quantized or packed; never copy its weights.
        remainder = {key: value.float() if value.is_floating_point() else value
                     for key, value in model.state_dict().items()
                     if not key.startswith("model.layers.")}
        self._save_shard("model-non-block.safetensors", _snapshot(remainder))
        config = copy.deepcopy(model.config)
        config.qat_latent_checkpoint = dict(
            bits=self.record["bits"], group_size=self.record["group_size"],
            latent_weights_preserved=True, full_training_resume=False)
        config.save_pretrained(self.output)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(self.output)
        tokenizer.save_pretrained(self.output)
        # Detect missing/modified shards before publishing the HF index.
        for name, expected in self.record["files"].items():
            if _hash(self.output / name) != expected:
                raise ValueError(f"Latent shard changed: {name}")
        _json(self.output / "model.safetensors.index.json",
              dict(metadata=dict(total_size=self.total_size), weight_map=self.weight_map))
        self.record.update(status="COMPLETE", latent_weights_preserved=True)
        _json(self.output / MANIFEST, self.record)
        return self.output


def mark_materialized_export(output, latent_output):
    """Do not label a floating-point inference export as a latent checkpoint."""
    _json(Path(output) / "checkpoint_kind.json", dict(
        artifact_kind="materialized_inference", latent_weights_preserved=False,
        full_training_resume=False, latent_checkpoint=str(Path(latent_output).resolve())))


def require_training_save_contents(contents):
    required = {"model", "optimizer", "extra"}
    if contents is None or not required.issubset(contents):
        raise ValueError("QAT training checkpoints must save model, optimizer and extra state")
