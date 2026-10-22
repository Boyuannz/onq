# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
"""Export a trusted FSDP fake-quant checkpoint using its recorded contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from OnQ.config.schema import ConfigError
from OnQ.paths import repository_root
from OnQ.quantization.export_contract import (
    TOKENIZER_FILES, file_sha256, native_export_contract, read_export_contract, tokenizer_identity,
)


SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def shard_files(actor: Path) -> list[Path]:
    indexed = []
    for path in actor.glob("model_world_size_*_rank_*.pt"):
        match = SHARD_RE.fullmatch(path.name)
        if match is None or path.stat().st_size == 0:
            raise ConfigError(f"invalid model shard: {path}")
        indexed.append((int(match[1]), int(match[2]), path))
    if not indexed:
        raise ConfigError(f"no model shards under {actor}")
    world = indexed[0][0]
    if (world < 1 or any(size != world for size, _, _ in indexed)
            or sorted(rank for _, rank, _ in indexed) != list(range(world))):
        raise ConfigError("incomplete or ambiguous FSDP shard rank coverage")
    return [path for _, _, path in sorted(indexed)]


def assemble_tensor(parts):
    """Honor shard metadata; never concatenate plain replicated buffers."""
    import torch
    first = parts[0]
    if hasattr(first, "placements"):
        from torch.distributed.tensor import Replicate, Shard
        if len(first.placements) != 1:
            raise ConfigError("export supports only one-dimensional DTensor meshes")
        if any(tuple(p.shape) != tuple(first.shape) or p.dtype != first.dtype
               or p.placements != first.placements for p in parts):
            raise ConfigError("DTensor metadata mismatch")
        local = [p.to_local().detach().cpu() for p in parts]
        placement = first.placements[0]
        if isinstance(placement, Replicate):
            merged = assemble_tensor(local)
        elif isinstance(placement, Shard):
            merged = torch.cat(local, dim=placement.dim)
        else:
            raise ConfigError(f"unsupported DTensor placement {placement}")
        if tuple(merged.shape) != tuple(first.shape):
            raise ConfigError("DTensor merged shape disagrees with metadata")
        return merged
    if hasattr(first, "local_shards"):
        shape = tuple(first.size())
        if any(tuple(p.size()) != shape for p in parts):
            raise ConfigError("ShardedTensor global shape mismatch")
        shards = [s for p in parts for s in p.local_shards()]
        shards.sort(key=lambda s: s.metadata.shard_offsets[0])
        cursor = 0
        tensors = []
        for shard in shards:
            offsets, sizes = shard.metadata.shard_offsets, shard.metadata.shard_sizes
            if (offsets != [cursor] + [0] * (len(shape) - 1)
                    or tuple(sizes[1:]) != shape[1:]
                    or tuple(shard.tensor.shape) != tuple(sizes)):
                raise ConfigError("unsupported, overlapping, or incomplete ShardedTensor slices")
            if tensors and shard.tensor.dtype != tensors[0].dtype:
                raise ConfigError("ShardedTensor dtype mismatch")
            cursor += sizes[0]
            tensors.append(shard.tensor.detach().cpu())
        if cursor != shape[0]:
            raise ConfigError("ShardedTensor row coverage mismatch")
        return torch.cat(tensors, dim=0)
    if not isinstance(first, torch.Tensor):
        raise ConfigError(f"unsupported checkpoint value {type(first).__name__}")
    if any(not isinstance(p, torch.Tensor) or p.dtype != first.dtype
           or not torch.equal(p, first) for p in parts[1:]):
        raise ConfigError("plain rank tensors differ; missing shard metadata")
    return first.detach().cpu()


def export_checkpoint(checkpoint: Path, output: Path, *, contract_path=None,
                      tokenizer_source=None, expected_bits=None, expected_group_size=None,
                      checkpoint_format='legacy') -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    checkpoint, output = checkpoint.resolve(), output.resolve()
    if checkpoint_format not in ('legacy', 'native_verl'):
        raise ConfigError('Unknown checkpoint format')
    native = checkpoint_format == 'native_verl'
    contract = (native_export_contract(checkpoint, contract_path) if native
                else read_export_contract(checkpoint, contract_path))
    for expected, key in ((expected_bits, "bits"), (expected_group_size, "group_size")):
        if expected is not None and expected != contract[key]:
            raise ConfigError(f"{key} disagrees with export contract")
    source = Path(tokenizer_source or contract["tokenizer_source"])
    if tokenizer_identity(source) != contract["tokenizer_files"]:
        raise ConfigError("tokenizer source does not match training identity")
    actor = checkpoint / "actor"
    for name in (("data.pt",) if native else ("data.pt", "training_state.json")):
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ConfigError(f"checkpoint save is incomplete: missing {path}")
    files = shard_files(actor)
    if native:
        from OnQ.integrations.verl_v080.utils.resume import checkpoint_inventory
        try:
            checkpoint_inventory(checkpoint, len(files))
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    config_dir = actor / "huggingface"
    if not (config_dir / "config.json").is_file():
        raise ConfigError("checkpoint is missing its own HF model config")
    config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
    if config.model_type != "qwen3":
        raise ConfigError("export supports Qwen3 checkpoints only")
    if output.exists():
        raise ConfigError(f"output directory must be absent: {output}")
    output.mkdir(parents=True)
    source_shards = [{"file": str(p), "sha256": file_sha256(p)} for p in files]
    shards = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
    if any(set(shard) != set(shards[0]) for shard in shards):
        raise ConfigError("FSDP shard key mismatch")
    merged = {key: assemble_tensor([shard[key] for shard in shards]) for key in shards[0]}
    del shards
    if native and any(value.is_floating_point() and value.dtype != torch.float32
                      for value in merged.values()):
        raise ConfigError('Native checkpoint must contain actual FP32 masters, not upcast exports')
    if native and any(not torch.isfinite(value).all().item() for value in merged.values()):
        raise ConfigError('Non-finite native checkpoint tensors')
    # Rehydrate the actor's FP32 master parameters before quantization, even
    # when an explicitly reviewed legacy artifact stored them at lower precision.
    merged = {key: value.float() if value.is_floating_point() else value
              for key, value in merged.items()}

    from OnQ.quantization.efficientqat_helpers import dequantize_efficientqat_weight
    dtype_names = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    output_dtype = dtype_names[contract["materialized_dtype"]]
    quant_dtype = dtype_names[contract["quant_math_dtype"]]
    suffix = ".weight_quantizer.scale"
    projections = {key.removesuffix(suffix) for key in merged if key.endswith(suffix)}
    expected = {f"model.layers.{layer}.{group}.{name}"
                for layer in range(config.num_hidden_layers)
                for group, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                                     ("mlp", ("gate_proj", "up_proj", "down_proj")))
                for name in names}
    if projections != expected:
        raise ConfigError("quantized projection coverage mismatch")
    fixed_states = []
    dense = {}
    for prefix in sorted(projections):
        for key in (prefix + ".weight", prefix + ".weight_quantizer.zero_point"):
            if key not in merged:
                raise ConfigError(f"incomplete quantizer state: {key}")
        if native:
            weight = merged[prefix + '.weight']
            group = contract['group_size']
            shape = (weight.numel() // group, 1)
            if (weight.ndim != 2 or weight.shape[-1] % group
                    or merged[prefix + suffix].shape != shape
                    or merged[prefix + '.weight_quantizer.zero_point'].shape != shape):
                raise ConfigError('Native quantizer group/shape mismatch: ' + prefix)
        wbits = merged.get(prefix + ".weight_quantizer.wbits_buf")
        fixed_buffer = merged.get(prefix + ".weight_quantizer.fixed_codes_buf")
        if native and (wbits is None or wbits.numel() != 1 or wbits.dtype != torch.int32
                       or fixed_buffer is None or fixed_buffer.numel() != 1
                       or fixed_buffer.dtype != torch.int8 or int(fixed_buffer.item()) != 0):
            raise ConfigError('Missing or invalid native latent quantizer metadata: ' + prefix)
        if wbits is not None and int(wbits.item()) != contract["bits"]:
            raise ConfigError("embedded bit width disagrees with contract")
        fixed = bool(fixed_buffer.item()) if fixed_buffer is not None else False
        fixed_states.append(fixed)
        dense[prefix + ".weight"] = dequantize_efficientqat_weight(
            merged[prefix + ".weight"], merged[prefix + suffix],
            merged[prefix + ".weight_quantizer.zero_point"],
            wbits=contract["bits"], group_size=contract["group_size"],
            fixed_integer_codes=fixed, compute_dtype=quant_dtype,
            scale_floor=contract["scale_floor"], scale_ceil=contract["scale_ceil"],
        ).to(output_dtype)
    if any(fixed_states) and not all(fixed_states):
        raise ConfigError("partially fixed-code checkpoint")
    for key, tensor in merged.items():
        if ".weight_quantizer." not in key and key not in dense:
            dense[key] = tensor.to(output_dtype) if tensor.is_floating_point() else tensor
    del merged
    if not all(torch.isfinite(t).all().item() for t in dense.values() if t.is_floating_point()):
        raise ConfigError("non-finite materialized weights")
    # Architecture comes from the checkpoint, not a mutable HF hub revision.
    if native:
        # Inference exports must not advertise a training adapter or teacher.
        # The original config's identity remains in the export manifest.
        for key in tuple(vars(config)):
            if key.startswith('qat_'):
                delattr(config, key)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=output_dtype)
    model.load_state_dict(dense, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise ConfigError("dense model state mismatch: uninitialized parameters")
    from accelerate.utils import other as accelerate_other
    original_probe = accelerate_other.is_deepspeed_available
    accelerate_other.is_deepspeed_available = lambda: False
    try:
        model.save_pretrained(output, safe_serialization=True)
    finally:
        accelerate_other.is_deepspeed_available = original_probe
    for name in TOKENIZER_FILES:
        if name not in contract["tokenizer_files"]:
            (output / name).unlink(missing_ok=True)
    for name in contract["tokenizer_files"]:
        shutil.copyfile(source / name, output / name)
    AutoTokenizer.from_pretrained(output, local_files_only=True, use_fast=True)
    if tokenizer_identity(output) != contract["tokenizer_files"]:
        raise ConfigError("exported tokenizer identity changed")
    manifest = {"status": "EXPORTED", "artifact_kind": "evaluation_dense",
                "checkpoint_format": checkpoint_format,
                "resumable": False, "checkpoint": str(checkpoint), "contract": contract,
                "source_shards": source_shards,
                "config_sha256": file_sha256(config_dir / "config.json"),
                "output_files": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
                "parity_status": "REQUIRES_LOGPROB_VALIDATION"}
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--tokenizer-source", type=Path)
    parser.add_argument("--expected-bits", type=int)
    parser.add_argument("--expected-group-size", type=int)
    parser.add_argument("--checkpoint-format", choices=('legacy', 'native_verl'), default='legacy',
                        help='native_verl reads the saved QAT HF config and full native shard inventory')
    args = parser.parse_args()
    result = export_checkpoint(args.checkpoint, args.output, contract_path=args.contract,
                               tokenizer_source=args.tokenizer_source, expected_bits=args.expected_bits,
                               expected_group_size=args.expected_group_size,
                               checkpoint_format=args.checkpoint_format)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
