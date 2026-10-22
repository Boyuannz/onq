# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from OnQ.utils import sha256_file as _sha256


CUSTOM_RECIPES = {"filtered_reasoning", "math_train", "math_ot_mix"}


def _official_module():
    import datautils_block

    return datautils_block




def _load_split_records(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {"train": [], "val": []}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get("meta") or {}
            split = meta.get("filtered_mix_split")
            if split not in result:
                raise ValueError(
                    f"{path}:{line_number}: expected meta.filtered_mix_split "
                    "to be 'train' or 'val'"
                )
            if not meta.get("decontaminated"):
                raise ValueError(
                    f"{path}:{line_number}: calibration record is not marked "
                    "decontaminated"
                )
            conversations = record.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError(
                    f"{path}:{line_number}: expected a prompt/response conversation"
                )
            if conversations[0].get("role") != "user":
                raise ValueError(f"{path}:{line_number}: first role must be user")
            if conversations[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_number}: final role must be assistant")
            result[split].append(record)
    return result


def _tokenize(tokenizer, records: list[dict], seqlen: int) -> list[tuple]:
    loader = []
    for index, record in enumerate(records):
        rendered = tokenizer.apply_chat_template(
            record["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(rendered, return_tensors="pt")
        token_count = int(encoded.input_ids.shape[1])
        if token_count < seqlen:
            raise ValueError(
                f"calibration record {index} has {token_count} tokens; "
                f"at least {seqlen} are required"
            )
        input_ids = encoded.input_ids[:, :seqlen]
        loader.append((input_ids, input_ids.clone()))
    return loader


def _validate_loader(name: str, loader: list[tuple], size: int, seqlen: int) -> None:
    if len(loader) != size:
        raise RuntimeError(f"{name}: expected {size} records, got {len(loader)}")
    for index, (input_ids, targets) in enumerate(loader):
        expected_shape = (1, seqlen)
        if tuple(input_ids.shape) != expected_shape or tuple(targets.shape) != expected_shape:
            raise RuntimeError(
                f"{name}[{index}] has shapes {tuple(input_ids.shape)} and "
                f"{tuple(targets.shape)}; expected {expected_shape}"
            )


def resolve_calibration_split(
    train_size: int,
    val_size: int,
    reasoning_ratio: float,
    *,
    reasoning_train_size: int | None = None,
    general_train_size: int | None = None,
    reasoning_val_size: int | None = None,
    general_val_size: int | None = None,
) -> tuple[int, int, int, int]:
    explicit = (
        reasoning_train_size,
        general_train_size,
        reasoning_val_size,
        general_val_size,
    )
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("explicit calibration split requires all four sizes")
        split = tuple(int(value) for value in explicit if value is not None)
    else:
        reasoning_train = int(train_size * reasoning_ratio)
        reasoning_val = int(val_size * reasoning_ratio)
        split = (
            reasoning_train,
            train_size - reasoning_train,
            reasoning_val,
            val_size - reasoning_val,
        )
    if min(split) < 0:
        raise ValueError(f"calibration split sizes must be nonnegative: {split}")
    if split[0] + split[1] != train_size or split[2] + split[3] != val_size:
        raise ValueError(
            f"calibration split {split} does not match train={train_size}, val={val_size}"
        )
    return split


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in (None, "") else int(value)


def _custom_loaders(
    name,
    tokenizer,
    train_size,
    val_size,
    seed,
    seqlen,
    test_only,
):
    if test_only:
        raise ValueError(f"{name} is calibration-only")
    path_value = os.environ.get("CALIBRATION_CALIBRATION_PATH")
    if not path_value:
        raise RuntimeError("CALIBRATION_CALIBRATION_PATH is required")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    split_records = _load_split_records(path)

    if name == "math_train":
        reasoning_ratio = 1.0
        general_source = None
    elif name == "filtered_reasoning":
        reasoning_ratio = 0.8
        general_source = "fineweb_edu"
    elif name == "math_ot_mix":
        reasoning_ratio = 0.8
        general_source = "openthoughts"
    else:
        raise ValueError(f"unknown custom Calibration recipe {name!r}")

    (
        reasoning_train_size,
        general_train_size,
        reasoning_val_size,
        general_val_size,
    ) = resolve_calibration_split(
        train_size,
        val_size,
        reasoning_ratio,
        reasoning_train_size=_optional_env_int("CALIBRATION_REASONING_TRAIN_SIZE"),
        general_train_size=_optional_env_int("CALIBRATION_GENERAL_TRAIN_SIZE"),
        reasoning_val_size=_optional_env_int("CALIBRATION_REASONING_VAL_SIZE"),
        general_val_size=_optional_env_int("CALIBRATION_GENERAL_VAL_SIZE"),
    )
    if len(split_records["train"]) < reasoning_train_size:
        raise RuntimeError(
            f"need {reasoning_train_size} reasoning train records, "
            f"found {len(split_records['train'])}"
        )
    if len(split_records["val"]) < reasoning_val_size:
        raise RuntimeError(
            f"need {reasoning_val_size} reasoning val records, "
            f"found {len(split_records['val'])}"
        )

    trainloader = _tokenize(
        tokenizer, split_records["train"][:reasoning_train_size], seqlen
    )
    valloader = _tokenize(
        tokenizer, split_records["val"][:reasoning_val_size], seqlen
    )
    official = _official_module()
    if general_source == "fineweb_edu":
        general_train, general_val = official.get_redpajama_concat(
            tokenizer,
            general_train_size,
            general_val_size,
            seed,
            seqlen,
            test_only=False,
        )
    elif general_source == "openthoughts":
        general_train, general_val = official.get_openthoughts_shuffled(
            tokenizer,
            general_train_size,
            general_val_size,
            seed,
            seqlen,
            test_only=False,
        )
    else:
        general_train, general_val = [], []

    trainloader.extend(general_train)
    valloader.extend(general_val)
    random.Random(seed).shuffle(trainloader)
    random.Random(seed + 1).shuffle(valloader)
    _validate_loader("trainloader", trainloader, train_size, seqlen)
    _validate_loader("valloader", valloader, val_size, seqlen)
    print(
        json.dumps(
            {
                "recipe": name,
                "calibration_path": str(path),
                "calibration_sha256": _sha256(path),
                "reasoning_train": reasoning_train_size,
                "general_train": general_train_size,
                "reasoning_val": reasoning_val_size,
                "general_val": general_val_size,
                "general_source": general_source,
                "seqlen": seqlen,
                "seed": seed,
            },
            sort_keys=True,
        )
    )
    return trainloader, valloader


def get_loaders(
    name,
    tokenizer,
    train_size=128,
    val_size=64,
    seed=0,
    seqlen=2048,
    test_only=False,
    **kwargs,
):
    if name == "paper_tokens":
        from .token_cache import load_windows
        return load_windows(os.environ["CALIBRATION_CALIBRATION_PATH"], tokenizer, train_size, val_size, seqlen)
    if name in CUSTOM_RECIPES:
        return _custom_loaders(
            name,
            tokenizer,
            train_size,
            val_size,
            seed,
            seqlen,
            test_only,
        )
    return _official_module().get_loaders(
        name,
        tokenizer,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        seqlen=seqlen,
        test_only=test_only,
        **kwargs,
    )
