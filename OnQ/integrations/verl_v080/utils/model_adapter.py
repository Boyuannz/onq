"""Resolve QAT identity from the HF source and native model-config overrides."""
import json
import math

from omegaconf import OmegaConf


def adapter_from_config(config):
    encoded, plain = config.get('qat_adapter_json'), config.get('qat_adapter')
    if encoded is not None and plain is not None:
        raise ValueError('Ambiguous saved/current QAT adapter configuration')
    adapter = json.loads(encoded) if encoded is not None else plain
    if OmegaConf.is_config(adapter):
        adapter = OmegaConf.to_container(adapter, resolve=True)
    if not isinstance(adapter, dict):
        raise ValueError('Missing QAT adapter identity; cannot verify checkpoint objective')
    return adapter


def adapter_quant_lr(adapter):
    """Optional explicit LR for the quantizer (scale) parameter group.

    Absent means the quantizer group uses the shared actor LR, exactly as
    before this key existed. A present key must be a finite positive number;
    an explicit null or any other value fails closed.
    """
    if 'quant_lr' not in adapter:
        return None
    value = adapter['quant_lr']
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('QAT adapter quant_lr must be a finite positive number')
    return float(value)


def require_fp32_start(adapter):
    """Historical BF16/FP16 source-rounding runs are not resumable on main."""
    if adapter.get('initial_weight_dtype') not in (None, 'float32'):
        raise ValueError('Historical reduced-precision start (initial_weight_dtype='
                         f"{adapter.get('initial_weight_dtype')}) is not supported; "
                         'continue it on its original branch')


def effective_model_adapter(model):
    """Read config only; never load weights or silently ignore a missing source.

    veRL loads hf_config_path (falling back to path), then applies either the
    nested model_config override or the outer override. JSON adapter overrides
    replace the whole specification. A plain mapping override is unsupported:
    the pinned native updater recurses via getattr/setattr, not dictionary keys.
    Plain mappings already stored in the source HF config remain supported.
    """
    from transformers import AutoConfig

    source = model.get('hf_config_path')
    if source is None:
        source = model.get('path')
    if not source:
        raise ValueError('Missing HF config source; cannot verify effective QAT objective')
    override = model.get('override_config', {})
    try:
        saved = AutoConfig.from_pretrained(str(source),
            trust_remote_code=model.get('trust_remote_code', False),
            attn_implementation=override.get('attn_implementation', 'flash_attention_2')).to_dict()
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read HF config source for QAT validation: {source}') from exc
    override = override.get('model_config', override)
    if override.get('qat_adapter') is not None:
        raise ValueError('Use qat_adapter_json for native QAT overrides, not a mapping')
    # Other HF fields cannot alter QAT identity. Explicit nulls clear inherited
    # fields, just as the native scalar updater does; absent keys retain them.
    effective = {key: override.get(key, saved.get(key))
                 for key in ('qat_adapter_json', 'qat_adapter')}
    return adapter_from_config(effective)
