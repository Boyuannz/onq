"""Validate native QAT resume identity before allocating workers or loading shards."""
import json
import re
from pathlib import Path

from omegaconf import OmegaConf

from .method_config import validate_method
from .model_adapter import adapter_from_config, adapter_quant_lr, effective_model_adapter, require_fp32_start


def checkpoint_objective(actor_path, adapter):
    """Check the HF config already saved by native FSDP, before any state load.

    A model-only start is not a native optimizer resume. Missing metadata is
    never interpreted as evidence that a checkpoint used ordinary GKD.
    """
    source = Path(actor_path) / 'huggingface/config.json'
    if not source.is_file():
        raise ValueError('Missing saved HF config; cannot verify checkpoint objective')
    saved = json.loads(source.read_text())
    original = adapter_from_config(saved)
    if original.get('union') != adapter.get('union'):
        raise ValueError('Union objective/teacher cannot change on native resume')
    if original.get('mixed_observation') != adapter.get('mixed_observation'):
        raise ValueError('Mixed worker schedule/data contract cannot change on native resume')
    if original.get('r8_schedule') != adapter.get('r8_schedule'):
        raise ValueError('paper curriculum/validation cannot change on native resume')
    old_dynamic, new_dynamic = original.get('dynamic_kl'), adapter.get('dynamic_kl')
    if old_dynamic != new_dynamic:
        raise ValueError('Checkpoint objective mismatch: dynamic KL and native GKD/PG '
                         'or different dynamic options cannot be silently resumed')
    require_fp32_start(original)
    require_fp32_start(adapter)
    # Native optimizer restore reloads saved group LRs; a changed quant_lr
    # would be silently discarded, so it is part of the resume contract.
    if adapter_quant_lr(original) != adapter_quant_lr(adapter):
        raise ValueError('Quantizer learning rate (quant_lr) cannot change on native resume')
    for key, default in (('bits', None), ('group_size', 128), ('scale_floor', 1e-4)):
        if original.get(key, default) != adapter.get(key, default):
            raise ValueError('Checkpoint quantization contract mismatch: ' + key)
    if saved.get('qat_quant_input_dtype') != 'float32':
        raise ValueError('Checkpoint quantization input precision is not the current FP32 contract')
    return dict(status='PASS', source=str(source), dynamic_kl=old_dynamic,
                quant_input_dtype='float32')


def checkpoint_inventory(checkpoint, world_size):
    checkpoint = Path(checkpoint).resolve()
    match = re.fullmatch(r'global_step_([1-9][0-9]*)', checkpoint.name)
    if match is None:
        raise ValueError('Resume requires an explicit native global_step_N directory')
    if type(world_size) is not int or world_size <= 0:
        raise ValueError('Invalid resume world size')
    files = [checkpoint / 'data.pt', checkpoint / 'actor/huggingface/config.json']
    files += [checkpoint / 'actor' / f'{kind}_world_size_{world_size}_rank_{rank}.pt'
              for rank in range(world_size) for kind in ('model', 'optim', 'extra_state')]
    if any(not p.is_file() or p.stat().st_size == 0 for p in files):
        raise ValueError('Incomplete native checkpoint or different world size: '
                         'model, optimizer, extra state and data.pt are all required')
    return int(match.group(1))


def validate_resume(config, mode):
    """Read-only admission, shared by preparation, supervisor and direct runner."""
    resume_mode = config.trainer.get('resume_mode', 'disable')
    if resume_mode == 'disable':
        return None
    if resume_mode != 'resume_path':
        if mode == 'dynamic_kl':
            raise ValueError('Dynamic KL recovery requires explicit resume_path, never auto')
        return None
    if not config.trainer.get('resume_from_path'):
        raise ValueError('Missing explicit resume checkpoint')
    checkpoint = Path(config.trainer.resume_from_path).resolve()
    world_size = config.trainer.get('nnodes', 1) * config.trainer.n_gpus_per_node
    step = checkpoint_inventory(checkpoint, world_size)
    if config.trainer.total_training_steps <= step:
        raise ValueError('Resume target must exceed the saved step; total steps are absolute')
    if config.trainer.get('del_local_ckpt_after_load', False):
        raise ValueError('Recovery must preserve the source checkpoint')
    contents = config.actor_rollout_ref.actor.get('checkpoint', {}).get(
        'load_contents', ['model', 'optimizer', 'extra'])
    if not {'model', 'optimizer', 'extra'}.issubset(contents):
        raise ValueError('Recovery must load model, optimizer, scheduler and RNG (extra)')
    adapter = effective_model_adapter(config.actor_rollout_ref.model)
    identity = checkpoint_objective(checkpoint / 'actor', adapter)
    if mode == 'dynamic_kl':
        if not config.get('qat_restore_epoch_boundary_dataloader', False):
            raise ValueError('Dynamic KL recovery requires epoch-boundary dataloader restoration')
        source_config = checkpoint.parent / 'resolved.yaml'
        if not source_config.is_file():
            raise ValueError('Missing original dynamic resolved.yaml next to checkpoint run')
        original = OmegaConf.load(source_config)
        validate_method(original, mode, require_external=True)
        if Path(original.trainer.default_local_dir).resolve() != checkpoint.parent:
            raise ValueError('Original config does not belong to the checkpoint run')
        # Preserve the original absolute target and scheduler/data contract.
        # Output locations and tracing/transport audit hooks may change.
        fields = ('trainer.total_training_steps', 'trainer.total_epochs',
                  'qat_mixed_plan', 'qat_r8_plan', 'trainer.test_freq',
                  'actor_rollout_ref.rollout.val_kwargs',
                  'trainer.n_gpus_per_node', 'trainer.nnodes', 'data', 'algorithm',
                  'actor_rollout_ref.actor', 'distillation', 'opd_external_teacher_json',
                  'actor_rollout_ref.model.path', 'actor_rollout_ref.model.hf_config_path',
                  'actor_rollout_ref.model.tokenizer_path',
                  'actor_rollout_ref.model.override_config',
                  'actor_rollout_ref.rollout.n', 'actor_rollout_ref.rollout.temperature',
                  'actor_rollout_ref.rollout.top_k', 'actor_rollout_ref.rollout.top_p',
                  'actor_rollout_ref.rollout.agent.num_workers',
                  'actor_rollout_ref.rollout.response_length')
        for key in fields:
            if OmegaConf.select(original, key) != OmegaConf.select(config, key):
                raise ValueError('Dynamic resume changed the original training contract: ' + key)
    return dict(status='PASS', checkpoint=str(checkpoint), saved_step=step,
                next_step=step + 1, target_step=config.trainer.total_training_steps,
                world_size=world_size, objective=identity,
                boundary='metadata and file inventory; actual state loading requires a resume drill')
