"""Prepare paper Union OPD from a resolved fresh Pure OPD base (never launch)."""
import copy
import hashlib
import os
import json
from pathlib import Path
from omegaconf import open_dict
from .prompt_data import sha
from .schedule import V4_SCHEDULE
from ..utils.method_config import validate_method
from ..utils.paper_contract import verify_files


def configure_union(base, *, root, epsilon=1e-8, mass_tolerance=1e-6, chunk_size=128,
              union_teacher=None, active_dir=None):
    validate_method(base, 'gkd', require_external=True)
    c = copy.deepcopy(base)
    root = Path(root).resolve()
    if root == Path(base.trainer.default_local_dir).resolve():
        raise ValueError('Dynamic KL requires an independent output directory')
    active = Path(active_dir or Path.home()/'OnQ/runs/wandb_active'/root.name).expanduser().resolve()
    previous = Path(base.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR).resolve()
    if (active.name != root.name or active == previous
            or active.is_relative_to(root) or root.is_relative_to(active)):
        raise ValueError('Active log directory must be independent and match the new run basename')
    parent = active
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError('Active log parent is not writable; set --active-dir on this host')
    with open_dict(c):
        loss = c.distillation.distillation_loss
        loss.log_prob_min_clamp = None
        loss.loss_max_clamp = None
        override = c.actor_rollout_ref.model.override_config
        # Existing resolved configs are never migrated implicitly. Only this
        # fresh-run preparation defaults to the accepted union implementation.
        if not override.get('qat_adapter_json') or override.get('qat_adapter') is not None:
            raise ValueError('Expected one unambiguous qat_adapter_json configuration')
        spec = json.loads(override.qat_adapter_json)
        spec['dynamic_kl'] = dict(epsilon=epsilon, mass_tolerance=mass_tolerance, chunk_size=chunk_size)
        from ..utils.union_transport import validate_union_spec
        spec['union'] = copy.deepcopy(validate_union_spec(union_teacher))
        override.qat_adapter_json = json.dumps(spec)
        c.trainer.default_local_dir = str(root)
        c.trainer.experiment_name = root.name
        c.trainer.resume_mode = 'disable'
        c.trainer.resume_from_path = None
        c.qat_restore_epoch_boundary_dataloader = False
        c.actor_rollout_ref.rollout.trace.experiment_name = root.name
        c.ray_kwargs.ray_init._temp_dir = '/tmp/' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        c.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR = str(active)
    validate_method(c, 'dynamic_kl', require_external=True)
    return c


def configure(base, *, root, data_dir, union_teacher, active_dir, model_size):
    from ..utils.paper_contract import MODEL_STEPS
    if model_size not in MODEL_STEPS:
        raise ValueError('Expected model size 1.7b, 4b or 8b')
    validate_method(base, 'opd', require_external=True)
    gkd = copy.deepcopy(base)
    with open_dict(gkd):
        gkd.distillation.distillation_loss.loss_mode = 'forward_kl_topk'
        gkd.distillation.distillation_loss.use_policy_gradient = False
        gkd.distillation.distillation_loss.topk = 20
        gkd.actor_rollout_ref.model.use_fused_kernels = False
    c = configure_union(gkd, root=root, union_teacher=union_teacher, active_dir=active_dir,
                mass_tolerance=1e-5)
    data_dir, root = Path(data_dir).resolve(), Path(root).resolve()
    m = json.loads((data_dir / 'manifest.json').read_text())
    if (m['prompts'], m['appearances'], m['rows'], m['validation_prompts'], m['seed'], m['schedule']) != (
            25600, 2, 51200, 256, 42, V4_SCHEDULE):
        raise ValueError('Unexpected paper data manifest')
    with open_dict(c):
        adapter = json.loads(c.actor_rollout_ref.model.override_config.qat_adapter_json)
        adapter['r8_schedule'] = dict(version=1, schedule=V4_SCHEDULE, validation_cap=16384)
        c.actor_rollout_ref.model.override_config.qat_adapter_json = json.dumps(adapter)
        c.qat_r8_plan = dict(version=1, model_size=model_size, manifest_sha256=sha(data_dir / 'manifest.json'),
            train_sha256=m['files']['train.parquet'], validation_sha256=m['files']['validation.parquet'],
            original_train_sha256=m['original_train_sha256'],
            original_validation_sha256=m['original_validation_sha256'], validation_ids=m['validation_ids'])
        c.data.update(train_files=[str(data_dir / 'train.parquet')],
                      val_files=[str(data_dir / 'validation.parquet')], shuffle=False,
                      train_batch_size=128, val_batch_size=64, seed=42, filter_overlong_prompts=False,
                      truncation='error', max_prompt_length=4096, max_response_length=16384)
        actor, rollout = c.actor_rollout_ref.actor, c.actor_rollout_ref.rollout
        actor.shuffle = False
        actor.ppo_mini_batch_size = 128
        actor.ppo_max_token_len_per_gpu = 20480
        actor.fsdp_config.max_token_len_per_gpu = 20480
        rollout.update(response_length=16384, prompt_length=4096, max_model_len=20480,
                       log_prob_max_token_len_per_gpu=20480, max_num_seqs=128,
                       max_num_batched_tokens=20480)
        rollout.val_kwargs.update(temperature=0., top_p=1., top_k=-1, n=1, do_sample=False)
        c.trainer.update(total_epochs=1, total_training_steps=MODEL_STEPS[model_size], save_freq=50,
                         test_freq=50, val_before_train=False, max_actor_ckpt_to_keep=None,
                         del_local_ckpt_after_load=False, rollout_data_dir=str(root / 'rollouts'))
        c.reward.reward_model.enable = False
        c.reward.custom_reward_function.path = str(Path(__file__).with_name('no_reward.py'))
        c.reward.custom_reward_function.name = 'compute_score'
        env = c.ray_kwargs.ray_init.runtime_env.env_vars
        env.WANDB_MODE = 'offline'
        for key in ('WANDB_RUN_ID', 'WANDB_RESUME', 'QAT_WEIGHT_AUDIT_DIR'):
            env.pop(key, None)
    validate_method(c, 'dynamic_kl', require_external=True)
    verify_files(c)
    return c
