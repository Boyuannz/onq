"""Prepare native k1-PG Pure OPD with the existing QAT adapter; never launch."""
import json
import math
from pathlib import Path

from ..utils.prepare import command as native_command


def handoff_selection(model, *, resume_from=None):
    """Use the calibration latent directly; resume keeps native trainer state."""
    selected = str(Path(model).expanduser().resolve())
    return dict(selected_model=selected, selection='calibration',
                resume_from=str(Path(resume_from).expanduser().resolve()) if resume_from else None)


def command(source, model, teacher, train_file, val_file, run_dir, bits, python,
            *, student_gpus=4, teacher_gpus=1, student_tp=1, teacher_tp=1,
            batch_size=8, prompt_length=1024, response_length=8192, lr=1e-6,
            target_step=10, resume_from=None, teacher_url=None, teacher_token_file=None, teacher_ca_file=None,
            rollout_gpu_mem_util=0.4, disable_auto_wrap=False, save_freq=10,
            teacher_private_http_endpoint=None, teacher_request_timeout_seconds=180,
            wandb_dir=None,
            quant_lr=None):
    handoff = handoff_selection(model, resume_from=resume_from)
    model = handoff['selected_model']
    sizes = (student_gpus, teacher_gpus, student_tp, teacher_tp, batch_size,
             prompt_length, response_length, save_freq)
    if any(type(x) is not int or x <= 0 for x in sizes):
        raise ValueError('Topology, batch and lengths must be positive integers')
    if bits not in (2, 3, 4) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Expected W2/W3/W4 and a finite positive shared LR')
    if quant_lr is not None and (isinstance(quant_lr, bool) or not isinstance(quant_lr, (int, float))
                                 or not math.isfinite(quant_lr) or quant_lr <= 0):
        raise ValueError('Quantizer (scale) LR must be finite and positive when given')
    if student_gpus % student_tp or teacher_gpus % teacher_tp or batch_size % student_gpus:
        raise ValueError('TP must divide its pool and student world size must divide batch')
    if type(target_step) is not int or target_step <= 0 or (resume_from is not None and target_step == 1):
        raise ValueError('Target must be a positive integer; resume must target a step greater than one')
    if bool(teacher_url) != bool(teacher_token_file):
        raise ValueError('External teacher requires both URL and token-file path')
    if teacher_ca_file and not teacher_url:
        raise ValueError('Teacher CA file requires an external URL')
    if teacher_private_http_endpoint is not None and teacher_private_http_endpoint != teacher_url:
        raise ValueError('Private teacher HTTP authorization must match the exact teacher URL')
    if (not isinstance(teacher_request_timeout_seconds, (int, float)) or
            isinstance(teacher_request_timeout_seconds, bool) or
            not math.isfinite(teacher_request_timeout_seconds) or teacher_request_timeout_seconds <= 0):
        raise ValueError('Teacher request timeout must be finite and positive')
    if not math.isfinite(rollout_gpu_mem_util) or not 0 < rollout_gpu_mem_util <= 1:
        raise ValueError('Rollout GPU memory utilization must be in (0, 1]')
    if disable_auto_wrap and student_gpus != 1:
        raise ValueError('Root-only FSDP is currently accepted only for single-GPU diagnostics')
    active_wandb = (Path(wandb_dir).expanduser() if wandb_dir is not None else
                    Path.home() / 'OnQ/runs/wandb_active' / Path(run_dir).name).resolve()
    if active_wandb.is_relative_to(Path(run_dir).expanduser().resolve()):
        raise ValueError('Active W&B files require a local directory outside the checkpoint run')
    argv, env = native_command(source, model, train_file, val_file, run_dir, bits, python,
                               quant_lr=quant_lr)
    argv[1] = str(Path(source).resolve() / 'examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh')
    # Explicitly fix every upstream environment override instead of inheriting
    # a previous experiment's loss, LR or resource allocation from the shell.
    settings = dict(
        STUDENT_MODEL=str(Path(model).resolve()), TEACHER_MODEL=str(Path(teacher).resolve()),
        NNODES='1', NGPUS_PER_NODE=str(student_gpus), TEACHER_WORLD_SIZE=str(teacher_gpus),
        DISTILLATION_LOSS_MODE='k1', USE_POLICY_GRADIENT='True', DISTILLATION_TOPK='64',
        TRAIN_BATCH_SIZE=str(batch_size), PPO_MINI_BATCH_SIZE=str(batch_size),
        MAX_PROMPT_LENGTH=str(prompt_length), MAX_RESPONSE_LENGTH=str(response_length),
        PPO_MAX_TOKEN_LEN_PER_GPU=str(prompt_length + response_length), ACTOR_LR=str(lr),
        ROLLOUT_TP=str(student_tp), ROLLOUT_GPU_MEM_UTIL=str(rollout_gpu_mem_util),
        TEACHER_TP=str(teacher_tp), TEACHER_GPU_MEM_UTIL='0.4',
        TOTAL_EPOCHS=str(max(15, target_step)), SAVE_FREQ=str(save_freq), TEST_FREQ='-1',
        PROJECT_NAME='qat-native-opd', EXPERIMENT_NAME=Path(run_dir).name,
        WANDB_MODE='offline', WANDB_DIR=str(active_wandb),
    )
    env.update(settings)
    env.pop('RAY_ADDRESS', None)
    argv += [
        'data.train_files=' + json.dumps([str(Path(train_file).resolve())]),
        'data.val_files=' + json.dumps([str(Path(val_file).resolve())]),
        'distillation.enabled=True',
        'distillation.distillation_loss.loss_mode=k1',
        'distillation.distillation_loss.use_policy_gradient=True',
        'distillation.distillation_loss.use_task_rewards=False',
        'distillation.distillation_loss.loss_max_clamp=10.0',
        'distillation.distillation_loss.log_prob_min_clamp=-10.0',
        'actor_rollout_ref.actor.use_kl_loss=False',
        'actor_rollout_ref.actor.fsdp_config.param_offload=False',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False',
        'algorithm.use_kl_in_reward=False',
        'actor_rollout_ref.rollout.temperature=1.0',
        'actor_rollout_ref.rollout.n=1',
        'actor_rollout_ref.rollout.calculate_log_probs=True',
        'actor_rollout_ref.actor.ppo_epochs=1',
        'actor_rollout_ref.actor.optim.lr_scheduler_type=constant',
        'actor_rollout_ref.actor.optim.lr_warmup_steps=0',
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0',
        'distillation.teacher_models.teacher_model.inference.temperature=1.0',
        'trainer.total_training_steps=' + str(target_step),
        'trainer.logger=[console,wandb]',
        '++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE=offline',
        '++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR=' + settings['WANDB_DIR'],
        'ray_kwargs.ray_init.num_cpus=' + str(max(12, 6 * student_gpus)),
        '+ray_kwargs.ray_init.address=local',
    ]
    if resume_from is not None:
        argv += ['trainer.resume_mode=resume_path',
                 '+qat_restore_epoch_boundary_dataloader=True',
                 'trainer.resume_from_path=' + str(Path(resume_from).resolve())]
    if disable_auto_wrap:
        argv += ['+actor_rollout_ref.actor.fsdp_config.wrap_policy.disable=True']
    if teacher_url:
        external = dict(base_url=teacher_url, model_name='qat-opd-dense-qwen3-4b',
                        token_file=str(Path(teacher_token_file).resolve()))
        if teacher_ca_file:
            external['ca_file'] = str(Path(teacher_ca_file).resolve())
        if teacher_private_http_endpoint is not None:
            external['private_http_endpoint'] = teacher_private_http_endpoint
        external['request_timeout_seconds'] = teacher_request_timeout_seconds
        argv += ["+opd_external_teacher_json='" + json.dumps(external, separators=(',', ':')) + "'"]
    return argv, env, settings
