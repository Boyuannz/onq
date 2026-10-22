"""Fail closed on retired methods and unsupported native loss configurations."""
from .model_adapter import adapter_quant_lr, effective_model_adapter


def validate_target_step(step, mode):
    if mode not in ('opd', 'gkd', 'dynamic_kl'):
        raise ValueError('Only Pure OPD, native top20 GKD and dynamic_kl are supported')
    if type(step) is not int or step <= 0:
        raise ValueError(f'{mode.upper()} target step must be a positive integer')


def validate_method(config, mode, *, require_external=False):
    validate_target_step(config.trainer.total_training_steps, mode)
    loss = config.distillation.distillation_loss
    if (not config.distillation.enabled or loss.use_task_rewards
            or config.actor_rollout_ref.actor.use_kl_loss
            or config.algorithm.get('use_kl_in_reward', False)):
        raise ValueError('Pure distillation requires no task rewards or additional KL')
    if require_external and not config.get('opd_external_teacher_json'):
        raise ValueError('OnQ requires an external teacher')
    actor = config.actor_rollout_ref.actor
    engine = actor.get('fsdp_config', {})
    # The plugin replaces only the native FSDP1 registry entry. Checking inside
    # QATFSDPEngine is too late: FSDP2 selects another class entirely.
    if (actor.get('strategy', 'fsdp') != 'fsdp'
            or engine.get('strategy') not in (None, 'fsdp')
            or engine.get('use_orig_params', False) is not True):
        raise ValueError('QAT requires FSDP1 (strategy=fsdp) with use_orig_params=True')
    adapter = effective_model_adapter(config.actor_rollout_ref.model)
    adapter_quant_lr(adapter)
    dynamic = adapter.get('dynamic_kl')
    union = adapter.get('union')
    if config.get('qat_r8_plan') or adapter.get('r8_schedule'):
        from .paper_contract import validate_paper
        validate_paper(config, adapter, mode)
    if union is not None:
        from .union_transport import validate_union_spec
        if mode != 'dynamic_kl' or dynamic is None:
            raise ValueError('Union requires the explicit dynamic_kl objective')
        validate_union_spec(union)
        from .union_protocol import required_input_tokens
        required_input_tokens(config)
    if adapter.get('mixed_observation') is not None or config.get('qat_mixed_plan'):
        raise ValueError('Historical mixed diagnostics are not supported by OnQ')
    if mode != 'dynamic_kl' and dynamic is not None:
        raise ValueError('Dynamic KL must use the explicit dynamic_kl mode')
    if mode == 'opd':
        if loss.loss_mode != 'k1' or not loss.use_policy_gradient:
            raise ValueError('OPD requires native k1-PG Pure OPD')
    elif (loss.loss_mode != 'forward_kl_topk' or loss.use_policy_gradient
            or type(loss.topk) is not int or loss.topk != 20
            or config.actor_rollout_ref.model.use_fused_kernels):
        raise ValueError('GKD requires native forward_kl_topk20, direct gradients and explicit logits')
    if mode == 'dynamic_kl':
        from .dynamic_kl import validate_options
        validate_options(dynamic)
        rollout = config.actor_rollout_ref.rollout
        # Native FSDPActorConfig lets the legacy field override the engine
        # field only when it is >1; otherwise the engine field is authoritative.
        legacy_sp = actor.get('ulysses_sequence_parallel_size', 1)
        if type(legacy_sp) is not int or legacy_sp < 1:
            raise ValueError('Invalid legacy sequence parallel size')
        sp = legacy_sp if legacy_sp > 1 else engine.get('ulysses_sequence_parallel_size', 1)
        # The pinned native FSDP engine clips with OptimizerConfig.clip_grad,
        # not FSDPActorConfig.grad_clip. Its deprecated optimizer-level alias
        # overrides clip_grad in OptimizerConfig.__post_init__.
        optimizer = actor.get('optim', {})
        effective_clip = optimizer.get('grad_clip')
        if effective_clip is None:
            effective_clip = optimizer.get('clip_grad', 1.)
        if (loss.log_prob_min_clamp is not None or loss.loss_max_clamp is not None
                or actor.loss_agg_mode != 'token-mean' or actor.entropy_coeff != 0
                or effective_clip != 1.
                or actor.ppo_epochs != 1 or actor.ppo_mini_batch_size != config.data.train_batch_size
                or rollout.n != 1 or rollout.temperature != 1.
                or type(sp) is not int or sp != 1
                or adapter.get('soft_weighting') or config.get('qat_soft_weighting')
                or config.actor_rollout_ref.model.external_lib != 'OnQ.integrations.verl_v080.utils.engine'):
            raise ValueError('Dynamic KL requires one T=1 on-policy update, SP=1, no clamps/auxiliary losses')
