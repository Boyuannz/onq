"""Opt-in dynamic-KL callbacks for the pinned veRL FSDP eager-logits path."""
import torch



def response_prediction_mask(data):
    """Align response mask to logits that predict each actual response token.

    Match veRL no_padding_2_padding's one-position shift independently per
    sequence. An actual EOS is retained; no end token is invented for truncation.
    """
    inputs = data['input_ids']
    if not inputs.is_nested:
        raise ValueError('Dynamic KL currently requires native jagged input IDs')
    mask = data['response_mask']
    mask = mask.to_padded_tensor(False) if mask.is_nested else mask
    prompts, responses = data['prompts'], data['responses']
    if prompts.is_nested:
        prompt_lens = prompts.offsets().diff()
        response_lens = responses.offsets().diff()
    else:
        attention = data['attention_mask']
        prompt_lens = attention[:, :prompts.shape[1]].sum(1)
        response_lens = attention[:, prompts.shape[1]:].sum(1)
    offsets = inputs.offsets()
    if (not torch.equal(offsets.diff(), prompt_lens + response_lens)
            or mask.shape[0] != len(prompt_lens) or (prompt_lens <= 0).any()):
        raise ValueError('Dynamic KL prompt/response lengths do not match inputs')
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError('Dynamic KL response mask must be binary')
    out = torch.zeros(inputs.values().shape[0], dtype=torch.bool, device=inputs.device)
    for i, (end, length) in enumerate(zip(offsets[1:].tolist(), response_lens.tolist())):
        if length > mask.shape[1] or mask[i, length:].any():
            raise ValueError('Active response mask extends beyond actual response')
        out[end - length - 1:end - 1] = mask[i, :length].bool()
    return out


def dynamic_kl_loss(*, config, distillation_config, options, vocab_size,
                    student_logits=None, model_output=None, data=None,
                    data_format='thd', dp_group=None, union=None):
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.utils.metric import Metric, AggregationType
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size
    from verl.workers.utils.padding import no_padding_2_padding

    loss_config = distillation_config.distillation_loss
    if (loss_config.loss_mode != 'forward_kl_topk' or loss_config.topk != 20
            or loss_config.use_policy_gradient or loss_config.use_task_rewards
            or loss_config.log_prob_min_clamp is not None or loss_config.loss_max_clamp is not None
            or config.loss_agg_mode != 'token-mean' or config.use_kl_loss
            or config.entropy_coeff != 0):
        raise ValueError('Dynamic KL requires pure, unclamped, token-mean direct KD')
    if get_ulysses_sequence_parallel_world_size() != 1:
        raise ValueError('Dynamic KL currently supports SP=1 only (multi-GPU DP is supported)')
    if student_logits is not None:
        if data_format != 'thd' or student_logits.ndim != 3 or student_logits.shape[0] != 1:
            raise ValueError('Dynamic KL expects [1, packed_tokens, vocabulary] logits')
        if union is None:
            raise ValueError('OnQ requires teacher/student union support')
        from .union_kl import union_token_losses
        result = union_token_losses(student_logits.squeeze(0), data, response_prediction_mask(data),
                                    vocab_size=vocab_size, options=options, spec=union)
        return {name: value.unsqueeze(0) for name, value in result.items()}

    losses = no_padding_2_padding(model_output['distillation_losses'], data)
    mask = data['response_mask']
    mask = mask.to_padded_tensor(False) if mask.is_nested else mask
    if losses.shape != mask.shape:
        raise ValueError('Dynamic KL final response alignment mismatch')
    info = config.global_batch_info
    if info.get('batch_num_tokens', 0) <= 0 or info.get('dp_size', 0) <= 0:
        raise ValueError('Dynamic KL requires the native global response-token denominator')
    # Native FSDP sums microbatch gradients and averages ranks. agg_loss already
    # supplies global token normalization and DP scaling; no second weighting.
    loss = agg_loss(loss_mat=losses, loss_mask=mask, loss_agg_mode='token-mean', **info)
    metrics = {'dynamic_kl/objective': Metric(AggregationType.SUM, loss.detach())}
    for name, value in model_output.items():
        if name.startswith('dynamic_') or name in ('student_mass', 'teacher_mass'):
            value = no_padding_2_padding(value, data).detach()
            reduced = agg_loss(loss_mat=value, loss_mask=mask, loss_agg_mode='token-mean', **info)
            metrics['dynamic_kl/' + name.removeprefix('dynamic_')] = Metric(AggregationType.SUM, reduced)
    return loss, metrics
