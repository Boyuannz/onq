"""Held-out on-policy Union objective: native infer_batch, never update_actor."""
import json
import math
from pathlib import Path


def summarize(parts, expected_ids):
    ids = [row['prompt_id'] for p in parts for row in p['rows']]
    if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError('Validation coverage is missing, duplicated or foreign')
    tokens = sum(p['tokens'] for p in parts)
    if tokens <= 0:
        raise ValueError('No validation tokens')
    keys = set(parts[0]['loss_metrics'])
    if any(set(p['loss_metrics']) != keys for p in parts):
        raise ValueError('Validation metrics changed between batches')
    means = {k: sum(p['loss_metrics'][k] * p['tokens'] for p in parts) / tokens for k in keys}
    if not all(math.isfinite(v) for v in means.values()):
        raise ValueError('Nonfinite validation loss')
    rows = [row for p in parts for row in p['rows']]
    return dict(prompts=len(ids), valid_tokens=tokens, loss_metrics=means,
                mean_response_length=tokens / len(ids),
                truncation_rate=sum(r['finish_reason'] == 'length' for r in rows) / len(ids), rows=rows)


def reduce_forward_metrics(metrics):
    """Reduce compute_log_prob metrics like native train_mini_batch does.

    Forward-only worker output is allgathered per DP rank ({key: [Metric per rank]});
    native update_actor already applies Metric.aggregate_dp (mean over ranks, then
    the Metric's own aggregation over micro-batches). reduce_metrics alone would
    call np.mean on Metric objects. Non-Metric values keep native reduce_metrics.
    """
    from verl.utils.metric import reduce_metrics
    from verl.utils.metric.utils import Metric
    metrics = dict(metrics)
    for key, value in metrics.items():
        if isinstance(value, list) and value and all(isinstance(v, Metric) for v in value):
            metrics[key] = float(Metric.aggregate_dp(value))
    return reduce_metrics(metrics)


def rollout_rows(batch, *, cap):
    lengths = batch.batch['response_mask'].sum(-1).tolist()
    metadata = batch.non_tensor_batch
    reasons = list(metadata.get('qat_finish_reason', []))
    if (len(reasons) != len(lengths) or any(r not in ('stop', 'length') for r in reasons)
            or list(metadata.get('qat_generated_tokens', [])) != lengths
            or list(metadata.get('qat_response_cap', [])) != [cap] * len(lengths)
            or any(not 0 < n <= cap for n in lengths)):
        raise ValueError('Missing or inconsistent actual rollout termination metadata')
    return [dict(prompt_id=row['prompt_id'], length=int(length), finish_reason=reason)
            for row, length, reason in zip(metadata['extra_info'], lengths, reasons)]


def validate(trainer):
    import numpy as np
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding
    from verl.trainer.ppo.ray_trainer import compute_response_mask

    parts = []
    world = trainer.config.trainer.n_gpus_per_node
    expected = list(trainer.config.qat_r8_plan.validation_ids)
    for item in trainer.val_dataloader:
        batch = DataProto.from_single_dict(item)
        if len(batch) % world or len(batch) % trainer.config.actor_rollout_ref.rollout.agent.num_workers:
            raise ValueError('Validation must not duplicate samples to pad worker batches')
        batch.non_tensor_batch['uid'] = np.array(
            [f"validation:{r['prompt_id']}" for r in batch.non_tensor_batch['extra_info']], dtype=object)
        gen = trainer._get_gen_batch(batch)
        gen.meta_info.update(eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.pad_token_id, recompute_log_prob=False,
            do_sample=False, validate=True, global_steps=trainer.global_steps)
        generated = trainer.async_rollout_manager.generate_sequences(gen)
        batch = batch.union(generated)
        if 'response_mask' not in batch.batch:
            batch.batch['response_mask'] = compute_response_mask(batch)
        rows = rollout_rows(batch, cap=16384)
        batch.meta_info.update(temperature=1., multi_turn=False)
        packed = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(packed, compute_loss=True, calculate_entropy=False,
                            distillation_use_topk=True, global_batch_size=len(batch))
        trainer.checkpoint_manager.sleep_replicas()
        try:
            # compute_log_prob -> actor.infer_batch -> no_grad/forward_only.
            # Same QAT engine/loss/teacher as training; no optimizer or scheduler call.
            output = trainer.actor_rollout_wg.compute_log_prob(packed)
            metrics = reduce_forward_metrics(tu.get(output, 'metrics'))
            loss_metrics = {k: float(v) for k, v in metrics.items() if k.startswith('dynamic_kl/')}
            if 'dynamic_kl/objective' not in loss_metrics:
                raise ValueError('Union validation did not execute the training objective')
        finally:
            trainer.checkpoint_manager.update_weights(trainer.global_steps)
        parts.append(dict(tokens=sum(r['length'] for r in rows), rows=rows, loss_metrics=loss_metrics))
    result = summarize(parts, expected)
    result.update(step=trainer.global_steps, response_cap=16384, temperature=0,
                  objective_temperature=1, optimizer_updates=0, status='COMPLETE')
    root = Path(trainer.config.trainer.default_local_dir) / 'validation'
    root.mkdir(exist_ok=True)
    name = f'step{trainer.global_steps:04d}.json'
    with (root / name).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    metrics = {'val/' + k: v for k, v in result['loss_metrics'].items()}
    for k in ('prompts', 'valid_tokens', 'mean_response_length', 'truncation_rate'):
        metrics['val/' + k] = result[k]
    return metrics
