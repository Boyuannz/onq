"""Check frozen random order without forcing legacy three-domain quotas."""
import json
import math
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from collections import Counter
from ..opd.schedule import V4_SCHEDULE, response_cap
from .paper_validation import rollout_rows


def compute_data_metrics(batch, use_critic=True, *, native_compute):
    metrics = native_compute(batch=batch, use_critic=use_critic)
    cap = int(batch.non_tensor_batch['qat_response_cap'][0])
    rows = rollout_rows(batch, cap=cap)
    # veRL pads responses to the configured 16K, even during the 4K/8K phases.
    cap_hit = sum(row['length'] == cap for row in rows) / len(rows)
    metrics.update({
        'response_length/clip_ratio': cap_hit,
        'response_length_non_aborted/clip_ratio': cap_hit,
        'response_length/actual_cap': cap,
        'response_length/truncation_rate': sum(
            row['finish_reason'] == 'length' for row in rows) / len(rows),
    })
    return metrics


@contextmanager
def curriculum_metrics(*, enabled):
    if not enabled:
        yield
        return
    from verl.trainer.ppo import ray_trainer
    original = ray_trainer.compute_data_metrics
    ray_trainer.compute_data_metrics = partial(compute_data_metrics, native_compute=original)
    try:
        yield
    finally:
        ray_trainer.compute_data_metrics = original


def update(trainer, batch, native_update):
    from verl.utils.metric import reduce_metrics
    step = trainer.global_steps
    info = batch.non_tensor_batch['extra_info']
    if (len(info) != 128 or {int(r['row_id']) for r in info} != set(range((step-1)*128, step*128))
            or any(r['validation'] or r['plan_step'] != step for r in info)):
        raise ValueError('paper training order changed or validation entered training')
    cap = response_cap(step, schedule=V4_SCHEDULE)
    rows = rollout_rows(batch, cap=cap)
    if any(r['response_cap'] != cap for r in info):
        raise ValueError('paper row curriculum mismatch')
    output = native_update(batch)
    metrics = reduce_metrics(output.meta_info['metrics'])
    if not math.isfinite(float(metrics['actor/grad_norm'])):
        raise ValueError('Nonfinite gradient; not recording a successful update')
    report = dict(status='COMPLETED_UPDATE', step=step, response_cap=cap,
                  tokens=sum(r['length'] for r in rows), metrics=metrics,
                  domain_counts=dict(Counter(r['domain'] for r in info)), rows=rows)
    with (Path(trainer.config.trainer.default_local_dir) / 'paper_completed_updates.jsonl').open('a') as stream:
        stream.write(json.dumps(report, allow_nan=False) + '\n')
    return output
