"""Preserve the private sampler generator across an exhausted-epoch resume."""
import hashlib
import json
from pathlib import Path

import torch


def restore_epoch_boundary_loader(trainer):
    config = trainer.config
    if config.trainer.resume_mode != 'resume_path':
        raise ValueError('Epoch-boundary restore requires an explicit resume path')
    step = trainer.global_steps
    steps_per_epoch = len(trainer.train_dataloader)
    if step <= 0 or steps_per_epoch <= 0 or step % steps_per_epoch:
        return None
    source = Path(config.trainer.resume_from_path) / 'data.pt'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    state = torch.load(source, map_location='cpu', weights_only=False)
    trainer.train_dataloader.load_state_dict(state)
    # Finish the saved exhausted iterator, retaining its sampler RNG for the
    # next __iter__. Never discard a real batch to force an epoch transition.
    iterator = iter(trainer.train_dataloader)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError('Boundary checkpoint unexpectedly contains an unconsumed batch')
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError('Source dataloader checkpoint changed during restore')
    report = dict(status='PASS', source=str(source), source_sha256=source_hash,
                  global_step=step, next_epoch=step // steps_per_epoch,
                  remaining_batches=0, sampler_state_restored=True)
    destination = Path(config.trainer.default_local_dir) / 'epoch_boundary_resume.json'
    with destination.open('x') as handle:
        json.dump(report, handle, indent=2)
    print('QAT_EPOCH_BOUNDARY_RESTORE ' + json.dumps(report), flush=True)
    return report
