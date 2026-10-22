"""Prepare, never launch, recovery from the original dynamic run configuration."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from ..utils.method_config import validate_method
from ..utils.resume import validate_resume


def configure(checkpoint, *, root):
    checkpoint, root = Path(checkpoint).resolve(), Path(root).resolve()
    source = checkpoint.parent / 'resolved.yaml'
    if not source.is_file():
        raise ValueError('Recovery requires the original dynamic run resolved.yaml')
    base = OmegaConf.load(source)
    validate_method(base, 'dynamic_kl', require_external=True)
    previous = Path(base.trainer.default_local_dir).resolve()
    if root.exists() or root.is_relative_to(previous) or previous.is_relative_to(root):
        raise ValueError('Recovery requires a new independent output directory; never remove launch.lock')
    c = copy.deepcopy(base)
    with open_dict(c):
        c.trainer.default_local_dir = str(root)
        c.trainer.experiment_name = root.name
        c.trainer.resume_mode = 'resume_path'
        c.trainer.resume_from_path = str(checkpoint)
        c.trainer.del_local_ckpt_after_load = False
        c.qat_restore_epoch_boundary_dataloader = True
        c.actor_rollout_ref.rollout.trace.experiment_name = root.name
        c.ray_kwargs.ray_init._temp_dir = '/tmp/' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        env = c.ray_kwargs.ray_init.runtime_env.env_vars
        env.WANDB_DIR = str(Path(env.WANDB_DIR).parent / root.name)
        for key in ('WANDB_RUN_ID', 'WANDB_RESUME'):
            env.pop(key, None)
        # These optional older run annotations must not label a resume fresh or
        # direct a future archive hook back into its source run.
        if 'qat_fresh_latent_phase' in c:
            c.qat_fresh_latent_phase = False
        if c.get('qat_checkpoint_archive'):
            c.qat_checkpoint_archive.root = str(root)
        for key in ('rollout_data_dir', 'validation_data_dir'):
            if c.trainer.get(key):
                c.trainer[key] = str(root / key)
    validate_method(c, 'dynamic_kl', require_external=True)
    report = validate_resume(c, 'dynamic_kl')
    return c, dict(status='PREPARED', source_config=str(source),
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   resume=report, fresh_optimizer=False, source_checkpoint_preserved=True,
                   pending=['live GPU/process ownership and teacher health',
                            'writable output storage and checkpoint retention budget',
                            'all-rank native load then a finite optimizer update',
                            'restored sampler, LR/RNG, export and rollout synchronization'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Native global_step_N with original run resolved.yaml')
    p.add_argument('--run-dir', type=Path, required=True, help='New independent run; absolute target is inherited')
    a = p.parse_args()
    config, report = configure(a.checkpoint, root=a.run_dir)
    a.run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, a.run_dir / 'resolved.yaml')
    (a.run_dir / 'prepare_manifest.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
