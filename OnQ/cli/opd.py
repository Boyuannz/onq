"""Prepare OnQ's complete training configuration in one command; no GPU launch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'teacher', 'data-dir', 'run-dir', 'active-dir', 'union-teacher-spec'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-size', choices=('1.7b', '4b', '8b'), required=True)
    parser.add_argument('--bits', type=int, choices=(2, 3, 4), required=True)
    parser.add_argument('--student-gpus', type=int, choices=(1, 4, 8), default=4)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--quant-lr', type=float)
    parser.add_argument('--rollout-gpu-mem-util', type=float, default=0.4)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from OnQ.integrations.verl_v080.opd.prepare import command
    from OnQ.integrations.verl_v080.opd.prepare_paper import configure
    from OnQ.integrations.verl_v080.utils.union_transport import validate_union_spec

    root = args.run_dir.expanduser().resolve()
    active = args.active_dir.expanduser().resolve()
    if root.exists() or args.run_dir.is_symlink() or not root.parent.is_dir():
        raise ValueError('Choose a new run directory under an existing checkpoint parent')
    if active.exists() or active.name != root.name or active.is_relative_to(root) or root.is_relative_to(active):
        raise ValueError('Choose a new, separate active directory with the same basename as the run')
    spec = validate_union_spec(json.loads(args.union_teacher_spec.read_text()))
    # Resolve the pinned native recipe internally. Only the final OnQ config is
    # exposed; this temporary base is never trained or presented as another stage.
    with tempfile.TemporaryDirectory(prefix='onq-config-') as temporary:
        base_root = Path(temporary) / 'base'
        argv, env, _ = command(args.source, args.model, args.teacher,
            args.data_dir / 'train.parquet', args.data_dir / 'validation.parquet',
            base_root, args.bits, sys.executable, student_gpus=args.student_gpus,
            batch_size=128, prompt_length=4096, response_length=16384,
            lr=args.lr, quant_lr=args.quant_lr, target_step=400, save_freq=50,
            teacher_url=spec['url'], teacher_token_file=spec['token_file'],
            teacher_ca_file=spec.get('ca_file'), teacher_request_timeout_seconds=float(spec['timeout']),
            rollout_gpu_mem_util=args.rollout_gpu_mem_util,
            wandb_dir=Path(temporary) / 'logs' / 'base')
        result = subprocess.run(argv + ['--cfg', 'job', '--resolve'], env=env,
            cwd=args.source, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError('Native configuration resolution failed:\n' + result.stderr)
        config = configure(OmegaConf.create(result.stdout), root=root, data_dir=args.data_dir,
            union_teacher=spec, active_dir=active, model_size=args.model_size)
        root.mkdir(exist_ok=False)
        OmegaConf.save(config, root / 'resolved.yaml')
        (root / 'resolve.log').write_text(result.stderr)
        (root / 'PREPARED.json').write_text(json.dumps(dict(status='PREPARED',
            model_size=args.model_size, bits=args.bits,
            steps=config.trainer.total_training_steps,
            objective='candidate-wise Dynamic KL on teacher/student top20 union plus rest'), indent=2) + '\n')
    print(f'Prepared {root / "resolved.yaml"}')


if __name__ == '__main__':
    main()
