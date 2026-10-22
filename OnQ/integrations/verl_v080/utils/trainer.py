"""Run OnQ with the native veRL trainer and external teacher."""
import argparse
from contextlib import contextmanager
import json
import warnings
from pathlib import Path

import torch

from .version_guard import assert_pinned_upstream
from .method_config import validate_method

assert_pinned_upstream()


@contextmanager
def tracking_lifecycle():
    """Finish W&B before the owning Ray task returns, not during worker teardown."""
    from verl.utils import tracking
    original = tracking.Tracking
    instances = []
    primary_error = None

    class OwnedTracking(original):
        def __init__(self, *args, **kwargs):
            self.logger = {}
            instances.append(self)
            super().__init__(*args, **kwargs)

    tracking.Tracking = OwnedTracking
    try:
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        tracking.Tracking = original
        errors = []
        for instance in instances:
            # Remove before finish, so native __del__ cannot retry with exit=0.
            backend = instance.logger.pop('wandb', None)
            if backend is not None:
                try:
                    backend.finish(exit_code=0 if primary_error is None else 1)
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            message = 'W&B finish failed: ' + '; '.join(repr(exc) for exc in errors)
            if primary_error is not None:
                primary_error.add_note(message)
                try:
                    warnings.warn(message, RuntimeWarning)
                except BaseException:
                    # A warnings-as-errors policy must not replace training's
                    # primary exception during lifecycle cleanup.
                    pass
            else:
                raise RuntimeError(message) from errors[0]


def inspect_checkpoint(root, step, world_size):
    """Validate optimizer updates; model shard loading is tested by resume."""
    root = Path(root)
    files = [root / 'data.pt'] + [
        root / 'actor' / f'{kind}_world_size_{world_size}_rank_{rank}.pt'
        for rank in range(world_size) for kind in ('model', 'optim', 'extra_state')]
    if any(not p.is_file() or p.stat().st_size == 0 for p in files):
        raise ValueError('Missing or empty native checkpoint artifacts')
    counts = []
    for rank in range(world_size):
        state = torch.load(root / 'actor' / f'optim_world_size_{world_size}_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        steps = [int(v['step']) for v in state['state'].values() if 'step' in v]
        if not steps or any(value != step for value in steps):
            raise ValueError(f'Rank {rank} optimizer did not reach step {step}')
        for entry in state['state'].values():
            for value in entry.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError(f'Rank {rank} has non-finite optimizer state')
        counts.append(len(steps))
        del state
    return dict(optimizer_step=step, optimizer_entries_by_rank=counts,
                checkpoint_files=[str(p) for p in files],
                boundary='optimizer parsed; model artifacts present; not a model-resume/parity certificate')


def expected_visible_gpus(config):
    if config.trainer.nnodes != 1:
        raise ValueError('OnQ requires a single-node student allocation')
    expected = config.trainer.n_gpus_per_node
    if config.distillation.enabled:
        if config.distillation.nnodes != 1:
            raise ValueError('OnQ requires a single-node teacher allocation')
        if not config.get('opd_external_teacher_json'):
            expected += config.distillation.n_gpus_per_node
    return expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Resolved native configuration')
    a = p.parse_args()
    import ray
    from omegaconf import OmegaConf, open_dict
    from verl.trainer import main_ppo as native
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from ..opd.teacher_lifecycle import teacher_clients_lifecycle

    class OnQTrainer(RayPPOTrainer):
        def init_workers(self):
            with teacher_clients_lifecycle(self, close_on_success=False):
                return super().init_workers()

        def fit(self):
            from .paper_training import curriculum_metrics
            with (teacher_clients_lifecycle(self),
                  curriculum_metrics(enabled=bool(self.config.get('qat_r8_plan')))):
                return super().fit()

        def _load_checkpoint(self):
            result = super()._load_checkpoint()
            if self.config.get('qat_restore_epoch_boundary_dataloader', False):
                from .epoch_boundary_resume import restore_epoch_boundary_loader
                restore_epoch_boundary_loader(self)
            return result

        def _update_actor(self, batch):
            from .paper_training import update
            return update(self, batch, super()._update_actor)

        def _validate(self, merged=False):
            from .paper_validation import validate
            return validate(self)

    class OnQTaskRunner(native.TaskRunner):
        def init_resource_pool_mgr(self, config):
            manager = super().init_resource_pool_mgr(config)
            if config.get('opd_external_teacher_json'):
                from ..opd.external_teacher import externalize_teacher_pool
                return externalize_teacher_pool(manager)
            return manager

        def run(self, config):
            # Supported custom TaskRunner; the OnQ recipe changes the
            # trainer class binding inside its isolated Ray process.
            previous = native.RayPPOTrainer
            native.RayPPOTrainer = OnQTrainer
            import verl.experimental.teacher_loop as teacher_loop
            previous_teacher = teacher_loop.MultiTeacherModelManager
            try:
                if config.get('opd_external_teacher_json'):
                    from ..opd.external_teacher import ExternalTeacherModelManager
                    teacher_loop.MultiTeacherModelManager = ExternalTeacherModelManager
                if config.get('qat_r8_plan'):
                    from verl.workers.rollout.vllm_rollout import vllm_async_server as server_module
                    from .paper_rollout import PaperRolloutServer
                    previous_server = server_module.vLLMHttpServer
                    server_module.vLLMHttpServer = PaperRolloutServer
                with tracking_lifecycle():
                    return super().run(config)
            finally:
                if 'previous_server' in locals():
                    server_module.vLLMHttpServer = previous_server
                native.RayPPOTrainer = previous
                teacher_loop.MultiTeacherModelManager = previous_teacher

    config = OmegaConf.load(a.config)
    validate_method(config, 'dynamic_kl', require_external=True)
    if not config.get('qat_r8_plan'):
        raise ValueError('Use OnQ opd prepare to create the training configuration')
    if config.get('qat_r8_plan'):
        from .paper_contract import verify_files
        verify_files(config)
    from .resume import validate_resume
    validate_resume(config, 'dynamic_kl')
    if not config.actor_rollout_ref.rollout.calculate_log_probs:
        raise ValueError('OnQ requires rollout log probabilities')
    if config.actor_rollout_ref.model.external_lib != 'OnQ.integrations.verl_v080.utils.engine':
        raise ValueError('The QAT engine adapter must be selected')
    expected_gpus = expected_visible_gpus(config)
    external = bool(config.get('opd_external_teacher_json'))
    if external:
        manager = 'OnQ.integrations.verl_v080.opd.teacher_lifecycle.ExternalTeacherAgentLoopManager'
        agent = config.actor_rollout_ref.rollout.agent
        if agent.get('agent_loop_manager_class') not in (None, manager):
            raise ValueError('External teacher pooling requires its native agent-loop manager extension')
        with open_dict(agent):
            agent.agent_loop_manager_class = manager
        OmegaConf.save(config, Path(config.trainer.default_local_dir) / 'runtime_transport_config.yaml')
        from ..opd.external_teacher import ExternalTeacherModelManager
        teacher = ExternalTeacherModelManager(config, None)
        with (Path(config.trainer.default_local_dir) / 'teacher_endpoint.json').open('x') as handle:
            json.dump(teacher.endpoint_metadata, handle, indent=2)
    if torch.cuda.device_count() != expected_gpus:
        raise ValueError('Visible GPUs do not match the separate student and teacher pools')
    if (config.actor_rollout_ref.actor.ppo_epochs != 1 or
            config.actor_rollout_ref.actor.ppo_mini_batch_size != config.data.train_batch_size):
        raise ValueError('Checkpoint counter check requires one optimizer update per trainer step')
    native.auto_set_device(config)
    config = native.migrate_legacy_reward_impl(config)
    native.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(OnQTaskRunner))
    step = config.trainer.total_training_steps
    root = Path(config.trainer.default_local_dir)
    report = inspect_checkpoint(root / f'global_step_{step}', step, config.trainer.n_gpus_per_node)
    with (root / 'checkpoint_check.json').open('x') as handle:
        json.dump(report, handle, indent=2)


if __name__ == '__main__':
    main()
