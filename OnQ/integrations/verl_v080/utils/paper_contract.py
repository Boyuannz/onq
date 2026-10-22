"""Separate opt-in admission for five-domain paper OPD and held-out validation."""
from ..opd.schedule import V4_SCHEDULE, response_cap


MODEL_STEPS = {"1.7b": 400, "4b": 300, "8b": 300}

def rollout_cap(weight_step, temperature):
    if type(weight_step) is not int or not 0 <= weight_step <= 400:
        raise ValueError('Missing synchronized rollout step')
    if temperature == 0:
        return 16384
    if temperature != 1:
        raise ValueError('paper recipe supports T=1 training and greedy validation only')
    return response_cap(weight_step + 1, schedule=V4_SCHEDULE)


def validate_paper(config, adapter, mode):
    plan = config.get('qat_r8_plan')
    spec = adapter.get('r8_schedule')
    if not plan or spec != dict(version=1, schedule=V4_SCHEDULE, validation_cap=16384):
        raise ValueError('paper data and worker schedule must opt in together')
    rollout = config.actor_rollout_ref.rollout
    if (mode != 'dynamic_kl' or not adapter.get('union') or config.get('qat_mixed_plan')
            or adapter.get('mixed_observation') or plan.version != 1
            or config.data.train_batch_size != 128 or config.data.shuffle
            or config.data.max_prompt_length != 4096 or config.data.max_response_length != 16384
            or rollout.response_length != 16384 or rollout.prompt_length != 4096
            or config.data.val_batch_size != 64 or config.data.filter_overlong_prompts
            or config.data.truncation != 'error' or plan.get("model_size") not in MODEL_STEPS
            or config.trainer.total_training_steps != MODEL_STEPS.get(plan.get("model_size"))
            or config.trainer.total_epochs != 1 or config.trainer.test_freq != 50
            or config.trainer.save_freq != 50 or config.trainer.val_before_train
            or rollout.val_kwargs.temperature != 0 or rollout.val_kwargs.do_sample
            or rollout.val_kwargs.n != 1 or rollout.val_kwargs.top_p != 1
            or rollout.val_kwargs.top_k != -1
            or config.trainer.n_gpus_per_node not in (1, 4, 8)
            or config.trainer.nnodes != 1):
        raise ValueError('paper training/validation contract changed')
    if len(plan.validation_ids) != 256 or len(set(plan.validation_ids)) != 256:
        raise ValueError('Expected exactly 256 held-out prompt identities')


def verify_files(config):
    from ..opd.prompt_data import sha
    plan = config.qat_r8_plan
    for paths, digest in ((config.data.train_files, plan.train_sha256),
                          (config.data.val_files, plan.validation_sha256)):
        if len(paths) != 1 or sha(paths[0]) != digest:
            raise ValueError('paper data file differs from the accepted manifest')
