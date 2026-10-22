"""Internal QAT overrides for the Pure OPD preparation entrypoint."""
import json
import os
from pathlib import Path

from .version_guard import verify_source

def command(source, model, train_file, val_file, run_dir, bits, python, quant_lr=None):
    source = Path(source).resolve()
    verify_source(source)
    project = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    # Do not inherit the old project's PYTHONPATH containing OnQ/verl.
    env['PYTHONPATH'] = os.pathsep.join((str(source), str(project)))
    # Keep the venv bin directory: resolving its python symlink selects system Python.
    env['PATH'] = str(Path(python).absolute().parent) + os.pathsep + env.get('PATH', '')
    env.update(MODEL_PATH=str(Path(model).resolve()), TRAIN_FILE=str(Path(train_file).resolve()),
               TEST_FILE=str(Path(val_file).resolve()), PROJECT_NAME='qat-native-verl080',
               EXPERIMENT_NAME=Path(run_dir).name)
    adapter = {'bits': bits, 'group_size': 128}
    if quant_lr is not None:
        # Optional separate scale LR; absent keeps the historical adapter bytes.
        adapter['quant_lr'] = quant_lr
    argv = ['bash', str(source/'examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh'),
            'actor_rollout_ref.model.external_lib=OnQ.integrations.verl_v080.utils.engine',
            "+actor_rollout_ref.model.override_config.qat_adapter_json='"+
            json.dumps(adapter, separators=(',',':'))+"'",
            'actor_rollout_ref.actor.fsdp_config.use_orig_params=True',
            'actor_rollout_ref.actor.fsdp_config.model_dtype=fp32',
            'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]',
            '+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:fp32,reduce_dtype:fp32,buffer_dtype:fp32}',
            'actor_rollout_ref.ref.fsdp_config.use_orig_params=True',
            'actor_rollout_ref.ref.fsdp_config.param_offload=False',
            'actor_rollout_ref.ref.fsdp_config.model_dtype=fp32',
            '+actor_rollout_ref.ref.fsdp_config.mixed_precision={param_dtype:fp32,reduce_dtype:fp32,buffer_dtype:fp32}',
            'trainer.default_local_dir='+str(Path(run_dir).resolve())]
    return argv, env
