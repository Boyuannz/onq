"""Opt-in native FSDP1 engine extension; no trainer or RL-loss replacement.

Import via model.external_lib. Only models with hf_config.qat_adapter activate
the adapter.
"""
import json
from functools import partial

from .version_guard import assert_pinned_upstream

assert_pinned_upstream()

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.config.optimizer import build_optimizer
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

from .qlinear_adapter import dense_full_state_weights
from .qlinear_adapter import summoned_full_state
from .qlinear_adapter import install_qlinears
from .qlinear_adapter import parameter_groups
from .qlinear_adapter import restore_quantizer_state
from .quantizer_checkpoint import load_quantizer_state
from .reporting import detach_reporting_metadata
from .frozen_bf16 import clear_frozen_weight_caches, mark_frozen_caches_fsdp_managed


class QATFSDPEngine(FSDPEngineWithLMHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_adapter = getattr(self.model_config.hf_config, "qat_adapter", None)
        encoded = getattr(self.model_config.hf_config, 'qat_adapter_json', None)
        if encoded is not None:
            self.qat_adapter = json.loads(encoded)
        if self.qat_adapter is None:
            return
        if self.engine_config.strategy != "fsdp" or not self.engine_config.use_orig_params:
            raise ValueError("QAT adapter requires FSDP1 with use_orig_params=True")
        if self._qat_enabled or self._is_lora:
            raise ValueError("Do not combine EfficientQAT with native QAT or LoRA")
        mp = self.engine_config.mixed_precision or {}
        if mp.get("param_dtype") not in ("fp32", "float32"):
            raise ValueError("QAT FSDP must retain FP32 forward parameters for quantization")
        if self.engine_config.model_dtype not in ("fp32", "float32"):
            raise ValueError("QAT source weights must be loaded as FP32")
        if not isinstance(self.qat_adapter, dict):
            raise ValueError("hf_config.qat_adapter must be a mapping")
        from OnQ.integrations.verl_v080.utils.model_adapter import require_fp32_start
        require_fp32_start(self.qat_adapter)
        if not self.engine_config.forward_only:
            from OnQ.quantization.runtime_compat import require_compatible_quantization_gradients
            require_compatible_quantization_gradients()
            from OnQ.quantization.latent_checkpoint import require_training_save_contents
            require_training_save_contents(self.checkpoint_config.save_contents)

    def _build_module(self):
        if self.qat_adapter is not None:
            from OnQ.quantization.latent_checkpoint import require_fp32_latent_source
            require_fp32_latent_source(self.model_config.local_path,
                bits=int(self.qat_adapter['bits']),
                group_size=int(self.qat_adapter.get('group_size', 128)))
        module = super()._build_module()
        if self.qat_adapter is None:
            return module
        spec = self.qat_adapter
        module = install_qlinears(module, bits=int(spec['bits']),
                                 group_size=int(spec.get('group_size', 128)),
                                 scale_floor=float(spec.get('scale_floor', 1e-4)),
                                 cache_frozen_weights=spec.get('cache_frozen_weights', True))
        if self.rank == 0:
            print('[QAT activations] ' + json.dumps(dict(dtype='bfloat16',
                embedding_norm_modules=len(module._qat_bf16_activation_modules),
                parameter_storage='float32')), flush=True)
            state = load_quantizer_state(self.model_config.local_path)
            metadata = None
            if state and any(not any(k.endswith('.'+tail) for k in state)
                             for tail in ('wbits_buf', 'fixed_codes_buf')):
                from .legacy_source import verified_legacy_metadata
                metadata = verified_legacy_metadata(
                    self.model_config.local_path, bits=int(spec['bits']),
                    group_size=int(spec.get('group_size',128)))
            restored = restore_quantizer_state(module, state, legacy_metadata=metadata)
            print('[QAT restore] ' + json.dumps(restored), flush=True)
        # Native FSDP sync_module_states broadcasts rank-zero quantizer state.
        if self.engine_config.forward_only:
            module.requires_grad_(False)
        return module

    def _build_fsdp_module(self, module):
        wrapped = super()._build_fsdp_module(module)
        if self.qat_adapter is not None:
            # Keep FSDP parameters FP32 while Transformer operations use BF16.
            self._autocast_dtype = torch.bfloat16
            mark_frozen_caches_fsdp_managed(wrapped)
        return wrapped

    def load_checkpoint(self, local_path, *args, **kwargs):
        try:
            if self.qat_adapter is not None and local_path is not None:
                from .native_precision import require_native_fp32_resume
                require_native_fp32_resume(
                    local_path, self.qat_adapter,
                    rank=self.checkpoint_manager.rank,
                    world_size=self.checkpoint_manager.world_size)
            return super().load_checkpoint(local_path, *args, **kwargs)
        finally:
            if self.qat_adapter is not None:
                clear_frozen_weight_caches(self.module)

    def _build_optimizer(self, module):
        if self.qat_adapter is None:
            return super()._build_optimizer(module)
        from OnQ.integrations.verl_v080.utils.model_adapter import adapter_quant_lr
        # Scales use the optional adapter quant_lr; absent keeps the shared LR.
        groups = parameter_groups(module, lr=float(self.optimizer_config.lr),
                                  weight_decay=float(self.optimizer_config.weight_decay),
                                  quant_lr=adapter_quant_lr(self.qat_adapter))
        return build_optimizer(groups, self.optimizer_config)

    def forward_step(self, micro_batch, loss_function, forward_only):
        is_native_gkd = False
        dynamic = self.qat_adapter.get('dynamic_kl') if self.qat_adapter is not None else None
        dynamic_selected = False
        if self.qat_adapter is not None and isinstance(loss_function, partial):
            from verl.trainer.distillation.losses import distillation_ppo_loss

            if loss_function.func is distillation_ppo_loss:
                # v0.8.0 computes OPD before ppo_loss populates these fields.
                # FSDP has already all-reduced the current batch token count.
                config = loss_function.keywords['config']
                for key in ('dp_size', 'batch_num_tokens', 'global_batch_size'):
                    config.global_batch_info[key] = micro_batch[key]
                config.global_batch_info['loss_scale_factor'] = config.loss_scale_factor
                distillation_config = loss_function.keywords['distillation_config']
                is_native_gkd = distillation_config.distillation_loss.loss_mode == 'forward_kl_topk'
                if dynamic is not None:
                    from .dynamic_kl_native import dynamic_kl_loss
                    if not is_native_gkd:
                        raise ValueError('Dynamic KL requires the native top-k teacher transport')
                    loss_function = partial(dynamic_kl_loss, config=config,
                        distillation_config=distillation_config, options=dynamic,
                        vocab_size=int(self.model_config.hf_config.vocab_size),
                        union=self.qat_adapter.get('union'))
                    dynamic_selected = True
        if dynamic is not None and not dynamic_selected and not (forward_only and loss_function is None):
            raise ValueError('Refusing to bypass the configured dynamic KL loss')
        loss, output = super().forward_step(micro_batch, loss_function, forward_only)
        if is_native_gkd and not forward_only:
            output['model_output'] = detach_reporting_metadata(output['model_output'])
        return loss, output


    def train_batch(self, data, loss_function):
        result = super().train_batch(data, loss_function)
        if 'quant_lr' in (self.qat_adapter or {}) and self.is_mp_src_rank_with_outputs():
            from verl.utils.metric import Metric, AggregationType
            # Quantizer-group LR applied by this update (native "lr" reports group 0).
            result['metrics']['lr_quant'] = Metric(AggregationType.MEAN,
                                                   float(self.optimizer.param_groups[1]['lr']))
        return result

    def lr_scheduler_step(self):
        applied_lr = float(self.optimizer.param_groups[0]['lr'])
        next_lr = super().lr_scheduler_step()
        # Native workers log this return value after train_batch. For split LR,
        # match lr_quant's applied-update timing while advancing the scheduler once.
        return applied_lr if 'quant_lr' in (self.qat_adapter or {}) else next_lr

    def get_per_tensor_param(self, *args, **kwargs):
        if self.qat_adapter is None:
            return super().get_per_tensor_param(*args, **kwargs)

        def materialized():
            load_fsdp_model_to_gpu(self.module)
            try:
                # Consume uncloned views of the summoned full parameters. A
                # FULL_STATE_DICT state_dict() here would clone every full FP32
                # tensor on the device (a second full model per rank); names,
                # order, dtypes and exported values are unchanged.
                with FSDP.summon_full_params(self.module, writeback=False, rank0_only=False):
                    weights = dense_full_state_weights(self.module, summoned_full_state(self.module))
                    yield from weights
            finally:
                if self._is_offload_param:
                    clear_frozen_weight_caches(self.module)
                    offload_fsdp_model_to_cpu(self.module)

        return materialized(), None


def register():
    registry = EngineRegistry._engines['language_model']['fsdp']
    existing = registry.get('cuda')
    if existing is QATFSDPEngine:
        return
    if existing is not FSDPEngineWithLMHead:
        raise RuntimeError('Refusing to replace an unknown FSDP engine plugin')
    # v0.8.0 register() does not overwrite entries. Explicit opt-in replacement
    # keeps unconfigured dense models delegated to the original engine.
    registry['cuda'] = QATFSDPEngine


register()
