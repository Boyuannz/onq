# Directory and file guide


The tables include every tracked file. Library modules are imported by the entry points;
they do not each need to be launched separately. `agent_loop` refers to veRL rollout
workers used by training. Only the four workflows and their runtime dependencies are included.

<details>
<summary>. — Repository root</summary>

| File | Purpose |
| --- | --- |
| [.gitignore](../../.gitignore) | Exclude generated outputs, caches and local environments from version control. |
| [README.md](../../README.md) | Project overview, installation, models and paper citation. |
| [pyproject.toml](../../pyproject.toml) | Package metadata, dependencies, included assets and onq CLI registration. |

</details>

<details>
<summary>OnQ — Package entry points and shared helpers</summary>

| File | Purpose |
| --- | --- |
| [UPSTREAMS.md](../../OnQ/UPSTREAMS.md) | Upstream source revisions, attribution and license locations. |
| [__main__.py](../../OnQ/__main__.py) | Dispatcher for preparedata, calibration, opd and evaluation. |
| [__init__.py](../../OnQ/__init__.py) | Python package initialization and namespace declaration. |
| [paths.py](../../OnQ/paths.py) | Locate the installed source and repository roots. |
| [runtime.py](../../OnQ/runtime.py) | Represent backend commands and build their process environments. |
| [utils.py](../../OnQ/utils.py) | Project-owned file hashing. |

</details>

<details>
<summary>OnQ/calibration — Stage 1 orchestration</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/calibration/__init__.py) | Python package initialization and namespace declaration. |
| [audit.py](../../OnQ/calibration/audit.py) | Run tensor-level checks on the Stage 1 checkpoint. |
| [block_ap.py](../../OnQ/calibration/block_ap.py) | Build the Block-AP command and run-local compatibility overlay. |
| [data.py](../../OnQ/calibration/data.py) | Load and tokenize calibration conversations for the selected recipe. |
| [parallel.py](../../OnQ/calibration/parallel.py) | Preserve the original two-GPU Block-AP forward and global batch-2 loss. |
| [recipe_router.py](../../OnQ/calibration/recipe_router.py) | Map named calibration recipes to their backend settings. |
| [run.py](../../OnQ/calibration/run.py) | Orchestrate Stage 1, save outputs and write the handoff manifest. |
| [token_cache.py](../../OnQ/calibration/token_cache.py) | Consume preparation-script token windows without retokenizing or resampling. |

</details>

<details>
<summary>OnQ/calibration/upstream — Attributed upstream Block-AP source</summary>

| File | Purpose |
| --- | --- |
| [PAYLOAD_SHA256.json](../../OnQ/calibration/upstream/PAYLOAD_SHA256.json) | Original upstream and locally patched source hashes. |
| [UPSTREAM_PROVENANCE.md](../../OnQ/calibration/upstream/UPSTREAM_PROVENANCE.md) | Origin, retained scope and local patch notes for Block-AP. |
| [datautils_block.py](../../OnQ/calibration/upstream/datautils_block.py) | Upstream dataset loaders, block datasets and perplexity evaluation. |
| [main_block_qat.py](../../OnQ/calibration/upstream/main_block_qat.py) | Upstream Block-AP entry point with latent-checkpoint saving. |
| [requirements.txt](../../OnQ/calibration/upstream/requirements.txt) | Original upstream dependency list; use the resolved installation profile for this release. |
| [utils.py](../../OnQ/calibration/upstream/utils.py) | Upstream gradient scaling, norm calculation and logging. |

</details>

<details>
<summary>OnQ/calibration/upstream/quantize — Upstream quantization layers</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/calibration/upstream/quantize/__init__.py) | Python package initialization and namespace declaration. |
| [block_qat.py](../../OnQ/calibration/upstream/quantize/block_qat.py) | Optimize quantized blocks against full-precision block outputs. |
| [int_linear_fake.py](../../OnQ/calibration/upstream/quantize/int_linear_fake.py) | Upstream fake-quantized linear layers and model loading. |
| [int_linear_real.py](../../OnQ/calibration/upstream/quantize/int_linear_real.py) | Upstream packed integer linear layers and model loading. |
| [quantizer.py](../../OnQ/calibration/upstream/quantize/quantizer.py) | Upstream quantizers and straight-through gradient operations. |
| [utils.py](../../OnQ/calibration/upstream/quantize/utils.py) | Select trainable parameters and apply quantization to blocks. |

</details>

<details>
<summary>OnQ/calibration/upstream/quantize/triton_utils — Upstream GPU kernels</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/calibration/upstream/quantize/triton_utils/__init__.py) | Python package initialization and namespace declaration. |
| [custom_autotune.py](../../OnQ/calibration/upstream/quantize/triton_utils/custom_autotune.py) | Autotune the upstream Triton kernel configurations. |
| [kernels.py](../../OnQ/calibration/upstream/quantize/triton_utils/kernels.py) | Triton dequantization kernels used by packed linear layers. |

</details>

<details>
<summary>OnQ/cli — Calibration CLI</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/cli/__init__.py) | Python package initialization and namespace declaration. |
| [opd.py](../../OnQ/cli/opd.py) | Resolve the native recipe and prepare the complete OnQ configuration in one command. |
| [train.py](../../OnQ/cli/train.py) | Parse the calibration command and dispatch Stage 1. |

</details>

<details>
<summary>OnQ/config — Configuration parsing</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/config/__init__.py) | Python package initialization and namespace declaration. |
| [loader.py](../../OnQ/config/loader.py) | Merge CLI, profiles and overrides; validate the resolved configuration. |
| [resolved.py](../../OnQ/config/resolved.py) | Serialize resolved settings and configuration hashes. |
| [schema.py](../../OnQ/config/schema.py) | Typed training configuration and configuration errors. |

</details>

<details>
<summary>OnQ/configs/calibration — Configuration presets</summary>

| File | Purpose |
| --- | --- |
| [filtered_reasoning.yaml](../../OnQ/configs/calibration/filtered_reasoning.yaml) | filtered reasoning calibration data and optimization recipe. |
| `math_ot_mix` | CLI recipe retained; numerical defaults shared with `official.yaml`, data routing remains distinct. |
| [math_train.yaml](../../OnQ/configs/calibration/math_train.yaml) | math train calibration data and optimization recipe. |
| [official.yaml](../../OnQ/configs/calibration/official.yaml) | official calibration data and optimization recipe. |
| [paper_tokens.yaml](../../OnQ/configs/calibration/paper_tokens.yaml) | paper tokens calibration data and optimization recipe. |

</details>

<details>
<summary>OnQ/configs/experiments/verified — Configuration presets</summary>

| File | Purpose |
| --- | --- |
| [q4b_w2_filtered_seed42.yaml](../../OnQ/configs/experiments/verified/q4b_w2_filtered_seed42.yaml) | q4b w2 filtered seed42 calibration preset; not a saved experiment result. |
| [q4b_w2_official_seed42.yaml](../../OnQ/configs/experiments/verified/q4b_w2_official_seed42.yaml) | q4b w2 official seed42 calibration preset; not a saved experiment result. |
| [q4b_w3_filtered_seed42.yaml](../../OnQ/configs/experiments/verified/q4b_w3_filtered_seed42.yaml) | q4b w3 filtered seed42 calibration preset; not a saved experiment result. |
| [q4b_w3_official_seed42.yaml](../../OnQ/configs/experiments/verified/q4b_w3_official_seed42.yaml) | q4b w3 official seed42 calibration preset; not a saved experiment result. |

</details>

<details>
<summary>OnQ/configs/model — Configuration presets</summary>

| File | Purpose |
| --- | --- |
| [qwen3_1p7b.yaml](../../OnQ/configs/model/qwen3_1p7b.yaml) | qwen3 1p7b model defaults. |
| [qwen3_4b.yaml](../../OnQ/configs/model/qwen3_4b.yaml) | qwen3 4b model defaults. |
| [qwen3_8b.yaml](../../OnQ/configs/model/qwen3_8b.yaml) | qwen3 8b model defaults. |

</details>

<details>
<summary>OnQ/configs/quant — Configuration presets</summary>

| File | Purpose |
| --- | --- |
| [w2g128.yaml](../../OnQ/configs/quant/w2g128.yaml) | W2G128 weight/group quantization settings. |
| [w3g128.yaml](../../OnQ/configs/quant/w3g128.yaml) | W3G128 weight/group quantization settings. |
| [w4g128.yaml](../../OnQ/configs/quant/w4g128.yaml) | W4G128 weight/group quantization settings. |

</details>

<details>
<summary>OnQ/configs/runtime — Configuration presets</summary>

| File | Purpose |
| --- | --- |
| [gpu.yaml](../../OnQ/configs/runtime/gpu.yaml) | gpu execution profile. |
| `h100` | CLI profile retained as an alias of `gpu.yaml`. |
| [local.yaml](../../OnQ/configs/runtime/local.yaml) | local execution profile. |

</details>

<details>
<summary>OnQ/configs/upstreams — Configuration presets</summary>

| File | Purpose |
| --- | --- |
| [verl_v080.json](../../OnQ/configs/upstreams/verl_v080.json) | Pinned veRL revision and backport requirements. |

</details>

<details>
<summary>OnQ/deployment — Installation and teacher service setup</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/deployment/__init__.py) | Python package initialization and namespace declaration. |
| [calibration.in](../../OnQ/deployment/calibration.in) | Direct dependency requirements for the calibration environment. |
| [calibration.lock](../../OnQ/deployment/calibration.lock) | Resolved Linux/Python 3.11 package versions for the calibration environment. |
| [doctor.py](../../OnQ/deployment/doctor.py) | Check the selected environment dependencies, imports and versions. |
| [eval.in](../../OnQ/deployment/eval.in) | Direct dependency requirements for the eval environment. |
| [eval.lock](../../OnQ/deployment/eval.lock) | Resolved Linux/Python 3.11 package versions for the eval environment. |
| [install.py](../../OnQ/deployment/install.py) | Create a new isolated Linux environment from a resolved profile. |
| [teacher.py](../../OnQ/deployment/teacher.py) | Prepare the exact-candidate teacher and probe its real scoring contract. |
| [train.in](../../OnQ/deployment/train.in) | Direct dependency requirements for the train environment. |
| [train.lock](../../OnQ/deployment/train.lock) | Resolved Linux/Python 3.11 package versions for the train environment. |

</details>

<details>
<summary>OnQ/docs — Detailed reference documentation</summary>

| File | Purpose |
| --- | --- |
| [FILES.md](../../OnQ/docs/FILES.md) | This file-by-file source map. |
| [CHECKPOINTS.md](../../OnQ/docs/CHECKPOINTS.md) | Latent, native-resume and inference checkpoint formats. |
| [DATA.md](../../OnQ/docs/DATA.md) | Data preparation and prompt conversion reference. |
| [ENVIRONMENT.md](../../OnQ/docs/ENVIRONMENT.md) | Dependency and runtime compatibility reference. |
| [METHOD.md](../../OnQ/docs/METHOD.md) | Objective, training schedule and validation implementation. |
| [REPRODUCE.md](../../OnQ/docs/REPRODUCE.md) | Pointer to the canonical reproduction commands in the root README. |
| [RUNTIME.md](../../OnQ/docs/RUNTIME.md) | Installation profiles and CUDA prerequisites. |

</details>

<details>
<summary>OnQ/evaluation/evalscope — Benchmark clients and isolated code scorers</summary>

| File | Purpose |
| --- | --- |
| [Dockerfile.lcb](../../OnQ/evaluation/evalscope/Dockerfile.lcb) | Build the isolated native LiveCodeBench scorer image. |
| [Dockerfile.plus](../../OnQ/evaluation/evalscope/Dockerfile.plus) | Build the isolated EvalPlus 0.3.1 scorer image. |
| [README.md](../../OnQ/evaluation/evalscope/README.md) | Detailed evaluation protocol and sandbox commands. |
| [__init__.py](../../OnQ/evaluation/evalscope/__init__.py) | Python package initialization and namespace declaration. |
| [lcb.py](../../OnQ/evaluation/evalscope/lcb.py) | Keep EvalScope's native LiveCodeBench scoring inside a no-network container. |
| [lcb_judge.py](../../OnQ/evaluation/evalscope/lcb_judge.py) | Load the pinned native scorer in the isolated container without model imports. |
| [mbpp_extract.py](../../OnQ/evaluation/evalscope/mbpp_extract.py) | Adopted entry-point extraction revision; no tests are consulted. |
| [plus.py](../../OnQ/evaluation/evalscope/plus.py) | EvalScope prompts/extraction with the full EvalPlus 0.3.1 test suite. |
| [plus_judge.py](../../OnQ/evaluation/evalscope/plus_judge.py) | Container-only full EvalPlus scorer; generated code must not run on the host. |
| [prepare_aime90.py](../../OnQ/evaluation/evalscope/prepare_aime90.py) | Prepare a local AIME 2024–2026 adapter input; publishes no benchmark rows. |
| [prepare_native.py](../../OnQ/evaluation/evalscope/prepare_native.py) | Download five pinned native benchmarks and record their versions and file hashes. |
| [prepare_plus.py](../../OnQ/evaluation/evalscope/prepare_plus.py) | Download full EvalPlus problems locally; no datasets are shipped in the release. |
| [run.py](../../OnQ/evaluation/evalscope/run.py) | Run EvalScope 1.5.1 against an OpenAI-compatible model endpoint. |

</details>

<details>
<summary>OnQ/integrations — External backend adapters</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/integrations/__init__.py) | Python package initialization and namespace declaration. |

</details>

<details>
<summary>OnQ/integrations/verl_v080 — Pinned veRL integration</summary>

| File | Purpose |
| --- | --- |
| [README.md](../../OnQ/integrations/verl_v080/README.md) | Integration architecture, source pin and supported modes. |
| [__init__.py](../../OnQ/integrations/verl_v080/__init__.py) | Python package initialization and namespace declaration. |

</details>

<details>
<summary>OnQ/integrations/verl_v080/backports — Pinned upstream patch</summary>

| File | Purpose |
| --- | --- |
| [manifest.json](../../OnQ/integrations/verl_v080/backports/manifest.json) | Expected hashes and revision for the bundled veRL backport. |
| [verl_7873.patch](../../OnQ/integrations/verl_v080/backports/verl_7873.patch) | Required rollout-transfer patch applied to official veRL. |

</details>

<details>
<summary>OnQ/integrations/verl_v080/opd — Stage 2 preparation and teacher services</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/integrations/verl_v080/opd/__init__.py) | Python package initialization and namespace declaration. |
| [external_teacher.py](../../OnQ/integrations/verl_v080/opd/external_teacher.py) | Transport adapter for native k1 OPD and a separately served dense vLLM. |
| [no_reward.py](../../OnQ/integrations/verl_v080/opd/no_reward.py) | Required native reward transport slot; no scoring, tests or reward learning. |
| [schedule.py](../../OnQ/integrations/verl_v080/opd/schedule.py) | The 4K / 8K / 16K response-token curriculum. |
| [prepare.py](../../OnQ/integrations/verl_v080/opd/prepare.py) | Internal native recipe resolver used by the OnQ configuration preparer. |
| [prepare_dynamic_resume.py](../../OnQ/integrations/verl_v080/opd/prepare_dynamic_resume.py) | Prepare, never launch, recovery from the original dynamic run configuration. |
| [prepare_paper.py](../../OnQ/integrations/verl_v080/opd/prepare_paper.py) | Apply OnQ loss, curriculum, data and validation settings to the native configuration. |
| [prompt_data.py](../../OnQ/integrations/verl_v080/opd/prompt_data.py) | Freeze two random passes over an already accepted prompt pool; never reselect it. |
| [teacher_lifecycle.py](../../OnQ/integrations/verl_v080/opd/teacher_lifecycle.py) | Worker-owned HTTP pool cleanup through native agent-loop extensions. |
| [union_teacher_vllm.py](../../OnQ/integrations/verl_v080/opd/union_teacher_vllm.py) | Same-version vLLM teacher with exact union retrieval on prompt positions. |
| [union_vllm_worker.py](../../OnQ/integrations/verl_v080/opd/union_vllm_worker.py) | Run-scoped vLLM prompt-score side channel; ordinary logits never changed. |

</details>

<details>
<summary>OnQ/integrations/verl_v080/utils — Stage 2 loss, execution, checkpoint and diagnostic components</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/integrations/verl_v080/utils/__init__.py) | Python package initialization and namespace declaration. |
| [backports.py](../../OnQ/integrations/verl_v080/utils/backports.py) | Apply or verify the exact reviewed backport on an isolated pinned checkout. |
| [dynamic_kl.py](../../OnQ/integrations/verl_v080/utils/dynamic_kl.py) | Numerical option validation for Dynamic KL. |
| [dynamic_kl_native.py](../../OnQ/integrations/verl_v080/utils/dynamic_kl_native.py) | Opt-in dynamic-KL callbacks for the pinned veRL FSDP eager-logits path. |
| [engine.py](../../OnQ/integrations/verl_v080/utils/engine.py) | Opt-in native FSDP1 engine extension; no trainer or RL-loss replacement. |
| [epoch_boundary_resume.py](../../OnQ/integrations/verl_v080/utils/epoch_boundary_resume.py) | Preserve the private sampler generator across an exhausted-epoch resume. |
| [frozen_bf16.py](../../OnQ/integrations/verl_v080/utils/frozen_bf16.py) | Reusable BF16 execution copies; FP32 frozen parameters remain authoritative. |
| [legacy_source.py](../../OnQ/integrations/verl_v080/utils/legacy_source.py) | Validate old bufferless QAT formats without relabeling their weight provenance. |
| [method_config.py](../../OnQ/integrations/verl_v080/utils/method_config.py) | Fail closed on retired methods and unsupported native loss configurations. |
| [model_adapter.py](../../OnQ/integrations/verl_v080/utils/model_adapter.py) | Resolve QAT identity from the HF source and native model-config overrides. |
| [native_precision.py](../../OnQ/integrations/verl_v080/utils/native_precision.py) | Inspect saved native shards before FSDP can cast them into FP32 parameters. |
| [paper_contract.py](../../OnQ/integrations/verl_v080/utils/paper_contract.py) | Separate opt-in admission for five-domain paper OPD and held-out validation. |
| [paper_rollout.py](../../OnQ/integrations/verl_v080/utils/paper_rollout.py) | Curriculum for training; fixed 16K greedy held-out generation. |
| [paper_training.py](../../OnQ/integrations/verl_v080/utils/paper_training.py) | Check frozen random order without forcing legacy three-domain quotas. |
| [paper_validation.py](../../OnQ/integrations/verl_v080/utils/paper_validation.py) | Held-out on-policy Union objective: native infer_batch, never update_actor. |
| [prepare.py](../../OnQ/integrations/verl_v080/utils/prepare.py) | Internal QAT overrides for the Pure OPD preparation entrypoint. |
| [process_groups.py](../../OnQ/integrations/verl_v080/utils/process_groups.py) | Cleanup of owned new-session process groups, including live descendants. |
| [qlinear_adapter.py](../../OnQ/integrations/verl_v080/utils/qlinear_adapter.py) | QAT-only operations; no GRPO, reward, sampler or old veRL imports. |
| [quantizer_checkpoint.py](../../OnQ/integrations/verl_v080/utils/quantizer_checkpoint.py) | Read saved quantizers without assuming HF shard filenames start with model. |
| [reporting.py](../../OnQ/integrations/verl_v080/utils/reporting.py) | Shared reporting-only graph cleanup; loss tensors are not passed here. |
| [resume.py](../../OnQ/integrations/verl_v080/utils/resume.py) | Validate native QAT resume identity before allocating workers or loading shards. |
| [trainer.py](../../OnQ/integrations/verl_v080/utils/trainer.py) | OnQ training, validation, teacher lifecycle and final checkpoint checks. |
| [supervise.py](../../OnQ/integrations/verl_v080/utils/supervise.py) | Launch training from a resolved configuration and preserve logs on exit. |
| [union_batching.py](../../OnQ/integrations/verl_v080/utils/union_batching.py) | Bounded coalescing for frozen-teacher scoring; no CUDA/HTTP dependencies. |
| [union_kl.py](../../OnQ/integrations/verl_v080/utils/union_kl.py) | Exact union support on the current actor; detached teacher and alpha only. |
| [union_protocol.py](../../OnQ/integrations/verl_v080/utils/union_protocol.py) | Capacity of the reviewed single-teacher producer, independent of CUDA. |
| [union_teacher_scores.py](../../OnQ/integrations/verl_v080/utils/union_teacher_scores.py) | Exact union probability capture; kept independent of CUDA/vLLM imports. |
| [union_transport.py](../../OnQ/integrations/verl_v080/utils/union_transport.py) | Union targets are requested during actor forward, never during rollout. |
| [version_guard.py](../../OnQ/integrations/verl_v080/utils/version_guard.py) | Reject legacy namespace shadowing or an unpinned native veRL checkout. |

</details>

<details>
<summary>OnQ/licenses/verl — Third-party license notices</summary>

| File | Purpose |
| --- | --- |
| [LICENSE](../../OnQ/licenses/verl/LICENSE) | Original veRL license. |
| [Notice.txt](../../OnQ/licenses/verl/Notice.txt) | Original veRL attribution notice. |

</details>

<details>
<summary>OnQ/quantization — Shared quantizers and checkpoint export</summary>

| File | Purpose |
| --- | --- |
| [__init__.py](../../OnQ/quantization/__init__.py) | Python package initialization and namespace declaration. |
| [bit_router.py](../../OnQ/quantization/bit_router.py) | Validate weight bit width and group size. |
| [checkpoint.py](../../OnQ/quantization/checkpoint.py) | Write and validate calibration-to-training checkpoint manifests. |
| [checkpoint_checks.py](../../OnQ/quantization/checkpoint_checks.py) | Internal checks for the calibration-to-OPD checkpoint handoff. |
| [efficientqat_helpers.py](../../OnQ/quantization/efficientqat_helpers.py) | Quantized actor parameter groups and weight materialization for on-policy distillation. |
| [export.py](../../OnQ/quantization/export.py) | Export a trusted FSDP fake-quant checkpoint using its recorded contract. |
| [export_contract.py](../../OnQ/quantization/export_contract.py) | Identity and numerical contract shared by training launches and exports. |
| [latent_checkpoint.py](../../OnQ/quantization/latent_checkpoint.py) | Lossless model-state handoff, separate from materialized inference exports. |
| [qlinear.py](../../OnQ/quantization/qlinear.py) | Replace model linear layers with trainable fake-quantized layers. |
| [quantizer.py](../../OnQ/quantization/quantizer.py) | W2/W3/W4 quantizers and straight-through weight gradients. |
| [runtime_compat.py](../../OnQ/quantization/runtime_compat.py) | Cheap startup check for the existing EfficientQAT clipping derivative. |

</details>

<details>
<summary>OnQ/scripts/data — Dataset preparation</summary>

| File | Purpose |
| --- | --- |
| [README.md](../../OnQ/scripts/data/README.md) | Dataset sources, quotas, output schemas and screening rules. |
| [check_data_overlap.py](../../OnQ/scripts/data/check_data_overlap.py) | Audit written training files for exact/contained benchmark overlap and duplicate records. |
| [dataset_sources.py](../../OnQ/scripts/data/dataset_sources.py) | Provide pinned dataset discovery, cached row access and seeded sampling; imported by preparation scripts. |
| [format_conversations.py](../../OnQ/scripts/data/format_conversations.py) | Format original dataset records into prompts and calibration conversations without rewriting content. |
| [prepare_benchmark_texts.py](../../OnQ/scripts/data/prepare_benchmark_texts.py) | Download pinned public benchmark text and write the held-out query index used for decontamination. |
| [prepare_training_data.py](../../OnQ/scripts/data/prepare_training_data.py) | Sample source records with a chosen seed and write calibration token windows and OPD prompt pools. |
| [requirements.txt](../../OnQ/scripts/data/requirements.txt) | Dependencies for the isolated data-preparation environment. |
| [sampling_recipe.json](../../OnQ/scripts/data/sampling_recipe.json) | Dataset revisions, per-source quotas and sampling settings. |
| [selection_rules.py](../../OnQ/scripts/data/selection_rules.py) | Apply whole-record eligibility rules for calibration and OPD data; imported by the data builder. |
| [text_quality.py](../../OnQ/scripts/data/text_quality.py) | Provide text normalization, near-duplicate indexing and prompt quality checks; shared helper module. |

</details>

## Provenance and evaluation checks

| File | Purpose |
| --- | --- |
| [resources.py](../evaluation/evalscope/resources.py) | Check or explicitly provision IFEval tokenizer resources. |
| [ifeval.py](../evaluation/evalscope/ifeval.py) | Reject scorer errors and incomplete or mismatched 541-item output. |
| [provenance/README.md](../provenance/README.md) | Explain recovered paper evidence and remaining gaps. |
| [paper_records.json](../provenance/paper_records.json) | Anonymous model, data and configuration provenance inventory. |
