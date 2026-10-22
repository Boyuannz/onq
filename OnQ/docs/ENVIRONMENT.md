# OnQ environment

For new installs, use [the runtime setup](RUNTIME.md) and the resolved profiles
in `OnQ/deployment/*.lock`. The historical inventory below is retained as reference;
the installer profiles have a separate validation record in [RUNTIME.md](RUNTIME.md).

Use **separate environments for online training, evaluation clients, and
Block-AP/Calibration**. The table below records the installed H20 reference stacks
used for the reference implementation. These are observed
versions, not minimum requirements or a tested clean-install lockfile. Matching
package versions alone does not establish matching source, wheels or runtime
behavior. The package metadata permits Python `>=3.10`; these deployments use
**Python 3.11.11**.

### Host and CUDA

| Component | Observed H20 reference |
| --- | --- |
| OS | Ubuntu 22.04, Linux x86-64 |
| GPU | 8 × NVIDIA H20 per host, 97,871 MiB reported memory per GPU |
| NVIDIA driver | 580.95.05 |
| System CUDA compiler (`nvcc`) | 12.1.105 |
| Online PyTorch CUDA runtime | **12.8**, PyTorch `2.9.0+cu128` |
| Calibration CUDA runtime package | `nvidia-cuda-runtime-cu12==12.1.105` |
| Build/archive tools | Git 2.34.1, GCC/G++ 11.4.0, CMake 3.25.2, Ninja, GNU tar, zstd |

**System `nvcc` and the CUDA runtime bundled with PyTorch are different.** Do
not rebuild FlashAttention or other CUDA extensions against an arbitrary
toolkit: match the target PyTorch/CUDA ABI and verify the resulting imports and
kernels. A Python environment archive does not include the NVIDIA driver or
the system CUDA toolkit.

Eight GPUs describe these hosts, not a minimum allocation. Actor, rollout and
teacher topology are run-specific; reserve their actual GPU memory separately.
W2/W3 QAT uses latent floating-point weights and fake quantization, so its
training memory is not the packed INT2/INT3 model size.

The evaluation profile pins the scoring-related versions recorded for the original
main evaluation in `OnQ/deployment/eval.in`, including OpenAI 3.3.1, pandas 3.0.5
and regex 2026.9.10. Code scoring uses the separate Python 3.12.13 containers.

### Python stacks

| Package | Online training / teacher / vLLM | EvalScope client / scorer | Block-AP / Calibration base |
| --- | --- | --- | --- |
| Python | 3.11.11 | 3.11.11 | 3.11.11 |
| torch | 2.9.0+cu128 | Not installed | 2.3.1 |
| torchvision / torchaudio | 0.24.0 / 2.9.0 | Not installed | 0.18.1 / 2.3.1 |
| vllm | **0.12.0** | Not installed | Not installed |
| verl | 0.8.0.dev0 + source pin below | Not installed | Not installed |
| ray | 2.58.0 | Not installed | Not installed |
| transformers | 4.57.6 | 4.57.6 | 4.55.4 |
| tokenizers | 0.22.2 | 0.22.2 | 0.21.4 |
| flash-attn | 2.8.3 | Not installed | 2.7.1.post4 |
| triton | 3.5.0 | Not installed | 2.3.1 |
| numpy | 1.26.4 | 2.2.6 | 1.26.4 |
| datasets | 5.0.1 | 3.6.0 | 3.6.0 |
| accelerate | 1.14.0 | Not installed | 1.12.0 |
| peft | 0.20.0 | Not installed | 0.18.1 |
| safetensors | 0.8.0 | 0.8.0 | 0.7.0 |
| hydra-core / omegaconf | 1.3.6 / 2.3.1 | Not required for client entrypoint | 1.3.2 / 2.3.0 |
| torchdata / tensordict | 0.11.0 / 0.10.0 | Not installed | 0.11.0 / not installed |
| wandb | 0.30.0 | 0.30.0 | Not installed |
| EvalScope | Not installed | **1.5.1** | 1.4.1 (historical, not current evaluation protocol) |
| sympy | 1.13.3 | 1.14.0 | 1.13.3 |
| antlr4-python3-runtime | 4.9.3 | 4.9.3 | 4.9.3 |
| latex2sympy2-extended | 1.10.2 | 1.11.0 | 1.11.0 |
| math-verify | 0.8.0 | Not installed | 0.9.0 |

Additional observed Calibration packages include `bitsandbytes==0.49.1`,
`deepspeed==0.17.4` and `sentencepiece==0.2.1`; online training has
`sentencepiece==0.2.2`. Installed packages are an inventory, not a statement
that every entrypoint needs every package.

Use isolated virtual environments with explicit interpreter paths. Install and
validate compiled dependencies against the target CUDA/PyTorch ABI; an archived
virtual environment alone is not a portable environment specification.

### Source setup and verification

New installations use [official veRL](https://github.com/verl-project/verl) at
**`7aed6b230776f963fa09509c10d9c3a767d1102c`**, with the bundled reviewed backport
applied automatically by the installer. A `verl==0.8.0.dev0` version
string is insufficient. Keep its source checkout separate from this repo;
only the reviewed edits described in the
[integration guide](../integrations/verl_v080/README.md) are admitted.
The union teacher producer currently guards **vLLM 0.12.0**.

With an already provisioned compatible online environment:

```bash
export QAT=/path/to/QAT
export VERL=/path/to/pinned/verl
export TRAIN_PY=/path/to/online-venv/bin/python
export PATH="$(dirname "$TRAIN_PY"):$PATH"
export PYTHONPATH="$VERL:$QAT"

"$TRAIN_PY" -m pip install --no-deps -e "$QAT"
"$TRAIN_PY" -c 'from OnQ.integrations.verl_v080.utils.version_guard import assert_pinned_upstream; print(assert_pinned_upstream())'
"$TRAIN_PY" -c 'import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)'
"$TRAIN_PY" -m pip check
```

Keep the venv interpreter path; resolving its symlink to the system interpreter
can select the wrong environment. Do not add an old vendored `OnQ/verl` to
`PYTHONPATH`. The `[train]` extra in `pyproject.toml` contains broad dependency
ranges, **not** the pinned runtime above. Do not use it to upgrade a working
experiment environment in place. Investigate any `pip check` failure before
admitting a rebuilt environment; imports alone are not GPU acceptance.

For each environment, save `python -m pip freeze --all`, `python -m pip list
--format=json`, its `pyvenv.cfg` when present, Python/PyTorch/CUDA versions,
source commits and local diffs alongside the resolved run config. A package
list still needs the compatible wheel/build artifacts and host stack for
restoration. Preserve native latent model, quantizer, optimizer, scheduler,
RNG and data state separately from inference exports; see
[checkpoint recovery requirements](CHECKPOINTS.md).

### Evaluation, services and storage

- **Evaluation uses a separate pinned client.** Install `evalscope[sandbox]==1.5.1`
  and follow [the evaluation guide](../evaluation/evalscope/README.md). Model
  serving uses a separate OpenAI-compatible endpoint. The eval installer provisions
  NLTK `punkt_tab`; `OnQ.deployment.doctor --profile eval` checks it before use. Code benchmarks use the
  EvalScope Docker sandbox. Native Plus scores are not equivalent to the
  paper's full EvalPlus scoring contract.
- **Math overlays are not full environments.** Historical
  locally prepared benchmark overlays must retain source and item-coverage records.


### Training precision handoff

The release uses calibration → OPD directly, accepting continuous **FP32** latent weight, scale,
and zero state only. Keep native model/optimizer/scheduler/RNG/data checkpoints
for exact resume. There is no source-rounding option; historical configs with a
BF16/FP16 `initial_weight_dtype` are rejected and must resume on their own branch.
Validate saved tensor dtypes before HF loading: casting an already rounded
checkpoint to FP32 does not restore lost precision. Materialized inference
exports are evaluation/rollout artifacts, never training starts. BF16 execution
and exports remain supported separately from FP32 training masters.
