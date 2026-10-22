# Runtime setup

The release provides isolated installation profiles, an exact-candidate teacher
setup/probe. Follow
[REPRODUCE.md](REPRODUCE.md) for data, calibration, training, resume, export and
the nine evaluation tasks.

## Install

Use Linux x86-64, uv-managed Python 3.11.16, Git,
GCC/G++, and an NVIDIA driver compatible with CUDA 12.8. FlashAttention builds
need a development toolkit: CUDA 12.8 for training/teacher and CUDA 12.1 for
calibration. Set `CUDA_HOME` for each installation. `MAX_JOBS` controls build
parallelism. The scripts do not alter the system Python or existing environments.

```bash
uv python install 3.11.16
export ONQ_PYTHON="$(uv python find --managed-python 3.11.16)"
export ENVS="$HOME/onq-envs"
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=4 "$ONQ_PYTHON" -m OnQ.deployment.install \
  --profile train --root "$ENVS" --python "$ONQ_PYTHON"
CUDA_HOME=/usr/local/cuda-12.1 MAX_JOBS=4 "$ONQ_PYTHON" -m OnQ.deployment.install \
  --profile calibration --root "$ENVS" --python "$ONQ_PYTHON"
"$ONQ_PYTHON" -m OnQ.deployment.install \
  --profile eval --root "$ENVS" --python "$ONQ_PYTHON"
```

Maui uses uv-managed CPython. These commands use that interpreter to create
separate `venv` environments and install the pinned packages with `pip`.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) before running them.

Add `--plan` to any command to inspect it without installing. Each environment
uses its own interpreter for every subprocess. The training profile also clones
official veRL at `7aed6b230776f963fa09509c10d9c3a767d1102c`, applies the bundled
rollout-transfer backport, and verifies the patched source hashes. It editable-installs
both sources without re-resolving their broad dependencies, runs `pip check` and the
import checks, then saves `installed.txt` and `INSTALL_PASS.json` in the venv.
No private dependency access or manual patch application is needed.
FlashAttention is installed after PyTorch using its matching build environment.
Keep both editable source directories for the lifetime of these environments.

The `*.lock` files resolve all dependencies of the `*.in` profiles for Linux
x86-64/Python 3.11. FlashAttention is separately pinned in the installer because
its build imports the installed PyTorch. These are package-version locks, not
wheel-hash locks. Updating an input requires regenerating its lock and rerunning
the checks. Data preparation retains its own environment in [DATA.md](DATA.md).
Code scoring additionally requires Docker and the images in the
[evaluation guide](../evaluation/evalscope/README.md).
