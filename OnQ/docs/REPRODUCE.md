# Reproduction

Run the commands in Bash from the repository root. Replace `/absolute/...` paths with your own.

## Environment

We use uv-managed Python with separate virtual environments for training,
calibration and evaluation. The installer creates each environment with `venv`
and installs its pinned packages with `pip`.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then:

```bash
git clone https://github.com/Boyuannz/onq.git
cd onq
export QAT="$PWD"

uv python install 3.11.16
export ONQ_PYTHON="$(uv python find --managed-python 3.11.16)"
export ENVS="$HOME/onq-envs"

CUDA_HOME=/usr/local/cuda-12.8 "$ONQ_PYTHON" -m OnQ.deployment.install \
  --profile train --root "$ENVS" --python "$ONQ_PYTHON"
CUDA_HOME=/usr/local/cuda-12.1 "$ONQ_PYTHON" -m OnQ.deployment.install \
  --profile calibration --root "$ENVS" --python "$ONQ_PYTHON"
"$ONQ_PYTHON" -m OnQ.deployment.install \
  --profile eval --root "$ENVS" --python "$ONQ_PYTHON"
export VERL="$ENVS/verl"
export TRAIN_PY="$ENVS/train/bin/python"
export CALIBRATION_PY="$ENVS/calibration/bin/python"
export EVAL_PY="$ENVS/eval/bin/python"
export PYTHONPATH="$VERL:$QAT"
export PATH="$(dirname "$TRAIN_PY"):$PATH"
```

Each command creates a new environment, installs the Linux/Python 3.11 resolved
package list, checks dependencies/imports and records the installed inventory.
The training installer checks out official veRL at the pinned commit and applies
the bundled, hash-verified rollout-transfer backport. No private dependency access
is required.
Use `--plan` to inspect commands without installing. See
[the runtime setup guide](../../OnQ/docs/RUNTIME.md) for installation and CUDA prerequisites.
The teacher uses the training environment. Scorer containers remain separate.

The veRL backport is documented in the
[integration guide](../../OnQ/integrations/verl_v080/README.md#rollout-transfer-and-teacher-connections).

## 1. preparedata

Create a separate data environment, obtain access to the public source datasets,
and build a seeded prompt pool and calibration windows. First prepare the same
AIME 2024–2026 questions that you will evaluate: 30 questions per year, with
`id`, `year`, `problem`, and `answer` fields. Run from the repository root:

```bash
"$ONQ_PYTHON" -m venv "$ENVS/data"
export DATA_PY="$ENVS/data/bin/python"
"$DATA_PY" -m pip install -r OnQ/scripts/data/requirements.txt
export ONQ_DATA_SEED=42
export DATA=/absolute/new/data/sample_${ONQ_DATA_SEED}
export BENCH=/absolute/new/benchmark_queries.jsonl
export CACHE=/absolute/data_download_cache
export AIME_DATA=/absolute/new/onq_aime90
"$DATA_PY" -m OnQ evaluation aime \
  --input /absolute/aime2024.jsonl /absolute/aime2025.jsonl /absolute/aime2026.jsonl \
  --output "$AIME_DATA"
"$DATA_PY" -m OnQ preparedata benchmarks --aime90 "$AIME_DATA" --out "$BENCH"
"$DATA_PY" -m OnQ preparedata build --stage all --seed "$ONQ_DATA_SEED" \
  --out "$DATA" --benchmarks "$BENCH" --cache "$CACHE"
"$DATA_PY" -m OnQ preparedata check --benchmarks "$BENCH" \
  --opd "$DATA/opd" --blockap "$DATA/blockap"
```

The default recipe produces 25,600 training prompts, 256 held-out prompts and
4,096 calibration windows. The repository distributes preparation scripts;
source datasets are downloaded separately. All 90 AIME evaluation questions are
included in the benchmark index used by selection and the overlap check. Reuse
`$AIME_DATA` for evaluation. See [data sources and quotas](../../OnQ/scripts/data/README.md).

After the builder succeeds, convert the sampled OPD pool in the training environment:

```bash
export PREPARED=/absolute/new/opd_prepared
"$TRAIN_PY" -m OnQ preparedata prompts \
  --train "$DATA/opd/opd_prompts.jsonl" \
  --validation "$DATA/opd/opd_val_prompts.jsonl" \
  --output "$PREPARED"
```

Expected outputs: `train.parquet` (51,200 rows), `validation.parquet` (256 rows)
and a manifest containing sampling and file identity records.

## 2. calibration

The `paper_tokens` loader consumes the builder's token IDs directly, preserving
window order and boundaries. It validates vocabulary bounds, lengths, unique
window IDs and counts. Use the same tokenizer for data preparation and the
selected dense model; do not mix tokenizers. The recipe uses 4,096 windows of
2,048 tokens, two epochs per block and no held-out calibration windows.

```bash
export DENSE=/absolute/hf/Qwen3-1.7B/snapshots/REPLACE_WITH_40_HEX_REVISION
export MODEL_SIZE=1.7b
export MODEL_CONFIG=qwen3_1p7b
export BITS=2
export CALIBRATION=/absolute/new/calibration_run
"$CALIBRATION_PY" -m OnQ calibration \
  --model "$MODEL_CONFIG" --model-path "$DENSE" --bits "$BITS" --gpus 0 \
  --calibration-recipe paper_tokens \
  --calibration-path "$DATA/blockap/train_tokens.jsonl.gz" --run-dir "$CALIBRATION"
export LATENT="$CALIBRATION/calibration_latent"
```

Use `qwen3_4b`/`4b` or `qwen3_8b`/`8b` for the larger models, and repeat for W2/W3.
For 8B, use `--gpus 0,1` to match the original two-GPU calibration. Each GPU
processes one sample; outputs are gathered before the global mean MSE and a
single optimizer update. `--dry-run` prints the resolved command.
OPD takes the FP32 `calibration_latent` checkpoint; see
[checkpoint formats](../../OnQ/docs/CHECKPOINTS.md).

## 3. On-policy distillation

### Start the teacher

Prepare a private service directory from the dense HF snapshot. Its directory
name must be the immutable 40-hex model revision. The helper hashes every snapshot
file, checks weight-shard coverage, creates a private token, and writes both the
server and client specifications:

```bash
export TEACHER_GPU=4
export SERVICE=/absolute/new/private/service
"$TRAIN_PY" -m OnQ opd teacher prepare --model "$DENSE" --root "$SERVICE"
export UNION_SPEC="$SERVICE/union_teacher.json"
CUDA_VISIBLE_DEVICES="$TEACHER_GPU" "$TRAIN_PY" -m \
  OnQ opd serve --root "$SERVICE"
```

In the actor terminal, check the service and a longest-context scoring request:

```bash
"$TRAIN_PY" -m OnQ opd teacher probe --root "$SERVICE" --wait-seconds 300
"$TRAIN_PY" -m OnQ opd teacher probe --root "$SERVICE" --tokens 20481
```

The probe validates source hashes, authentication, response identity, exact
candidate coverage, and probability normalization. The long probe checks context
capacity at two scoring positions. For a separate actor host, copy the private client
files securely and edit the client URL/token path for your SSH tunnel or HTTPS
endpoint. Do not expose the bearer token in a command or log.

The teacher runs in the foreground; keep that terminal open and use a second
terminal with the same environment variables for training. Allocate separate
GPUs to the teacher and students. This service provides `/score` and `/metadata`.

### Prepare training

Choose a new checkpoint directory and a separate active-log directory.
One command resolves the pinned veRL recipe and applies OnQ's data, loss and curriculum settings:

```bash
export STUDENT_GPUS=0,1,2,3
IFS=, read -r -a student_gpu_ids <<< "$STUDENT_GPUS"
export STUDENT_GPU_COUNT="${#student_gpu_ids[@]}"
export RUN_TAG="repro_$(date -u +%Y%m%dT%H%M%SZ)"
export CHECKPOINT_PARENT=/absolute/checkpoints
export LOG_ROOT=/absolute/local/active_logs
export RUN="$CHECKPOINT_PARENT/onq_${MODEL_SIZE}_w${BITS}_${RUN_TAG}"
mkdir -p "$CHECKPOINT_PARENT" "$LOG_ROOT"
"$TRAIN_PY" -m OnQ opd prepare \
  --source "$VERL" --model "$LATENT" --teacher "$DENSE" \
  --data-dir "$PREPARED" --model-size "$MODEL_SIZE" --bits "$BITS" \
  --student-gpus "$STUDENT_GPU_COUNT" --lr 1e-6 --run-dir "$RUN" \
  --union-teacher-spec "$SERVICE/union_teacher.json" \
  --active-dir "$LOG_ROOT/$(basename "$RUN")"
```

The checkpoint parent and log-root ancestors must exist and be writable.
The active log directory must be outside the checkpoint run and have the run basename.
Preparation writes `resolved.yaml` and `PREPARED.json`.

Inspect the **effective** configuration before launch:

| Setting | OnQ configuration |
| --- | --- |
| Objective | Current teacher top20 ∪ current actor top20 + rest; detached adaptive FKL/RKL |
| Batch / rollout count | 128 prompts; one response per prompt |
| Steps / response cap | 1–50: 4,096; 51–100: 8,192; 101–final: 16,384 |
| Prompt limit | 4,096 |
| Validation | Same 256 held-out prompts every 50 steps; greedy 16K |
| Saves | Every 50 updates; native model/optimizer/data state |
| LR / trainability | `--lr` (default 1e-6); optional `--quant-lr` for scales |

The preparer sets **400 updates for 1.7B and 300 for 4B/8B**, matching the
main training setup. `--model-size` is required. Match this label to the actual
latent model. These final steps select the checkpoints reported in the main results.
The learning rate is constant with no warmup.

### Train and resume

The launcher validates the resolved training configuration, checkpoint state and
GPU topology before starting workers. See
[validation](../../OnQ/docs/METHOD.md#paper-prompt-pool-and-validation) for the held-out evaluation protocol.

```bash
"$TRAIN_PY" -m OnQ opd train \
  --source "$VERL" --repo "$QAT" --config "$RUN/resolved.yaml" \
  --checkpoint-parent "$CHECKPOINT_PARENT" --gpus "$STUDENT_GPUS"
```

These examples use student GPUs 0–3 and teacher GPU 4 on one host. Reserve
those GPUs before launch; change the IDs to your allocation. The supervisor supports
1, 4 or 8 student GPUs, matching `--student-gpus` in preparation. The archived
1.7B W2 run used eight student GPUs; use an eight-GPU actor allocation and a separate teacher allocation when matching that run.

Resume only from complete native state, into a new run:

```bash
export RECOVERY="$CHECKPOINT_PARENT/onq_recovery_${RUN_TAG}"
"$TRAIN_PY" -m OnQ opd resume \
  --checkpoint "$RUN/global_step_50" --run-dir "$RECOVERY"
```

It inherits the absolute final step, objective and optimizer/data contract.
Check the generated config and launch the supervisor with `--config "$RECOVERY/resolved.yaml"`. Retain the
original run and lock.

## 4. evaluation

### Export the checkpoint

In the compatible training environment, select the actual checkpoint:

```bash
case "$MODEL_SIZE" in 1.7b) export STEP=400 ;; 4b|8b) export STEP=300 ;; esac
export EXPORT=/absolute/new/exports/onq_model
"$TRAIN_PY" -m OnQ evaluation export "$RUN/global_step_$STEP" "$EXPORT" \
  --checkpoint-format native_verl --expected-bits "$BITS" --expected-group-size 128
```

The exporter requires complete native shards and data state, verifies the saved
QAT/tokenizer contract and refuses an existing output. This is a materialized
fake-quant inference model, not an INT2/INT3 execution kernel or a resumable
latent checkpoint. Keep native state separately.

### Serve and evaluate

After training, stop the teacher service in its terminal and choose a free inference GPU.
In a separate terminal with the same environment variables:

```bash
export EVAL_GPU=4
CUDA_VISIBLE_DEVICES="$EVAL_GPU" VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_DISABLE_COMPILE_CACHE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
"$TRAIN_PY" -m vllm.entrypoints.cli.main serve "$EXPORT" \
  --served-model-name onq --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
  --max-model-len 40960 --max-num-seqs 128
```

These settings match the recorded main-evaluation server: BF16, TP=1, 90% GPU
memory utilization, 40,960 total tokens and at most 128 concurrent sequences.
The output cap remains 32,768 tokens. The context limit must fit the input plus
the selected output cap; reduce the output
cap only for an explicitly intended lower-budget evaluation. Run the evaluation
commands below with `$EVAL_PY` (activate its environment for the `python` examples).

```bash
source "$ENVS/eval/bin/activate"
```

### Install and run

Install the client in a separate Python 3.11 environment:

```bash
python -m pip install -r OnQ/deployment/eval.lock
python -m pip install --no-deps -e .
python -m OnQ evaluation run \
  --model onq --api-url http://127.0.0.1:8000/v1/chat/completions \
  --benchmark math500 --seed 42 --output /absolute/results/math500_seed42 \
  --dry-run
```

Remove `--dry-run` to evaluate. Use seeds 43 and 44 with separate output paths
for the other two evaluation seeds. All six OnQ main experiments use seeds 42, 43 and 44. `EVAL_API_KEY` supplies an optional API key;
this wrapper does not write it to its requested config. Existing output directories are refused.

Decoding follows the recorded main-table clients: temperature 0.6, top-k 20,
top-p 1.0, 32K output limit, thinking enabled, one completion, zero-shot by
default. Dataset selection uses seed 42 independently of the generation seed.
The endpoint must support `top_k` and `chat_template_kwargs.enable_thinking`.

| CLI benchmark | EvalScope adapter |
| --- | --- |
| `gsm8k` | `gsm8k` |
| `math500` | `math_500` |
| `aime90` | `aime24`, explicitly prepared 90-question input |
| `humaneval_plus` | `humaneval_plus` |
| `mbpp_plus` | `mbpp_plus` |
| `livecodebench` | `live_code_bench` |
| `mmlu_redux` | `mmlu_redux` |
| `gpqa` | `gpqa_diamond` |
| `ifeval` | `ifeval` |

Use the pinned local inputs in [Section 5](#prepare-the-special-evaluation-inputs-once) for the paper benchmarks.
`--local-data` selects the prepared input directory; `--dataset-args` accepts
additional EvalScope adapter options. Build the Docker scorer images below
for code benchmarks.

### AIME 2024–2026

Reuse `$AIME_DATA`, prepared before training-data selection in Section 1. It
contains 30 questions per year and 90 distinct IDs. Keep the same question records
for decontamination and evaluation; do not prepare a different AIME set here.

```bash
"$EVAL_PY" -m OnQ evaluation run \
  --model onq --api-url http://127.0.0.1:8000/v1/chat/completions \
  --benchmark aime90 --local-data "$AIME_DATA" \
  --output /absolute/results/aime90_seed42
```

### Full paper code-scoring route

HumanEval+ and MBPP+ are routed to full EvalPlus 0.3.1 rather than native
EvalScope Plus scoring. Prompts and initial extraction follow EvalScope 1.5.1;
MBPP additionally selects an entry-point-containing code block if necessary.
The full 164/378 task sets are required. The score requires both base and plus
checks to pass. No test data or generated answers are distributed.

```bash
python -m pip install -r OnQ/deployment/eval.lock
python -m OnQ evaluation plus --output /absolute/new/plus_data
docker build -f OnQ/evaluation/evalscope/Dockerfile.plus -t onq-evalplus:0.3.1 OnQ/evaluation/evalscope
docker build -f OnQ/evaluation/evalscope/Dockerfile.lcb -t onq-lcb:1.5.1 OnQ/evaluation/evalscope
python -m OnQ evaluation run --model onq \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --benchmark mbpp_plus --local-data /absolute/new/plus_data \
  --max-tokens 32768 --seed 42 --output /absolute/new/mbpp_result
```

LiveCodeBench runs the pinned scorer in a no-network container with a six-second
per-test timeout and eight CPU judges. Scorer images use Python 3.12.13 and
NumPy 2.2.6. Infrastructure errors stop evaluation.

The main recipe uses `--max-tokens 32768`; `8192` is available for lower-cap
experiments. `requested_config.json` records the decoding settings. Keep the
predictions, reviews and reports for aggregation.

## 5. Six main experiments and all evaluation commands

This section covers only OnQ's six main settings and nine benchmarks. Run the
[environment](#environment) and [data preparation](#1-preparedata) commands once.
Data sampling uses the seed you selected; the prepared prompt order uses seed 42.
The three evaluation seeds below control generation. Each model/bit setting
uses one training run.

| Model | Calibration config | Weight / activation bits | On-policy updates | Final checkpoint |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B | `qwen3_1p7b` | 2 / 16 | 400 | `global_step_400` |
| Qwen3-1.7B | `qwen3_1p7b` | 3 / 16 | 400 | `global_step_400` |
| Qwen3-4B | `qwen3_4b` | 2 / 16 | 300 | `global_step_300` |
| Qwen3-4B | `qwen3_4b` | 3 / 16 | 300 | `global_step_300` |
| Qwen3-8B | `qwen3_8b` | 2 / 16 | 300 | `global_step_300` |
| Qwen3-8B | `qwen3_8b` | 3 / 16 | 300 | `global_step_300` |

### Run calibration and on-policy distillation for each setting

Point these variables to the dense Hugging Face snapshots listed in
[paper provenance](../../OnQ/provenance/README.md). The last path component is the
40-character revision. Use the corresponding model's
snapshot for both calibration and the teacher. All six runs use group size 128.
Run the following in Bash on one reserved host with five GPUs; the six calls are
sequential and each stops its teacher before the next starts. This is the four-student-GPU
recipe example; for a run with eight actors, change `STUDENT_GPUS` and use a separate
teacher allocation as described above.

```bash
export DENSE_1P7B=/absolute/hf/Qwen3-1.7B/snapshots/REPLACE_WITH_40_HEX_REVISION
export DENSE_4B=/absolute/hf/Qwen3-4B/snapshots/REPLACE_WITH_40_HEX_REVISION
export DENSE_8B=/absolute/hf/Qwen3-8B/snapshots/REPLACE_WITH_40_HEX_REVISION
export REPRO_ROOT=/absolute/new/onq_main
export STUDENT_GPUS=0,1,2,3
IFS=, read -r -a student_gpu_ids <<< "$STUDENT_GPUS"
export STUDENT_GPU_COUNT="${#student_gpu_ids[@]}"
export CALIBRATION_GPU=0
export CALIBRATION_GPUS_8B=0,1
export TEACHER_GPU=4
mkdir -p "$REPRO_ROOT"/{calibration,checkpoints,active,services,exports}

run_onq () (
  set -euo pipefail
  local size="$1" bits="$2" model_config="$3" dense="$4" steps="$5"
  local tag="qwen3_${size}_w${bits}"
  local calibration="$REPRO_ROOT/calibration/$tag"
  local calibration_gpus="$CALIBRATION_GPU"
  if [[ "$size" == 8b ]]; then calibration_gpus="$CALIBRATION_GPUS_8B"; fi
  local run="$REPRO_ROOT/checkpoints/$tag"
  local service="$REPRO_ROOT/services/$tag"
  local teacher_pid=""
  stop_teacher () {
    if [[ -n "$teacher_pid" ]]; then
      kill "$teacher_pid" 2>/dev/null || true
      wait "$teacher_pid" 2>/dev/null || true
    fi
  }
  trap stop_teacher EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  "$CALIBRATION_PY" -m OnQ calibration \
    --model "$model_config" --model-path "$dense" --bits "$bits" \
    --gpus "$calibration_gpus" --calibration-recipe paper_tokens \
    --calibration-path "$DATA/blockap/train_tokens.jsonl.gz" \
    --run-dir "$calibration"

  "$TRAIN_PY" -m OnQ opd teacher prepare --model "$dense" --root "$service"
  CUDA_VISIBLE_DEVICES="$TEACHER_GPU" "$TRAIN_PY" -m OnQ opd serve \
    --root "$service" > "$service/server.log" 2>&1 &
  teacher_pid=$!
  "$TRAIN_PY" -m OnQ opd teacher probe --root "$service" --wait-seconds 300
  "$TRAIN_PY" -m OnQ opd teacher probe --root "$service" --tokens 20481

  "$TRAIN_PY" -m OnQ opd prepare \
    --source "$VERL" --model "$calibration/calibration_latent" --teacher "$dense" \
    --data-dir "$PREPARED" --model-size "$size" --bits "$bits" \
    --student-gpus "$STUDENT_GPU_COUNT" --lr 1e-6 --run-dir "$run" \
    --union-teacher-spec "$service/union_teacher.json" \
    --active-dir "$REPRO_ROOT/active/$tag"
  "$TRAIN_PY" -m OnQ opd train \
    --source "$VERL" --repo "$QAT" --config "$run/resolved.yaml" \
    --checkpoint-parent "$REPRO_ROOT/checkpoints" --gpus "$STUDENT_GPUS"
  "$TRAIN_PY" -m OnQ evaluation export \
    "$run/global_step_$steps" "$REPRO_ROOT/exports/$tag" \
    --checkpoint-format native_verl --expected-bits "$bits" --expected-group-size 128
)

run_onq 1.7b 2 qwen3_1p7b "$DENSE_1P7B" 400
run_onq 1.7b 3 qwen3_1p7b "$DENSE_1P7B" 400
run_onq 4b   2 qwen3_4b   "$DENSE_4B"   300
run_onq 4b   3 qwen3_4b   "$DENSE_4B"   300
run_onq 8b   2 qwen3_8b   "$DENSE_8B"   300
run_onq 8b   3 qwen3_8b   "$DENSE_8B"   300
```

You can invoke one `run_onq` line at a time. Existing output directories are refused;
use the resume workflow above for an interrupted training run.

### Prepare the special evaluation inputs once

Reuse `$AIME_DATA` from Section 1, containing AIME I and II for 2024–2026.
The same 90 questions were included in training-data decontamination. Prepare
the five native benchmarks, full EvalPlus data and the scorer images below.

```bash
export EVAL_DATA=/absolute/new/onq_eval_data
mkdir -p "$EVAL_DATA"
"$DATA_PY" -m OnQ evaluation native --output "$EVAL_DATA/native" --cache "$CACHE"
"$EVAL_PY" -m OnQ evaluation plus --output "$EVAL_DATA/plus"
docker build -f OnQ/evaluation/evalscope/Dockerfile.plus \
  -t onq-evalplus:0.3.1 OnQ/evaluation/evalscope
docker build -f OnQ/evaluation/evalscope/Dockerfile.lcb \
  -t onq-lcb:1.5.1 OnQ/evaluation/evalscope
```

For LiveCodeBench, use the cumulative **release-v6 (1,055 problems)**. The following downloads the
six pinned source files already used by the data-preparation code and writes
EvalScope's local input. The download includes private test cases (about 4.5 GB).

```bash
"$DATA_PY" - <<'PYLCB'
import json, os
from pathlib import Path
from huggingface_hub import hf_hub_download
from OnQ.scripts.data.prepare_benchmark_texts import PINS
repo, revision, files = PINS["lcb"]
output = Path(os.environ["EVAL_DATA"]) / "livecodebench_v6"
output.mkdir(parents=True, exist_ok=False)
ids = set()
with (output / "release_latest_test.jsonl").open("w") as stream:
    for name in files:
        path = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
        with open(path) as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["question_id"] in ids:
                    raise ValueError("Duplicate LiveCodeBench question ID")
                ids.add(row["question_id"])
                stream.write(json.dumps(row) + "\n")
if len(ids) != 1055:
    raise ValueError(f"Expected 1055 release-v6 problems, got {len(ids)}")
(output / "SOURCE.json").write_text(json.dumps(dict(repo=repo, revision=revision, files=files, count=len(ids)), indent=2))
PYLCB
```

### Evaluate all nine benchmarks with three seeds

Run this function for each exported model. It starts a local inference server,
waits for readiness, runs **9 benchmarks × 3 seeds**, and stops that server.
Reserve one inference GPU with enough memory for the chosen model and 40K context.
Dataset access may require accepting the original dataset's terms beforehand.
All nine benchmarks use the prepared local inputs. The five native datasets
are downloaded at fixed revisions by `evaluation native`; `SOURCE.json` records
their source versions, row counts and file hashes.

```bash
export RESULTS=/absolute/new/onq_results
export EVAL_GPU=0
mkdir -p "$RESULTS"

evaluate_onq () (
  set -euo pipefail
  local size="$1" bits="$2"
  local tag="qwen3_${size}_w${bits}"
  local out="$RESULTS/$tag"
  mkdir "$out"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_DISABLE_COMPILE_CACHE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  "$TRAIN_PY" -m vllm.entrypoints.cli.main serve "$REPRO_ROOT/exports/$tag" \
    --served-model-name onq --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
    --max-model-len 40960 --max-num-seqs 128 \
    > "$out/server.log" 2>&1 &
  local server_pid=$!
  stop_server () {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }
  trap stop_server EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  local ready=0
  for ((attempt=0; attempt<360; attempt++)); do
    kill -0 "$server_pid" 2>/dev/null || { cat "$out/server.log"; exit 1; }
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
  done
  [[ "$ready" == 1 ]] || { echo "Inference server did not become ready" >&2; exit 1; }

  for seed in 42 43 44; do
    for benchmark in gsm8k math500 aime90 humaneval_plus mbpp_plus livecodebench mmlu_redux gpqa ifeval; do
      local args=()
      case "$benchmark" in
        aime90) args=(--local-data "$AIME_DATA") ;;
        humaneval_plus|mbpp_plus) args=(--local-data "$EVAL_DATA/plus") ;;
        livecodebench) args=(--local-data "$EVAL_DATA/livecodebench_v6") ;;
        *) args=(--local-data "$EVAL_DATA/native/$benchmark") ;;
      esac
      "$EVAL_PY" -m OnQ evaluation run \
        --model onq --api-url http://127.0.0.1:8000/v1/chat/completions \
        --benchmark "$benchmark" --seed "$seed" --max-tokens 32768 \
        --output "$out/${benchmark}_seed${seed}" "${args[@]}"
    done
  done
)

evaluate_onq 1.7b 2
evaluate_onq 1.7b 3
evaluate_onq 4b 2
evaluate_onq 4b 3
evaluate_onq 8b 2
evaluate_onq 8b 3
```

This produces **162 benchmark runs**. For each model/bit setting, report the mean
score across seeds 42/43/44 for each benchmark, then the unweighted mean of the
nine benchmark means. Use prompt-level strict accuracy for IFEval and full
base-plus accuracy for HumanEval+/MBPP+. Express all scores as percentages.
For the appendix, compute sample standard deviation (`ddof=1`) across the three
seed scores. For the Avg. column, first average the nine scores within each seed,
then compute the mean and sample SD of those three averages.

## Reproduction records

See [paper provenance](../../OnQ/provenance/README.md) for recovered model revisions,
data hashes, archived configuration fields and remaining per-cell gaps.

## IFEval scoring resources and coverage

Before evaluating, provision the tokenizer in the evaluation environment:

```bash
python -m OnQ.evaluation.evalscope.resources --download
python -m OnQ.evaluation.evalscope.resources
```

The eval installer provisions this resource. Offline installations can provide
`tokenizers/punkt_tab/english/` through `NLTK_DATA`.

IFEval checks the tokenizer before generation and requires 541 distinct scored
items. Scorer errors and incomplete outputs stop evaluation. Tokenizer hashes
and scored-item coverage are saved in `SCORER_RESOURCES.json` and
`SCORING_COVERAGE.json`.
