# EvalScope evaluation

The release uses **EvalScope 1.5.1**. The vendored Qwen-Math harness and older
local greedy HumanEval/IFEval runners have been removed. Model serving remains
separate: start an OpenAI-compatible server using the exported model and its
Qwen3 chat template, with enough context for the prompt plus 32,768 output tokens.

## Install and run

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
for additional evaluation samples. `EVAL_API_KEY` supplies an optional API key;
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

Use the pinned local inputs in [the reproduction guide](../../docs/REPRODUCE.md#prepare-the-special-evaluation-inputs-once) for the paper benchmarks.
`--local-data` selects the prepared input directory; `--dataset-args` accepts
additional EvalScope adapter options. Build the Docker scorer images below
for code benchmarks.

## AIME 2024–2026

Obtain the official question records locally, normalize each to `id`, `year`,
`problem`, and `answer`, then run:

```bash
python -m OnQ evaluation aime \
  --input /absolute/aime2024.jsonl /absolute/aime2025.jsonl /absolute/aime2026.jsonl \
  --output /absolute/data/aime90
python -m OnQ evaluation run \
  --model onq --api-url http://127.0.0.1:8000/v1/chat/completions \
  --benchmark aime90 --local-data /absolute/data/aime90 \
  --output /absolute/results/aime90_seed42
```

Preparation requires 30 questions per year and 90 distinct IDs.

## Full paper code-scoring route

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

### Pinned native inputs

Prepare GSM8K, MATH-500, GPQA-Diamond, IFEval and MMLU-Redux with the data
environment, then pass the corresponding directory to `--local-data`:

```bash
"$DATA_PY" -m OnQ evaluation native --output "$EVAL_DATA/native" --cache "$CACHE"
```

The output contains one directory per CLI benchmark name and a `SOURCE.json`
with fixed source revisions, row counts and file hashes. This avoids loading
a moving dataset revision when reproducing the main experiments.

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
