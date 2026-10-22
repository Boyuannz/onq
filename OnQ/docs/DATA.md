# Prepare data with the OnQ recipe

We release preparation scripts, not training data files or reference data hashes.
Users obtain the upstream datasets and randomly select their own training pool.
Any sampling seed is supported; no match to the authors' original samples is required.
The recipe specifies source quotas and processing rules, not fixed membership.

## Build your data

Use a separate data environment. Check the upstream datasets' access conditions
and obtain access where required. Run from the repository root:

```bash
python -m pip install -r OnQ/scripts/data/requirements.txt
export ONQ_DATA_SEED=123
export DATA=/absolute/new/data/sample_${ONQ_DATA_SEED}
export BENCH=/absolute/new/benchmark_queries.jsonl
export CACHE=/absolute/data_download_cache
export AIME_DATA=/absolute/new/onq_aime90
python -m OnQ evaluation aime \
  --input /absolute/aime2024.jsonl /absolute/aime2025.jsonl /absolute/aime2026.jsonl \
  --output "$AIME_DATA"
python -m OnQ preparedata benchmarks --aime90 "$AIME_DATA" --out "$BENCH"
python -m OnQ preparedata build --stage all --seed "$ONQ_DATA_SEED" \
  --out "$DATA" --benchmarks "$BENCH" --cache "$CACHE"
python -m OnQ preparedata check --benchmarks "$BENCH" \
  --opd "$DATA/opd" --blockap "$DATA/blockap"
```

Use the same AIME input for decontamination and evaluation: 30 questions per year,
90 distinct IDs, with `id`, `year`, `problem`, and `answer` fields. The questions
are obtained separately; this repository does not distribute them.

Choose any integer seed and a new output directory. The default recipe samples
uniformly within each source and refills rejected rows until each quota is met.
It produces 25,600 OPD training prompts, 256 validation prompts, and 4,096
calibration windows of 2,048 tokens.

See [the preparation reference](../scripts/data/README.md) for source quotas,
source revisions, filtering rules, dependencies and contamination-screening limits.
No data file or reference hash from the authors is needed. Runtime manifests
record the user's own files for consistency within their run, not comparison
against an author-provided dataset.

## Convert OPD prompts for training

In the training environment:

```bash
export PREPARED=/absolute/new/opd_prepared
python -m OnQ preparedata prompts \
  --train "$DATA/opd/opd_prompts.jsonl" \
  --validation "$DATA/opd/opd_val_prompts.jsonl" \
  --output "$PREPARED"
```

This checks counts, IDs, duplicates and train/validation separation, and creates
51,200 training rows (two passes) and 256 validation rows. Sampling the pool
uses `ONQ_DATA_SEED`; the separate training-order permutations use seeds 42 and 43.
The paper preparer accepts any pool satisfying this schema and count contract.
It does not compare the pool with the authors' training files.

The calibration output `blockap/train_tokens.jsonl.gz` contains saved token IDs;
do not feed it to a conversation-text loader or retokenize the windows. The
[reproduction guide](REPRODUCE.md) describes the current calibration
adapter release boundary and the online training commands.

Different random pools are valid inputs to the method. They need not produce
identical paper scores. Record your sampling seed, source versions and recipe
so you can reproduce your own experiment. Screening reduces benchmark overlap;
it does not prove the absence of every paraphrase or translated overlap.
