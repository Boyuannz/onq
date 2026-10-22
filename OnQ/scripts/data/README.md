# Data selection: OPD prompts and Block-AP calibration windows

One seeded pipeline that selects both training datasets from pinned public sources:

- **OPD prompts**: 25,600 train + 256 validation single-turn prompts. The student
  generates the answers during OPD.
- **Block-AP calibration**: 4,096 windows of 2,048 tokens. Each window is the head of
  one original conversation, question plus original answer.

Selection is **uniform random over each full source**. The builder draws a seeded
permutation of all eligible source rows and scans it in order. Records that fail a
rule are skipped, and the next record in the same order takes their place (in-order
refill), so quotas stay exact. Nothing is ranked by difficulty, length, teacher score
or answer correctness. The same `--seed` and recipe reproduce the same data. A new
seed gives a different sample that is equally valid.

Test-set decontamination is a hard gate. If the written files contain any exact or
contained benchmark text, any duplicate prompt, or any OPD train/validation overlap,
the build fails.

This directory prepares inputs for both calibration and OPD. For the public preparation workflow, use [OnQ](../../docs/DATA.md).
We release scripts only, not data files or reference hashes.
This pipeline reuses the Block-AP paraphrase matcher (`text_quality.EvalIndex`).

## Run

```bash
pip install -r OnQ/scripts/data/requirements.txt   # separate data environment

# 1. Prepare the same 90 AIME questions used for evaluation (30 per year).
python -m OnQ evaluation aime \
    --input /absolute/aime2024.jsonl /absolute/aime2025.jsonl /absolute/aime2026.jsonl \
    --output AIME_DATA
python -m OnQ preparedata benchmarks --aime90 AIME_DATA --out BENCH/HELDOUT_QUERIES.jsonl

# 2. Build. --stage opd | blockap | all.
python -m OnQ preparedata build --stage all --seed 42 --out OUT \
    --benchmarks BENCH/HELDOUT_QUERIES.jsonl --cache CACHE

# 3. Optional: re-run the gate on any built directory (exit code 1 on any hit).
python -m OnQ preparedata check --benchmarks BENCH/HELDOUT_QUERIES.jsonl \
    --opd OUT/opd --blockap OUT/blockap

```

`--out` must be a new directory; the builder never overwrites. Source row groups are
cached under `--cache`. `--readonly-cache DIR` reuses an existing cache without writing
to it; the previous build caches use the same layout. The first full build downloads
the row groups it needs from the Hugging Face Hub. The benchmark builder also downloads
LiveCodeBench release_v6, about 4.5 GB, because its test cases are stored inline.

### Outputs

| Path | Contents |
|---|---|
| `OUT/opd/opd_prompts.jsonl`, `opd_val_prompts.jsonl` | `messages` (user prompt only), `id`, `source`, `domain`, `source_repo`, `revision`, `row_index`, `prompt_sha256`, `prompt_tokens`, `split` |
| `OUT/blockap/train_tokens.jsonl.gz` | pre-tokenized windows: `input_ids` (2048), `segments` (source record and offsets) |
| `OUT/*/raw_records.jsonl.gz` | the untouched source row, the OCR2 original problem record, provenance (file, row group, row) and hashes |
| `OUT/*/rejections.jsonl.gz` | every scanned-and-rejected row with its reason |
| `OUT/*/MANIFEST.json` | per-source quota, scanned, accepted, rejection counts, eligible rows, catalog hash; file sha256 and row counts |
| `OUT/GATES.json` | decontamination and duplicate gate report |
| `OUT/BUILD.json` | command, seed, recipe and its hash, tokenizer pin and file hashes, benchmark file hash, code hashes, git revision, library versions, status |

The formats match the OnQ release. The OPD rows feed
`OnQ/integrations/verl_v080/opd`. Block-AP rows are token ids and are consumed in their
recorded order; do not re-tokenize them.

## Default recipe (`sampling_recipe.json`, the OnQ composition)

Tokenizer and chat template: `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`,
which is the same tokenizer files as every Qwen3 size.

**OPD**: prompt of at most 4,096 tokens, measured with the Qwen3 template,
`add_generation_prompt=True` and `enable_thinking=True`.

| Source | Dataset @ revision | Subset | Train | Val |
|---|---|---|---:|---:|
| omr_cot | nvidia/OpenMathReasoning @ d3d08664 | `data/cot-*` | 7,680 | 77 |
| osr2 | nvidia/OpenScienceReasoning-2 @ 174b02c9 | all (multi-discipline) | 3,840 | 38 |
| ocr2 | nvidia/OpenCodeReasoning-2 @ eadf5359 | `train/python/` only | 7,680 | 77 |
| tulu_wildchat | allenai/tulu-3-sft-mixture @ b14afda6 | `source = …tulu_v3.9_wildchat_100k` | 2,240 | 23 |
| tulu_no_robots | same | `…no_robots_converted` | 1,440 | 14 |
| tulu_flan | same | `…flan_v2_converted` | 1,440 | 14 |
| if_multi | allenai/IF_multi_constraints_upto5 @ 2e3a7740 | all | 1,280 | 13 |

**Block-AP**: windows of 2,048 tokens.

| Source | Dataset @ revision | Subset | Windows |
|---|---|---|---:|
| general_dolci | allenai/Dolci-Think-SFT-7B @ 72ec0fe3 | the two WildChat-R1 subsets (`dataset_source`), union drawn uniformly | 819 |
| if_dolci | same | `if_qwq_reasoning_verified_filtered_decontam_v2` | 205 |
| omr_cot | nvidia/OpenMathReasoning @ d3d08664 | `data/cot-*`, `generated_solution` | 1,229 |
| osr2 | nvidia/OpenScienceReasoning-2 @ 174b02c9 | all, `output` | 614 |
| ocr2 | nvidia/OpenCodeReasoning-2 @ eadf5359 | `train/python/`, `r1_generation` | 1,229 |

`sampling_recipe.json` records the full 40-character revisions. OCR2 rows store only a
reference to their problem. The builder resolves the original problem text from
BAAI/TACO @ d593ed0a, deepmind/code_contests @ 802411c3, open-r1/codeforces @ fbe3f6e9
and codeparrot/apps @ 21e74ddf, all pinned in `dataset_sources.py`.

To change the composition, edit a copy of `sampling_recipe.json` and pass `--recipe`. Each source
takes a dataset spec, an optional `cohort` (a column and its allowed values), and either
`train`/`val` counts (OPD) or `windows` (Block-AP).

## Pipeline

1. **Order.** Each source gets its own seed, `sha256(f"{seed}:{name}")`; Block-AP
   sources use `blockap:{name}`. The permutation covers the global row coordinates
   of every row group of every shard. Each integer seed defines a valid draw
   order. Batching changes I/O only, never membership.
2. **Text.** The builder takes the original problem only. It adds no system prompt,
   suffix, answer instruction or language requirement. OPD chat sources keep the
   native turns up to the last user turn. Codeforces problems are rendered in the
   official open-r1 `verifiable-prompts` layout without its answer preamble: time and
   memory limits, then `# Problem`, `## Input Format`, `## Output Format`,
   `## Examples` and `## Note`. `## Interaction Format` is a documented extension.
   TACO, APPS and code_contests text is written verbatim.
3. **Rules.** A rule excludes the whole record, then the refill takes the next record
   in order. The rules are listed in the next section.
4. **Block-AP windows.** The dialogue (question plus original answer) is rendered with
   the pinned Qwen3 `apply_chat_template(messages, tokenize=False)` using default
   arguments. Only dialogues of at least 2,048 tokens are used, and each contributes
   exactly its first 2,048 tokens as one window. Short dialogues are never packed.
   This matches the window rule of the ReasoningQAT Calibration reasoning data
   (`get_openthoughts_shuffled`). `format_conversations.template_audit` proves the template changed
   formatting only; if it would drop other text, the record is excluded.
5. **Split and order.** In each OPD source, the first accepted records form the train
   set and the next ones form the validation set. Train rows, validation rows and
   Block-AP windows are then shuffled with the seed.
6. **Gate.** `check_data_overlap.gate` re-reads the written files and checks them independently.

## Rules

Both stages:

- **Non-English:** any non-Latin script, or prose that langdetect (seed 0) reports as
  non-English with p ≥ 0.99 and at least 8 words (`non_english`). Block-AP checks the
  whole dialogue, including answers.
- **Missing options:** the question asks "which of the following" or "choose … below"
  but lists fewer than two options (`missing_choice_candidates`).
- **Control tokens:** Qwen chat control tokens inside source text
  (`native_chat_control_tokens`).
- **Minors:** sexual content involving minors, a recall rule
  (`sexual_content_involving_minors`). It runs on chat and IF sources only: in OnQ,
  all 27 recall hits in code and science problems were false positives, such as tree
  `child` nodes next to `.strip()`.
- **Duplicates:** an identical normalized prompt (`duplicate_question`), or a
  near-duplicate question (`near_duplicate_question`). A near-duplicate has word
  5-gram containment of at least 0.8 of the shorter text; texts under 20 five-grams
  must have identical words. Easy and hard versions of one problem count as different
  questions. OCR2 rows must also come from the train split and have a unique
  `(dataset, index)`.
- **Benchmark overlap** (`test_overlap:*`): the decontamination checks below, plus the
  rare-8-gram, same-numbers and shared-formula paraphrase matcher from
  `text_quality.py`.

OPD only:

- **Single turn:** multi-turn conversations are excluded whole
  (`not_single_turn_conversation`).
- **Prompt length:** a prompt over 4,096 tokens is excluded, never truncated.
- **Code:**
  - inline `<image>` placeholders that replace part of a sentence or formula;
  - statements with only examples;
  - Q# tasks;
  - a "following signature:" with no signature;
  - Codeforces items without input format, output format or examples;
  - statements given only as a link or PDF, or with fewer than 25 words (recall rule).
- **IF:** IF_multi constraint conflicts, from the pinned open-instruct IFEvalG
  `INSTRUCTION_CONFLICTS` plus the semantic conflict rules. The registry is fetched
  at a pinned commit and checked against its sha256.
- **Science:** the answer is given in the prompt (recall rule).
- **WildChat:** prompts of 3 words or fewer; spam or ads (recall rule); requests that
  need a URL (recall rule).

**Recall rules are strict here.** The OnQ release sent every recall hit to a
per-record human decision. This builder excludes every recall hit automatically. That
over-excludes: in OnQ, reviewers confirmed only 21 of 153 minors hits, 2 of 17 spam
hits, 6 of 38 link hits and 40 of 56 unusable-code hits. Excluding a record is always
the safe direction, and refill keeps the quotas exact.

Block-AP general data uses the two Dolci WildChat-R1 subsets; OPD general data uses
Tulu. Stages are deduplicated independently; the same question may appear in both.

## Decontamination

**Benchmark set.** `prepare_benchmark_texts.py` combines the pinned public benchmark
texts with all 90 questions from the prepared AIME evaluation input. `--aime90`
is required and uses the same year/count/ID validation as evaluation. The original
public sources use pinned Hugging Face revisions and archives checked by SHA256:

- math: MATH500, MATH test, GSM8K, AIME 2024–2026, AMC23;
- code: HumanEval (prompt and canonical solution), MBPP (prompt and code),
  LiveCodeBench release_v6;
- instruction following: IFEval, IFBench;
- knowledge and QA: GPQA-Diamond, MMLU, MMLU-Redux 2.0, ARC-Easy/Challenge,
  HellaSwag, PIQA, SIQA, WinoGrande, OpenBookQA, TruthfulQA.

The prepared AIME questions are added under `aime2024`, `aime2025`, and
`aime2026` (30 each), alongside the older pinned 2024/2025 source texts. Reuse
this prepared AIME directory for scoring. Existing training pools can be checked
against the expanded index with `check_data_overlap.py`; an earlier PASS does
not establish coverage of the newly added questions.

Multiple-choice sets appear twice, as the question alone and as the question with its
options. Benchmark source versions define the screening corpus; they do not
constrain the randomly sampled training pool.

**Method.** Normalization is QAOPD `_norm`: `re.sub(r'\s+', ' ', s).strip().lower()`.
A record is contaminated if:

- **exact:** any user turn, or the whole prompt, equals a benchmark text; or
- **contained:** a benchmark text of at least 60 characters (70,507 of 77,679 distinct
  texts) occurs inside the training text. For OPD that is the prompt; for Block-AP it
  is the whole conversation, answers included.

Benchmark texts under 60 characters are matched by equality only. The builder applies
this check, plus the paraphrase matcher, while selecting. The gate then repeats the
exact and containment checks on the written files and fails the build on any hit.

**Limits.** This is verbatim matching plus a heuristic paraphrase matcher. It does not
prove zero contamination, and reworded or translated benchmark items can pass.

The original experiments also included steps that are **not part of this script**:

- an independent audit plus a LaTeX-normalized character 5-gram TF-IDF similarity
  screen against all benchmarks, with each flagged pair decided by a person
  (confirmed paraphrase leaks excluded: 10 Block-AP and 82 OPD records);
- per-source near-duplicate and quality scans of the final OPD set;
- per-record decisions for recall hits.

A new build therefore has the verbatim guarantee but not those manual decisions. Run a
similarity screen before claiming more.

## Sampling and reproducibility

Choose any integer seed and preserve your recipe and local preparation reports to
reproduce that draw. No published training data files or reference sample hashes
are required. Different samples can change evaluation results. The automatic gates
check the conditions described above; they do not reproduce the original manual
review decisions or guarantee the absence of paraphrased benchmark overlap.

See [data preparation](../../docs/DATA.md) for the workflow commands.
