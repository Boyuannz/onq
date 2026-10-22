# Paper reproduction records

[paper_records.json](paper_records.json) contains the recovered model revisions,
data hashes and archived configuration fields.

| Dense model | Recorded revision |
| --- | --- |
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |

The inventory includes model/config/tokenizer file hashes. The 1.7B W2 and
4B original configs independently record the matching teacher revision.

The expanded OPD train/validation files contain 51,200/256 rows. Their byte
hashes were recomputed and match the archived 1.7B W2, 4B W2 resume-from-250,
and 4B W3 configs. Original prompt-file hashes are separately labeled; do not
compare an expanded parquet against the original prompt-file hash. The
calibration-token hash identifies the archived 8B pool of 4,096 windows of
2,048 tokens. It has not been established as the pool for every paper cell.

The config records contain an explicit whitelist of numerical fields and
source-config hashes. Use the root README to generate runnable configs.
Original 4B configs permit 400 updates; the paper selects checkpoint 300.
The README commands use the paper's
selected training lengths and numerical settings. Pinned evaluation sources
and preparation commands remain in the main README.

Still needed for exact table reproduction:

- Selected 1.7B W3 and superseding 8B W2 original config/source/checkpoint lineage.
- Complete per-cell calibration input membership and model-to-latent linkage.
- Original manually reviewed prompt membership when regenerating the exact pool.
- Full 8B W3 config and all six checkpoint/export/multi-seed result hash chains.

The paper path uses integer-zero forward,
calibration weight/scale/zero training, and OPD weight/scale training with zero frozen.
