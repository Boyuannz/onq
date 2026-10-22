# Implementation sources

| Component | Source | Version and modifications |
|---|---|---|
| `calibration/upstream` | [ReasoningQAT](https://github.com/yasu0001/ReasoningQAT) | `c36060fefb05aa8c29dcad47b52feacf95201d37`; latent checkpoint saving and lossless reload support. See [provenance and patch hashes](calibration/upstream/UPSTREAM_PROVENANCE.md). |
| `calibration` wrappers | OnQ | Block-AP orchestration, data adapters and calibration checks. |
| `quantization` | EfficientQAT-style quantization and adapted helpers | Local QAT integration and checkpoint/export support. The original import revision is not recorded; individual source attributions remain in the code. |
| External `verl` | [veRL](https://github.com/verl-project/verl) | `7aed6b230776f963fa09509c10d9c3a767d1102c` with the bundled, hash-verified rollout-transfer backport. |
| Configuration, data preparation and training orchestration | OnQ | Project implementation. |
| Evaluation | [EvalScope](https://github.com/modelscope/evalscope) | Separately installed version 1.5.1. |

Original copyright and license notices are retained in the source files.
The veRL license and notice are in [`licenses/verl`](licenses/verl).
