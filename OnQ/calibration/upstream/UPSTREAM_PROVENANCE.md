# Upstream Provenance

- Repository: `https://github.com/yasu0001/ReasoningQAT.git`
- Commit: `c36060fefb05aa8c29dcad47b52feacf95201d37`
- Imported: 2026-08-09
- Scope: ReasoningQAT Stage1 Block-AP source used by OnQ calibration

The unused standalone HF conversion script and empty Triton mixin are omitted.
Python files have a provenance comment header. Unpatched payloads match the
commit above; `PAYLOAD_SHA256.json` records the original hashes and the local
patch hashes separately. The documented local patches preserve latent state before casting/materialization,
enable latent export, and require FP32 latent sources for training reload. The
block runner also supports the original two-GPU global-batch-2 execution, aborts
on nonfinite activations/loss, and skips absent validation caches during disk offload. See `OnQ/docs/CHECKPOINTS.md`.
The identical shadowed upstream clamp_ste definition has been removed;
the surviving function and quantization arithmetic are unchanged.
The unified runner does not edit this directory. When a custom calibration recipe
or a pinned local dense checkpoint requires an import compatibility change, it
creates a run-scoped copy and records the exact patch in that run's manifest.
