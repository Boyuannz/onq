# OnQ checkpoints

## Calibration → OPD

Use `calibration_latent/`, saved before rounding or casting, to initialize OPD.
Weights, scales and zero points must be FP32; all blocks must be present and
`latent_checkpoint.json` must report `COMPLETE`. Preparation checks tensor
structure, source identity and the producer's completion records.

The original stage-specific quantization arithmetic is preserved: with
`s = clamp(abs(scale), 1e-4, 1e4)`, Block-AP rounds `weight / (s + 1e-6)`,
while OPD rounds `weight / s`. The same latent tensors can therefore yield
different integer codes near rounding boundaries when entering OPD.

`calibration_fakequant/` is an inference export. It cannot replace latent weights
for training. Calibration starts OPD with a new optimizer; it does not provide
an interrupted Block-AP optimizer resume.

## Resume OPD

Use a complete native veRL `global_step_N` checkpoint containing model,
optimizer, extra state and `data.pt`. Resume into a new run directory using
`python -m OnQ opd resume`, retaining the saved objective and configuration.
Materialized HF exports do not contain optimizer or dataloader state.

## Export

Training retains FP32 latent state. Rollout and inference export materialize
quantized weights separately. Keep the native checkpoint when creating an HF
export. Follow [the reproduction guide](REPRODUCE.md) for commands.
