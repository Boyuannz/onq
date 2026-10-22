# OnQ objective

The implementation is the **single-teacher** implementation. It keeps
teacher top20 ∪ **current actor** top20 and a rest bucket (21–41 valid categories).
Student probabilities use full-vocabulary normalization. The teacher returns
actual probabilities for student-only candidates; missing candidates are never
filled with a guessed probability. Both tails use direct log-sum-exp reduction.

For teacher P and student Q on this common support, alpha = stopgrad(P/(P+Q)).
The loss is the response-token mean of
`sum(alpha * P * log(P/Q) + (1-alpha) * Q * log(Q/P))`.
Teacher probabilities are smoothed with epsilon=1e-8 over valid categories,
including rest. Padding categories have no mass. **Do not simplify the gradient
to Jeffreys divergence**: alpha is detached. This is coarse-grained KL, not exact
full-vocabulary KL. Weighted FKL/RKL metrics are loss terms, not parameter-gradient
contributions.

## Exact-candidate teacher service

The repository includes the Qwen3-1.7B, 4B and 8B BF16 TP1 producer, pinned to
**vLLM 0.12.0**. It injects a guarded hook into that version's prompt-score path;
a changed hook or unsupported version fails instead of silently approximating.
The full dense snapshot must be a directory named by its immutable HF commit.
The accepted context limit is 20496 tokens; input requests are bounded at 20481.
Preparation rejects prompt + response + 1 beyond this contract, considering
both data and rollout limits. The service advertises `max_input_tokens` and
`max_model_len` in `/metadata`; runtime checks both before any rollout.
Missing capacity metadata or a smaller endpoint is an admission error. Older
running services are not hot-updated by this change; a new deployment must use
the matching producer and pass its own preflight.
Other model families, teacher precisions and context limits need separate
acceptance rather than editing this service during training.

Generate `teacher.key`, `teacher_spec.json` and `union_teacher.json` with
`python -m OnQ opd teacher prepare --model SNAPSHOT --root SERVICE`.
This checks shard coverage and hashes all snapshot files. The server spec has this form:

```json
{
  "source": "/absolute/hf/snapshots/REPLACE_WITH_40_HEX_COMMIT",
  "port": 8297,
  "source_sha256": {
    "config.json": "REPLACE_WITH_SHA256",
    "model.safetensors": "REPLACE_WITH_SHA256"
  }
}
```

List the **actual complete snapshot files**, including every model shard/index
and tokenizer/config file; the example is schematic. Verify hashes against the
admitted source manifest. Keep the directory/token private (`chmod 700`/`600`).
Only after reserving a genuinely free teacher GPU:

```bash
CUDA_VISIBLE_DEVICES=TEACHER_GPU python -m \
  OnQ opd serve --root /absolute/private/service
```

The service binds loopback, verifies source hashes and requires bearer auth for
scoring. Each request binds exact input IDs, prediction positions and current
student candidate IDs by SHA256. The worker captures true normalized teacher
probabilities, including the tail. Its own transient scratch is removed only
after a successful response; request hashes and timings remain in the log.

Actor-time scoring retries connection resets, connection aborts, broken pipes
and a peer disconnect before a response, with at most three attempts and
0.25/0.5-second backoff. Each attempt uses the same input, positions, candidate
IDs, authentication and per-attempt timeout. Timeouts, TLS/DNS/HTTP errors,
invalid JSON and target-identity failures are not retried. A lost response may
cause an extra frozen-teacher forward, never an additional student update;
request counts therefore need not equal trajectory counts when a retry occurs.
Exhausted retries propagate the error. Logs identify attempts by request hash
without logging credentials or payloads. This change applies to newly loaded
code; repository updates do not hot-patch existing training processes.

The actor scores each packed sequence during its current forward. Backward
activation checkpointing recomputes student math only, not teacher RPCs. Rollout
uses no legacy teacher client. `opd_external_teacher_json` remains native routing
metadata for compatibility, and is **not** an additional active scoring service
when the adapter contains `union`.

## Paper prompt pool and validation

`python -m OnQ preparedata prompts` consumes the already accepted 25,600 training prompts and 256
held-out prompts. It checks IDs and train/validation prompt hashes, then writes
two independently seeded random permutations as one 51,200-row epoch. It does
not select a new pool or filter domains.
Keep the original prompt files and generated manifest with the experiment.

`python -m OnQ opd prepare` resolves the native recipe internally and takes a pinned Union teacher
spec, and that manifest. Its explicit contract is batch128, 400 updates for 1.7B or 300 for 4B/8B,
4K at steps1–50, 8K at51–100, and16K from step101 to the final update. Native FP32 latent handoff and
Union loss are preserved.

Every50 steps it evaluates the same256 held-out prompts, using greedy rollout
at a fixed16K response cap. Actor and teacher score those responses at objective
temperature1. Validation calls the native forward-only actor inference path;
it never calls the optimizer. The reported Union/FKL/RKL means are weighted
by actual valid response tokens across batches. Truncation uses vLLM's actual
finish reason, with exact prompt-ID coverage. These are held-out distillation
metrics, not benchmark accuracy or a fixed-trajectory KL measurement.

Training logs use `response_length/actual_cap` for the current curriculum
limit. `response_length/clip_ratio` (and its non-aborted counterpart) measures
the fraction reaching that cap, instead of the padded 16K tensor width.
`response_length/truncation_rate` uses the actual `finish_reason == "length"`;
a response that naturally stops exactly at the cap counts only in clip_ratio.
This reporting correction is enabled only for runs using the paper data contract.

When `quant_lr` is set, `actor/lr` and `actor/lr_quant` both report the rates
used by that optimizer update, before the scheduler advances. The scheduler
still advances once with the same state and next-step rates. Configurations
without `quant_lr` retain native veRL's existing LR logging convention.

The exact-candidate teacher producer supports the reviewed Qwen3-1.7B, 4B and
8B architectures (`REVIEWED_TEACHERS` in `union_teacher_vllm.py`, keyed by
layer count and hidden size) with explicit source hashes. GPU memory utilization is an
explicit CLI setting; serving on another host still requires authenticated,
encrypted transport and longest-context acceptance.
