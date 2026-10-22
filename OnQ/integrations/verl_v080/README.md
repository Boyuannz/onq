# OnQ native veRL integration

Use [the reproduction guide](../../docs/REPRODUCE.md) for data preparation,
calibration, teacher service, OPD, resume and export commands. The integration
uses the pinned external veRL revision in `OnQ/configs/upstreams/verl_v080.json`.

- `python -m OnQ opd prepare`: OnQ schedule and candidate-wise Dynamic KL configuration.
- `python -m OnQ opd serve`: exact-candidate teacher service.
- `python -m OnQ opd train`: training process entry point.
- `python -m OnQ opd resume`: native optimizer/data-state continuation.

The OnQ workflow uses the Union Dynamic KL objective described in
[the method notes](../../docs/METHOD.md).

### Rollout transfer and teacher connections

The reviewed subset of [veRL #7873](https://github.com/verl-project/verl/pull/7873)
removes full Python garbage collection from the colocated transfer/load path.
It drops receiver tensor views before bucket acknowledgements and releases the
IPC mapping before the final acknowledgement, so the sender can reclaim its
communication buffer. CUDA synchronization and allocator cleanup remain. The
backport retains this pin's single-argument receive callback; it is not a veRL
version upgrade and does not enable delta sync or asynchronous policy updates.

The default installer obtains the official veRL commit and automatically applies
this bundled patch. Its three modified files match the verified training sources
by SHA256. For a manually provisioned upstream checkout, apply it explicitly to a
separate checkout before launching a new run. Never patch a source tree used by
a running job. The standalone backport command defaults to dry-run,
requires the exact pin, rejects unknown edits and verifies every patched file;
repeating it on the exact reviewed patch is safe.

```bash
git clone --shared "$VERL" /path/to/isolated/verl-rollout
git -C /path/to/isolated/verl-rollout checkout 7aed6b230776f963fa09509c10d9c3a767d1102c
"$PY" -m OnQ.integrations.verl_v080.utils.backports --source /path/to/isolated/verl-rollout
"$PY" -m OnQ.integrations.verl_v080.utils.backports --source /path/to/isolated/verl-rollout --apply
export VERL=/path/to/isolated/verl-rollout
export PYTHONPATH="$VERL:$QAT"
```

External teacher clients reuse one lazy HTTP session per client/process/event
loop, without adding a concurrency cap or retry. Credentials are still read
for each request; TLS, timeout, request IDs, k1/top20 parsing and next-token
alignment are unchanged. No cookies persist. Serialization excludes sockets,
event loops and SSL contexts. The runtime selects a native AgentLoopManager
extension that closes worker-owned pools when training ends or fails; the
standalone teacher probe also closes its pool. A different custom manager is
rejected rather than silently overwritten. The effective manager selection is
saved in `runtime_transport_config.yaml`.

Shutdown is terminal for each client: it rejects new requests before any
asynchronous cleanup, including from late trajectories or serialized copies of
a closed client. Workers cancel and drain their tracked trajectory tasks before
closing pools. Concurrent shutdown calls share one shielded cleanup task; a
cancelled caller cannot cancel that cleanup. A trajectory that refuses to stop
within 20 seconds produces a cleanup error, with the pool closed and admission
still blocked. Create a new client/worker for a new lifetime.

These changes do not remove QAT's full-state export materialization. End-to-end
speedup depends on measured transfer and teacher-request time; no full-training
speedup is asserted by the transport checks.
