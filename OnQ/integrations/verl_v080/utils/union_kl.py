"""Exact union support on the current actor; detached teacher and alpha only."""

import errno
import hashlib
import http.client
import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint

from .union_transport import tls_options

logger = logging.getLogger(__name__)


def _retryable_connection_error(exc):
    # urllib wraps connection errors in URLError; response reads may raise the
    # underlying error directly. Never retry HTTP, TLS, timeout or parse errors.
    if isinstance(exc, urllib.error.HTTPError):
        return False
    cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(cause, (TimeoutError, ssl.SSLError)):
        return False
    return isinstance(cause, http.client.RemoteDisconnected) or (
        isinstance(cause, OSError)
        and cause.errno in (errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE)
    )


def request_teacher(input_ids, positions, student_ids, spec):
    payload = dict(input_ids=input_ids, positions=positions, student_ids=student_ids)
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    key = Path(spec["token_file"]).read_text().strip()
    request = urllib.request.Request(
        spec["url"].rstrip("/") + "/score",
        data=raw,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    request_sha = hashlib.sha256(raw).hexdigest()
    # Frozen-teacher scoring has no optimizer side effects. A lost response can
    # cause a duplicate teacher forward, but never an extra student update.
    # Reuse identical bytes/candidates/auth and close each failed reply before
    # retrying. Target validation stays outside this transport-only retry loop.
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=spec["timeout"], **tls_options(spec)) as reply:
                out = json.load(reply)
            break
        except (OSError, http.client.RemoteDisconnected) as exc:
            if not _retryable_connection_error(exc):
                raise
            retry = attempt < 3
            logger.warning(
                "Union teacher connection failure request_sha256=%s attempt=%d "
                "max_attempts=3 retry=%s error_type=%s",
                request_sha, attempt, retry, type(exc).__name__,
            )
            if not retry:
                raise
            time.sleep(0.25 * 2 ** (attempt - 1))
    assert (
        out["request_sha256"] == request_sha
    ), "Teacher input/candidate alignment changed"
    assert out["model_revision"] == spec["revision"] and out["temperature"] == 1
    assert out["positions"] == len(positions)
    return out


def student_distribution(z, ids, valid, indices=None, vocab_size=None):
    # Select inside the checkpoint: retaining the original logits plus another
    # response-token x vocabulary copy until backward is unnecessary.
    if indices is not None:
        z = z[indices, :vocab_size]
    z = z.float()
    z = z - z.amax(-1, keepdim=True).detach()
    norm = torch.logsumexp(z, -1, keepdim=True)
    safe = ids.clamp_min(0)
    # Write only True; all duplicate writes agree. Padding has its own sentinel
    # column, so it cannot erase or falsely select token0. Bool scatter_reduce
    # is unavailable on the pinned CUDA backend; scalar scatter is supported.
    selected = torch.zeros((len(z), z.shape[-1] + 1), dtype=torch.bool, device=z.device)
    selected.scatter_(-1, ids.masked_fill(~valid, z.shape[-1]), True)
    selected = selected[:, :-1]
    head = (z.gather(-1, safe) - norm).masked_fill(~valid, 0)
    rest = torch.logsumexp(z.masked_fill(selected, -torch.inf), -1, keepdim=True) - norm
    return torch.cat([head, rest], -1)


def category_loss(logits, ids, raw_log_p, *, epsilon, indices=None, vocab_size=None):
    n = len(logits) if indices is None else len(indices)
    assert (
        logits.ndim == 2
        and ids.shape == (n, 40)
        and raw_log_p.shape == (n, 41)
    )
    valid = ids >= 0
    categories = torch.cat([valid, torch.ones_like(valid[:, :1])], -1)
    assert ((valid.sum(-1) >= 20) & (valid.sum(-1) <= 40)).all()
    safe = ids.clamp_min(0)
    assert (safe < (logits.shape[-1] if vocab_size is None else vocab_size)).all()
    sorted_ids = ids.masked_fill(~valid, logits.shape[-1]).sort(-1).values
    assert not (
        (sorted_ids[:, 1:] == sorted_ids[:, :-1])
        & (sorted_ids[:, 1:] < logits.shape[-1])
    ).any()
    with torch.no_grad():
        assert torch.isfinite(raw_log_p).all() and not (raw_log_p[categories] > 0).any()
        raw = raw_log_p.double().exp().masked_fill(~categories, 0)
        total = raw.sum(-1, keepdim=True)
        assert (
            (total - 1).abs() < 1e-5
        ).all(), "Teacher coarse probability mass invalid"
        p = (raw + epsilon * categories) / (
            total + epsilon * categories.sum(-1, keepdim=True)
        )
        correction = (p - raw).abs().sum(-1).float()
        log_p = p.masked_fill(~categories, 1).log().float()
    if torch.is_grad_enabled() and logits.requires_grad:
        log_q = checkpoint(
            student_distribution, logits, ids, valid, indices, vocab_size,
            use_reentrant=False
        )
    else:
        log_q = student_distribution(logits, ids, valid, indices, vocab_size)
    delta = (log_p - log_q).masked_fill(~categories, 0)
    alpha = torch.sigmoid(delta).detach()
    p = log_p.exp().masked_fill(~categories, 0)
    q = log_q.exp().masked_fill(~categories, 0)
    forward = p * delta
    reverse = -q * delta
    wf = (alpha * forward).sum(-1)
    wr = ((1 - alpha) * reverse).sum(-1)
    loss = wf + wr
    assert torch.isfinite(loss).all() and torch.isfinite(log_q).all()
    return dict(
        distillation_losses=loss,
        dynamic_weighted_fkl=wf.detach(),
        dynamic_weighted_rkl=wr.detach(),
        dynamic_fkl=forward.sum(-1).detach(),
        dynamic_rkl=reverse.sum(-1).detach(),
        dynamic_alpha_mean=(alpha * categories).sum(-1) / categories.sum(-1),
        dynamic_alpha_rest=alpha[:, -1],
        teacher_mass=p[:, :-1].sum(-1),
        student_mass=q[:, :-1].sum(-1).detach(),
        dynamic_student_rest=q[:, -1].detach(),
        dynamic_teacher_correction_l1=correction,
        dynamic_teacher_entropy_union=-(p * log_p).sum(-1),
        dynamic_student_entropy_union=-(q * log_q).sum(-1).detach(),
        dynamic_union_categories=categories.sum(-1).float(),
        dynamic_student_only_count=valid.sum(-1).float() - 20,
    )


def union_token_losses(logits, data, mask, *, vocab_size, options, spec):
    assert spec["teacher_topk"] == spec["student_topk"] == 20
    assert logits.ndim == 2 and logits.shape[0] == data["input_ids"].values().shape[0]
    assert logits.shape[-1] >= vocab_size and mask.shape == logits.shape[:1]
    assert data["input_ids"].is_nested
    # Every scoring request uses top20 from these very logits, with no intervening
    # optimizer step. No rollout logprobs or stale actor candidate set is used.
    flat_ids = torch.full((len(logits), 40), -1, dtype=torch.long, device=logits.device)
    flat_lp = torch.zeros((len(logits), 41), dtype=torch.float32, device=logits.device)
    offsets = data["input_ids"].offsets().tolist()
    input_ids = data["input_ids"].values()
    for start, end in zip(offsets[:-1], offsets[1:]):
        pos = mask[start:end].nonzero().flatten()
        if not len(pos):
            continue
        assert int(pos.max()) < end - start - 1
        # topk ranking is invariant to full-vocabulary softmax normalization.
        chosen = torch.cat(
            [
                logits[start:end].detach()[positions, :vocab_size].topk(20, -1).indices
                for positions in pos.split(128)
            ],
            0,
        )
        requested = chosen.cpu().tolist()
        reply = request_teacher(
            input_ids[start:end].tolist(), pos.tolist(), requested, spec
        )
        ids = torch.tensor(reply["ids"], device=logits.device, dtype=torch.long)
        lp = torch.tensor(reply["logprobs"], device=logits.device, dtype=torch.float32)
        assert ids.shape == (len(pos), 40) and lp.shape == (len(pos), 41)
        # Prove every student candidate returned exactly once; the remaining
        # distinct candidates are the independently scored teacher top20.
        student = chosen
        assert ((ids[:, :, None] == student[:, None, :]).sum(1) == 1).all()
        flat_ids.index_copy_(0, pos + start, ids)
        flat_lp.index_copy_(0, pos + start, lp)
    fields = None
    pieces = {}
    for i, z in enumerate(logits.split(options["chunk_size"])):
        start = i * options["chunk_size"]
        active = mask[start : start + len(z)]
        indices = active.nonzero().flatten()
        zero = z[:, :0].float().sum(-1)
        if len(indices):
            result = category_loss(
                z,
                flat_ids[start : start + len(z)][indices],
                flat_lp[start : start + len(z)][indices],
                epsilon=options["epsilon"],
                indices=indices,
                vocab_size=vocab_size,
            )
            fields = tuple(result)
        else:
            # Fixed schema avoids needing a real token in any microbatch.
            result = {
                k: zero[:0]
                for k in (
                    "distillation_losses",
                    "dynamic_weighted_fkl",
                    "dynamic_weighted_rkl",
                    "dynamic_fkl",
                    "dynamic_rkl",
                    "dynamic_alpha_mean",
                    "dynamic_alpha_rest",
                    "teacher_mass",
                    "student_mass",
                    "dynamic_student_rest",
                    "dynamic_teacher_correction_l1",
                    "dynamic_teacher_entropy_union",
                    "dynamic_student_entropy_union",
                    "dynamic_union_categories",
                    "dynamic_student_only_count",
                )
            }
        for name, value in result.items():
            base = zero if name == "distillation_losses" else zero.detach()
            pieces.setdefault(name, []).append(
                base.index_copy(0, indices, value) if len(indices) else base
            )
    return {k: torch.cat(v) for k, v in pieces.items()}
