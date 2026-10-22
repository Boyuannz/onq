"""Exact union probability capture; kept independent of CUDA/vLLM imports."""

import json
import os
import re
from pathlib import Path

import torch


def record_union_chunk(request, start_idx, logprobs):
    extra = request.sampling_params.extra_args or {}
    query = extra.get("union_query")
    if query is None:
        return
    assert re.fullmatch("[0-9a-f]{32}", query)
    root = Path(os.environ["QAT_UNION_SCORE_ROOT"])
    payload = json.loads((root / "requests" / f"{query}.json").read_text())
    assert (
        payload["input_ids"] == request.prompt_token_ids
    ), "Position scoring request mismatch"
    relevant = [
        i
        for i, p in enumerate(payload["positions"])
        if start_idx <= p < start_idx + len(logprobs)
    ]
    if not relevant:
        return
    positions = [payload["positions"][i] for i in relevant]
    # Work in small vocabulary chunks, preserving vLLM's actual normalized
    # probabilities. Do not rescale logits or insert a missing-probability floor.
    pieces = []
    for start in range(0, len(relevant), 128):
        which = relevant[start : start + 128]
        rows = torch.tensor(
            [payload["positions"][i] - start_idx for i in which], device=logprobs.device
        )
        lp = logprobs.index_select(0, rows).float()
        teacher = lp.topk(20, -1).indices
        student = torch.tensor(
            [payload["student_ids"][i] for i in which],
            device=lp.device,
            dtype=torch.long,
        )
        ids = torch.cat([teacher, student], -1).sort(-1).values
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[:, 1:] = ids[:, 1:] != ids[:, :-1]
        selected = torch.zeros((len(lp), lp.shape[-1] + 1), dtype=torch.bool, device=lp.device)
        selected.scatter_(-1, ids.masked_fill(~valid, lp.shape[-1]), True)
        selected = selected[:, :-1]
        head = lp.gather(-1, ids).masked_fill(~valid, 0)
        rest = torch.logsumexp(lp.masked_fill(selected, -torch.inf), -1, keepdim=True)
        values = torch.cat([head, rest], -1)
        assert torch.isfinite(values).all()
        pieces.append((ids.masked_fill(~valid, -1).cpu(), values.cpu()))
    value = dict(
        positions=positions,
        ids=torch.cat([p[0] for p in pieces]),
        logprobs=torch.cat([p[1] for p in pieces]),
        request_sha256=payload["request_sha256"],
    )
    target = root / "responses" / f"{query}_{start_idx:08d}.pt"
    tmp = target.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(target)
