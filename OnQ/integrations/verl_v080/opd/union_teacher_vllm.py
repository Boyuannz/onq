"""Same-version vLLM teacher with exact union retrieval on prompt positions."""

import argparse, asyncio, hashlib, ipaddress, json, os, time, uuid
from pathlib import Path
import torch
from contextlib import aclosing, asynccontextmanager

from ..utils.union_batching import AsyncTeacherAdmission, TeacherBatcher, TeacherOverloaded
from ..utils.union_protocol import MAX_INPUT_TOKENS, MAX_MODEL_LEN


# Reviewed dense Qwen3 teachers, keyed by (num_hidden_layers, hidden_size).
REVIEWED_TEACHERS = {
    (28, 2048): 'Qwen3-1.7B',
    (36, 2560): 'Qwen3-4B',
    (36, 4096): 'Qwen3-8B',
}


def teacher_model_name(config):
    """Metadata label for a reviewed Qwen3 teacher config; reject anything else."""
    if config.get('model_type') != 'qwen3' or config.get('vocab_size') != 151936:
        raise ValueError('Union teacher must be a Qwen3 model with the 151936-token vocabulary')
    name = REVIEWED_TEACHERS.get((config.get('num_hidden_layers'), config.get('hidden_size')))
    if name is None:
        raise ValueError('Unreviewed teacher architecture')
    return name


def validate_bind(host, certfile, keyfile):
    """Loopback (SSH tunnel), or one specific private address that must use TLS."""
    address = ipaddress.ip_address(host)
    if bool(certfile) != bool(keyfile):
        raise ValueError("TLS needs both a certificate and a key file")
    if not address.is_loopback and (
        not certfile or not address.is_private or address.is_unspecified
    ):
        raise ValueError("A non-loopback teacher must bind a specific private address with TLS")


class UnionScorer:
    """Invoked by exactly one inference thread; identity is per sequence."""

    def __init__(self, root, llm, sampling_params, revision):
        self.root, self.llm = root, llm
        self.sampling_params, self.revision = sampling_params, revision
        self.count = 0

    def prepare(self, item, query):
        payload, sha = item['payload'], item['sha']
        file = self.root / 'requests' / f'{query}.json'
        file.write_text(json.dumps({**payload, 'request_sha256': sha}, separators=(',', ':')))
        # prompt_logprobs only enables the hooked prompt-scoring path (non-None also
        # skips prefix-cache reads). The hook reads full-vocabulary logprobs before
        # vLLM's top-k gather; 0 avoids materializing top-k rows nobody reads.
        params = self.sampling_params(max_tokens=1, temperature=1.0,
            prompt_logprobs=0, detokenize=False, extra_args={'union_query': query})
        return file, params

    def collect(self, item, query, output):
        payload, sha = item['payload'], item['sha']
        assert output is not None and output.finished, 'Teacher returned no finished output'
        assert output.prompt_token_ids == payload['input_ids'], 'Teacher prompt identity mismatch'
        chunks = sorted((self.root / 'responses').glob(query + '_*.pt'))
        assert chunks, 'Teacher returned no union scores'
        rows = []
        for path in chunks:
            result = torch.load(path, map_location='cpu', weights_only=True)
            assert result['request_sha256'] == sha, 'Teacher mixed request identities'
            rows += list(zip(result['positions'], result['ids'].tolist(), result['logprobs'].tolist()))
        rows.sort(key=lambda row: row[0])
        assert [r[0] for r in rows] == payload['positions'], 'Teacher position coverage mismatch'
        reply = dict(request_sha256=sha, ids=[r[1] for r in rows],
            logprobs=[r[2] for r in rows], model_revision=self.revision,
            temperature=1, peak_gib=None)
        return reply, chunks

    def __call__(self, items):
        started = time.monotonic()
        batch_id = uuid.uuid4().hex
        queries, params = [], []
        for item in items:
            query = uuid.uuid4().hex
            file, param = self.prepare(item, query)
            queries.append((query, file))
            params.append(param)
        forward_start = time.monotonic()
        outputs = self.llm.generate(
            [{'prompt_token_ids': item['payload']['input_ids']} for item in items],
            params, use_tqdm=False)
        forward_seconds = time.monotonic() - forward_start
        assert len(outputs) == len(items), 'Teacher batch lost a sequence'
        replies, scratch = [], []
        for item, output, (query, file) in zip(items, outputs, queries):
            reply, chunks = self.collect(item, query, output)
            replies.append(reply)
            scratch.extend([file, *chunks])
        # Validate every result before resolving any client or releasing scratch.
        service_seconds = time.monotonic() - started
        with (self.root / 'logs/teacher_vllm_batches.jsonl').open('a') as log:
            log.write(json.dumps(dict(batch_id=batch_id, batch_size=len(items),
                input_tokens=sum(len(x['payload']['input_ids']) for x in items),
                service_seconds=service_seconds, forward_seconds=forward_seconds,
                time=time.time())) + '\n')
        with (self.root / 'logs/teacher_vllm_requests.jsonl').open('a') as log:
            for item, reply in zip(items, replies):
                self.count += 1
                info = dict(request=self.count, positions=len(item['payload']['positions']),
                    input_tokens=len(item['payload']['input_ids']), batch_id=batch_id,
                    batch_size=len(items), seconds=service_seconds / len(items),
                    seconds_semantics='amortized_batch_service',
                    queue_wait_seconds=started - item['enqueued'], time=time.time())
                log.write(json.dumps(dict(**info, request_sha256=item['sha'])) + '\n')
                reply.update(info)
        # Only our successful transient scratch, never a native checkpoint.
        # An exception above preserves all batch inputs and partial outputs.
        for path in scratch:
            path.unlink()
        return replies


class AsyncUnionScorer(UnionScorer):
    """One AsyncLLM stream per query; no batch-wide completion barrier."""

    async def __call__(self, item):
        started = time.monotonic()
        query = uuid.uuid4().hex
        # File/tensor decoding must not block AsyncLLM's output event loop.
        # Failed/cancelled requests retain their own scratch for investigation.
        file, params = await asyncio.to_thread(self.prepare, item, query)
        forward_start = time.monotonic()
        output = None
        async with aclosing(self.llm.generate(
            {'prompt_token_ids': item['payload']['input_ids']}, params, request_id=query
        )) as stream:
            async for output in stream:
                pass
        forward_seconds = time.monotonic() - forward_start
        reply, chunks = await asyncio.to_thread(self.collect, item, query, output)
        self.count += 1
        info = dict(request=self.count, query_id=query,
            positions=len(item['payload']['positions']),
            input_tokens=len(item['payload']['input_ids']),
            seconds=time.monotonic() - started,
            seconds_semantics='async_request_wall', forward_seconds=forward_seconds,
            queue_wait_seconds=started - item['enqueued'], time=time.time())
        # Per-request wall times overlap: these are not GPU batch durations.
        with (self.root / 'logs/teacher_vllm_requests.jsonl').open('a') as log:
            log.write(json.dumps(dict(**info, request_sha256=item['sha'])) + '\n')
        reply.update(info)
        for path in [file, *chunks]:
            path.unlink()
        return reply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--max-batch-size", type=int, default=1)
    p.add_argument("--batch-token-budget", type=int, default=32768)
    p.add_argument("--batch-wait-ms", type=float, default=2.0)
    p.add_argument("--max-pending", type=int, default=32)
    p.add_argument("--engine", choices=("sync", "async"), default="async",
                   help="Default to official AsyncLLM; use sync for explicit synchronous replay")
    p.add_argument("--max-num-batched-tokens", type=int, default=32768,
                   help="vLLM scheduled tokens per iteration; separate from input admission")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--host", default="127.0.0.1",
                   help="Loopback, or a specific private address served over TLS")
    p.add_argument("--ssl-certfile")
    p.add_argument("--ssl-keyfile")
    a = p.parse_args()
    try:
        validate_bind(a.host, a.ssl_certfile, a.ssl_keyfile)
    except ValueError as exc:
        p.error(str(exc))
    from fastapi import FastAPI, HTTPException, Request, Response
    import uvicorn
    root = Path(a.root).resolve()
    if a.batch_token_budget < MAX_INPUT_TOKENS:
        p.error("Batch token budget must admit the full single-request context")
    if a.max_num_batched_tokens < 1 or not 0 < a.gpu_memory_utilization < 1:
        p.error("Invalid vLLM token or memory budget")
    # Validate admission before loading GPU weights.
    AsyncTeacherAdmission(None, max_batch=a.max_batch_size,
        max_tokens=a.batch_token_budget, max_pending=a.max_pending)
    if not 0 <= a.batch_wait_ms <= 100:
        p.error("Invalid batch wait")
    spec = json.loads((root / "teacher_spec.json").read_text())
    source = spec["source"]
    key = (root / "teacher.key").read_text().strip()
    config = json.loads((Path(source) / "config.json").read_text())
    try:
        model_name = teacher_model_name(config)
    except ValueError as exc:
        p.error(str(exc))
    assert spec["source_sha256"], "Teacher identity hashes are required"
    assert key, "Empty teacher token file"
    for folder in ["requests", "responses", "logs"]:
        (root / folder).mkdir(exist_ok=True, mode=0o700)
    os.environ["QAT_UNION_SCORE_ROOT"] = str(root)
    from vllm import SamplingParams
    import vllm

    assert vllm.__version__ == "0.12.0"
    for name, expected in spec["source_sha256"].items():
        h = hashlib.sha256()
        with (Path(source) / name).open("rb") as f:
            for c in iter(lambda: f.read(8 << 20), b""):
                h.update(c)
        assert h.hexdigest() == expected, name
    engine_options = dict(
        model=source,
        tokenizer=source,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=a.max_num_batched_tokens,
        max_num_seqs=256,
        gpu_memory_utilization=a.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_logprobs=20,
        worker_extension_cls="OnQ.integrations.verl_v080.opd.union_vllm_worker.UnionWorkerExtension",
        seed=0,
        trust_remote_code=False,
        disable_log_stats=True,
    )

    @asynccontextmanager
    async def lifespan(app):
        if a.engine == "async":
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
            llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_options))
            scorer = AsyncUnionScorer(root, llm, SamplingParams, Path(source).name)
            admission = AsyncTeacherAdmission(scorer, max_batch=a.max_batch_size,
                max_tokens=a.batch_token_budget, max_pending=a.max_pending)
        else:
            from vllm import LLM
            llm = LLM(**engine_options)
            scorer = UnionScorer(root, llm, SamplingParams, Path(source).name)
            admission = TeacherBatcher(scorer, max_batch=a.max_batch_size,
                max_tokens=a.batch_token_budget, wait_ms=a.batch_wait_ms, max_pending=a.max_pending)
        app.state.scorer, app.state.admission = scorer, admission
        try:
            yield
        finally:
            await admission.close()
            if a.engine == "async":
                llm.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/metadata")
    def metadata():
        return dict(
            model=model_name,
            revision=Path(source).name,
            temperature=1,
            dtype="bfloat16",
            backend="vllm_0.12.0",
            vocab_size=151936,
            max_input_tokens=MAX_INPUT_TOKENS,
            max_model_len=MAX_MODEL_LEN,
            source_sha256=spec["source_sha256"],
            requests=app.state.scorer.count,
            engine=a.engine,
            max_batch_size=a.max_batch_size,
            batch_token_budget=a.batch_token_budget,
            batch_wait_ms=a.batch_wait_ms if a.engine == "sync" else None,
            max_num_batched_tokens=a.max_num_batched_tokens,
            gpu_memory_utilization=a.gpu_memory_utilization,
            max_pending=a.max_pending,
            pid=os.getpid(),
        )

    @app.post("/score")
    async def score(request: Request):
        if request.headers.get("authorization") != "Bearer " + key:
            raise HTTPException(401)
        payload = await request.json()
        try:
            ids = payload["input_ids"]
            positions = payload["positions"]
            candidates = payload["student_ids"]
            v = 151936
            assert 2 <= len(ids) <= MAX_INPUT_TOKENS and all(
                type(i) == int and 0 <= i < v for i in ids
            )
            assert len(positions) == len(candidates) and 0 < len(positions) < len(ids)
            assert len(set(positions)) == len(positions) and positions == sorted(
                positions
            )
            assert all(type(i) == int and 0 <= i < len(ids) - 1 for i in positions)
            assert all(
                len(row) == 20
                and len(set(row)) == 20
                and all(type(i) == int and 0 <= i < v for i in row)
                for row in candidates
            )
        except (KeyError, AssertionError, TypeError):
            raise HTTPException(400, "Invalid exact-candidate request")
        sha = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        try:
            result = await app.state.admission.submit(
                dict(payload=payload, sha=sha, enqueued=time.monotonic()), tokens=len(ids)
            )
            # Our validated reply is already plain Python JSON data. Returning
            # a dict makes FastAPI recursively walk every token/category again
            # in jsonable_encoder before the JSON serializer. Encode only once.
            return Response(content=json.dumps(result, separators=(",", ":"), allow_nan=False),
                            media_type="application/json")
        except TeacherOverloaded as exc:
            raise HTTPException(503, str(exc)) from exc

    (root / "teacher_vllm.pid").write_text(str(os.getpid()))
    uvicorn.run(app, host=a.host, port=spec["port"], log_level="warning",
                ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile)


if __name__ == "__main__":
    main()
