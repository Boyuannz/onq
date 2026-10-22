"""Prepare the exact-candidate teacher and probe its real scoring contract."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

from OnQ.integrations.verl_v080.utils.union_protocol import MAX_INPUT_TOKENS, MAX_MODEL_LEN


def snapshot_hashes(source):
    if not re.fullmatch('[0-9a-f]{40}', source.name):
        raise ValueError('Use a dense HF snapshot directory named by its immutable 40-hex revision')
    config = json.loads((source / 'config.json').read_text())
    if (config.get('model_type') != 'qwen3' or config.get('vocab_size') != 151936
            or (config.get('num_hidden_layers'), config.get('hidden_size'))
            not in {(28, 2048), (36, 2560), (36, 4096)}
            or any(config.get(k) for k in ('quantization_config', 'qat_adapter', 'qat_adapter_json'))):
        raise ValueError('Use an original dense Qwen3-1.7B, 4B or 8B snapshot')
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json'}
    index = source / 'model.safetensors.index.json'
    if index.is_file():
        weight_map = json.loads(index.read_text())['weight_map']
        if not weight_map:
            raise ValueError('Empty weight map')
        required |= {'model.safetensors.index.json', *weight_map.values()}
    else:
        required.add('model.safetensors')
    for name in required:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not (source / relative).is_file():
            raise ValueError(f'Missing or invalid snapshot file: {name}')
    # Hash all actual snapshot files, including optional template/config files.
    hashes = {}
    for path in sorted(source.rglob('*')):
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8 << 20), b''):
                    h.update(chunk)
            hashes[str(path.relative_to(source))] = h.hexdigest()
    return hashes


def prepare(source, root, port):
    if not 1 <= port <= 65535:
        raise ValueError('Port must be in 1..65535')
    if root.exists():
        raise ValueError('Choose a new service directory')
    hashes = snapshot_hashes(source)
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    key = root / 'teacher.key'
    # Create with private permissions before writing any secret.
    import os
    with os.fdopen(os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        f.write(secrets.token_urlsafe(32) + '\n')
    service = dict(source=str(source), port=port, source_sha256=hashes)
    (root / 'teacher_spec.json').write_text(json.dumps(service, indent=2) + '\n')
    client = dict(url=f'http://127.0.0.1:{port}', token_file=str(key), revision=source.name,
                  timeout=1800, teacher_topk=20, student_topk=20)
    (root / 'union_teacher.json').write_text(json.dumps(client, indent=2) + '\n')
    return dict(root=str(root), files_hashed=len(hashes), revision=source.name)


def validate_reply(reply, payload, revision):
    sha = hashlib.sha256(json.dumps(payload, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
    if reply.get('request_sha256') != sha or reply.get('model_revision') != revision or reply.get('temperature') != 1:
        raise ValueError('Teacher reply identity mismatch')
    ids, scores = reply.get('ids'), reply.get('logprobs')
    if not isinstance(ids, list) or not isinstance(scores, list) or len(ids) != len(payload['positions']) or len(scores) != len(ids):
        raise ValueError('Teacher reply position coverage mismatch')
    for row, lp, candidates in zip(ids, scores, payload['student_ids']):
        if len(row) != 40 or len(lp) != 41:
            raise ValueError('Expected 40 union slots plus one residual probability')
        if any(type(i) is not int or not -1 <= i < 151936 for i in row):
            raise ValueError('Invalid union token ID')
        active = [i for i in row if i >= 0]
        if len(set(active)) != len(active) or not set(candidates) <= set(active):
            raise ValueError('Teacher lost exact student candidates')
        if any(type(x) not in (int, float) or not math.isfinite(x) or x > 0 for x in lp):
            raise ValueError('Invalid normalized log probabilities')
        mass = sum(math.exp(x) for i, x in zip(row, lp[:-1]) if i >= 0) + math.exp(lp[-1])
        if abs(mass - 1) > 1e-5:
            raise ValueError('Teacher probabilities do not sum to one')
    return True


def probe(root, tokens=64, wait_seconds=0, timeout=1800):
    if not 2 <= tokens <= MAX_INPUT_TOKENS:
        raise ValueError('Probe token length outside teacher capacity')
    spec = json.loads((root / 'union_teacher.json').read_text())
    expected = json.loads((root / 'teacher_spec.json').read_text())
    url = spec['url'].rstrip('/')
    parsed = urllib.parse.urlparse(url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password
            or parsed.path or parsed.query or parsed.fragment
            or (parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'))):
        raise ValueError('Use HTTPS or an SSH loopback tunnel')
    context = ssl.create_default_context(cafile=spec.get('ca_file')) if parsed.scheme == 'https' else None
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            with urllib.request.urlopen(url + '/metadata', timeout=10, context=context) as response:
                metadata = json.load(response)
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(2, max(0, deadline - time.monotonic())))
    if (metadata.get('revision') != spec['revision'] or metadata.get('temperature') != 1
            or metadata.get('dtype') != 'bfloat16' or metadata.get('backend') != 'vllm_0.12.0'
            or metadata.get('vocab_size') != 151936
            or metadata.get('source_sha256') != expected['source_sha256']
            or metadata.get('max_input_tokens', 0) < tokens
            or metadata.get('max_model_len', 0) < tokens + 1):
        raise ValueError('Teacher metadata, source identity or context mismatch')
    payload = dict(input_ids=[100] * tokens, positions=[0, tokens - 2] if tokens > 2 else [0],
                   student_ids=[list(range(20))] * (2 if tokens > 2 else 1))
    token = Path(spec['token_file']).read_text().strip()
    request = urllib.request.Request(url + '/score', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        reply = json.load(response)
    validate_reply(reply, payload, spec['revision'])
    return dict(status='PASS', revision=spec['revision'], input_tokens=tokens,
                source_files=len(expected['source_sha256']), positions=len(payload['positions']),
                max_model_len=metadata['max_model_len'], scoring_contract='exact union + rest')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    init = sub.add_parser('prepare')
    init.add_argument('--model', type=Path, required=True)
    init.add_argument('--root', type=Path, required=True)
    init.add_argument('--port', type=int, default=8297)
    check = sub.add_parser('probe')
    check.add_argument('--root', type=Path, required=True)
    check.add_argument('--tokens', type=int, default=64)
    check.add_argument('--wait-seconds', type=float, default=0)
    check.add_argument('--timeout', type=float, default=1800)
    a = p.parse_args()
    root = a.root.expanduser().absolute()
    if a.action == 'prepare':
        result = prepare(a.model.expanduser().absolute(), root, a.port)
    else:
        if not math.isfinite(a.wait_seconds) or a.wait_seconds < 0 or not math.isfinite(a.timeout) or a.timeout <= 0:
            p.error('Wait must be finite and nonnegative; timeout must be finite and positive')
        result = probe(root, a.tokens, a.wait_seconds, a.timeout)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
