"""Sample source records with a chosen seed and write calibration token windows and OPD prompt pools.

Seeded random selection of OPD prompts and Block-AP calibration windows from pinned sources.

For each recipe source: draw a uniform random permutation of ALL eligible source rows
(seeded per source), scan it in order, reject records that fail a rule, and accept
until the quota is met (in-order refill). Nothing is ranked by difficulty, length,
teacher score or answer correctness. A different ``--seed`` gives a different,
equally valid sample; the same seed and recipe reproduce the same data.

    python -m OnQ preparedata benchmarks --aime90 AIME_DATA --out BENCH/HELDOUT_QUERIES.jsonl
    python -m OnQ preparedata build --stage all --seed 42 --out OUT         --benchmarks BENCH/HELDOUT_QUERIES.jsonl --cache CACHE

The run fails (exit 1, status FAIL) if the independent decontamination/duplicate
gate (check_data_overlap.py) finds any hit in the written files."""

import argparse
import collections
import copy
import gzip
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

from OnQ.scripts.data import check_data_overlap as decontam
from OnQ.scripts.data import format_conversations as render
from OnQ.scripts.data import selection_rules as rules
from OnQ.scripts.data.dataset_sources import (Cache, CodeOriginals, ParquetSource, digest, file_sha256, random_order,
                     source_seed)

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')


class Exhausted(RuntimeError):
    pass


def draw(order, fetch, evaluate, target, batch=64, prepare=None, log=None, label=None):
    """Scan ``order`` and accept evaluated items until ``target``.

    ``fetch(coords) -> {coord: (row, provenance)}``; ``evaluate(row, provenance)``
    returns ``(reason, None)`` to reject or ``(None, item)`` to accept, and is called
    strictly in order so stateful deduplication sees every earlier acceptance.
    Batching changes I/O only, never membership.
    """
    accepted, rejected, scanned = [], collections.Counter(), 0
    while len(accepted) < target:
        coords = list(itertools.islice(order, batch))
        if not coords:
            raise Exhausted(f'source exhausted after {scanned} rows: {len(accepted)}/{target} accepted')
        rows, before = fetch(coords), scanned
        if prepare:
            prepare([rows[c][0] for c in coords])
        for c in coords:
            scanned += 1
            reason, item = evaluate(*rows[c])
            if reason:
                rejected[':'.join(reason.split(':')[:2]) if reason.startswith('test_overlap') else reason] += 1
                if log is not None:
                    log.append(dict(provenance=rows[c][1], reason=reason))
                continue
            accepted.append(item)
            if len(accepted) == target:
                break
        if label and scanned // 2048 != before // 2048:
            print('DRAW', label, f'{len(accepted)}/{target} scanned {scanned}', flush=True)
    return accepted, scanned, rejected


def write_jsonl(path, rows):
    """JSONL (gzip when ``.gz``) with a fixed gzip header, so equal rows give equal bytes."""
    data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    tmp.replace(path)
    return dict(file=path.name, rows=len(rows), sha256=file_sha256(path))


def record_id(name, prov):
    return f'{name}:{prov["file"]}:rg{prov["row_group"]}:{prov["row_in_group"]}'


class Selector:
    """Shared state for one stage: dedup indexes, decontamination and source access."""

    def __init__(self, args, tokenizer, bench, near_eval, conflicts):
        self.args, self.tok, self.bench, self.near_eval, self.conflicts = args, tokenizer, bench, near_eval, conflicts
        self.cache = Cache(args.cache, args.readonly_cache)
        self.originals = CodeOriginals(self.cache, args.workers)
        self.pools = {}
        self.seen, self.code_ids, self.near = set(), set(), rules.NearDuplicates()
        self.refs = {}

    def pool(self, spec):
        key = digest(spec)
        if key not in self.pools:
            self.pools[key] = ParquetSource(spec, self.cache, self.args.workers)
        return self.pools[key]

    def resolve(self, rows):
        todo = [r for r in rows if r.get('split') == 'train' and (r['dataset'], int(r['index'])) not in self.refs]
        self.refs.update(self.originals.resolve(todo))

    def identity(self, kind, row):
        """(reason, ref) for OCR2 split/identity checks; other kinds pass through."""
        if kind != 'code':
            return None, None
        if row['split'] != 'train':
            return 'source_not_train_split', None
        if (row['dataset'], str(row['index'])) in self.code_ids:
            return 'duplicate_question_identity', None
        return None, self.refs[row['dataset'], int(row['index'])]

    def duplicate(self, key, question):
        if key in self.seen:
            return 'duplicate_question'
        if self.near.match(question) is not None:
            return 'near_duplicate_question'
        return None

    def commit(self, kind, row, key, question):
        self.seen.add(key)
        self.near.add(question)
        if kind == 'code':
            self.code_ids.add((row['dataset'], str(row['index'])))

    def overlap(self, turns, texts, near_texts):
        """Benchmark exact/containment hit, else rare-n-gram paraphrase hit, as a reason."""
        hit = self.bench.check(turns, texts)
        if hit:
            return f'test_overlap:{hit["match"]}:{hit["benchmark"]}:{hit["benchmark_id"]}'
        for text in dict.fromkeys(near_texts):
            hit = self.near_eval.match(text)
            if hit:
                return f'test_overlap:{hit[0]}:{hit[1]}'
        return None

    def run_source(self, stage, source, target, evaluate, log):
        pool = self.pool(source['dataset'])
        cohort_spec = source.get('cohort')
        cohort, cohort_counts = pool.cohort(cohort_spec['column'], cohort_spec['values']) if cohort_spec else (None, {})
        size = pool.size if cohort is None else len(cohort)
        prefix = 'blockap:' if stage == 'blockap' else ''  # OnQ per-source seed derivation
        order = random_order(size, source_seed(self.args.seed, prefix + source['name']))

        def fetch(coords):
            if cohort is None:
                return pool.rows(coords)
            got = pool.rows([cohort[i] for i in coords])
            return {i: got[cohort[i]] for i in coords}
        started = time.time()
        accepted, scanned, rejected = draw(
            order, fetch, evaluate, target, self.args.batch,
            prepare=self.resolve if source['kind'] == 'code' else None, log=log, label=f'{stage}:{source["name"]}')
        stats = dict(target=target, accepted=len(accepted), scanned=scanned, rejections=dict(rejected),
                     source_rows=pool.size, eligible_rows=size, cohort_counts=cohort_counts,
                     dataset=source['dataset'], catalog_sha256=pool.catalog_sha256,
                     seconds=round(time.time() - started, 1))
        print('SOURCE', stage, source['name'], json.dumps({k: stats[k] for k in
              ('accepted', 'scanned', 'rejections', 'seconds')}), flush=True)
        return accepted, stats


# ---------------------------------------------------------------- OPD
def build_opd(sel, recipe, out):
    cfg = recipe['opd']
    train, val, raws, rejections, stats = [], [], [], [], {}
    for source in cfg['sources']:
        name, kind = source['name'], source['kind']
        n_train, n_val = source['train'], source['val']

        def evaluate(row, prov, name=name, kind=kind, source=source):
            if kind in ('chat', 'if') and not rules.single_turn(row['messages']):
                return 'not_single_turn_conversation', None
            reason, ref_env = sel.identity(kind, row)
            if reason:
                return reason, None
            ref = ref_env and ref_env['record']
            try:
                prompt, out_msgs = render.prompt_messages(kind, row, ref), render.output_prompt(kind, row, ref)
            except (ValueError, TypeError, KeyError) as exc:
                return f'unusable_record:{type(exc).__name__}', None
            text = render.text_of(prompt)
            key = decontam.normalize(text)
            question = render.code_statement(row['dataset'], ref) if kind == 'code' else text
            reason = (rules.admission(prompt) or rules.common_reason(kind, [m['content'] for m in prompt + out_msgs])
                      or rules.opd_reason(kind, row, prompt, ref, sel.conflicts, source.get('wildchat', False))
                      or sel.duplicate(key, question))
            if reason:
                return reason, None
            n_tokens = render.prompt_tokens(sel.tok, out_msgs)
            if n_tokens > cfg['max_prompt_tokens']:
                return f'prompt_over_{cfg["max_prompt_tokens"]}', None
            turns = [m['content'] for m in prompt + out_msgs if m['role'] == 'user']
            reason = sel.overlap(turns, [text, render.text_of(out_msgs)], [text, *turns])
            if reason:
                return reason, None
            sel.commit(kind, row, key, question)
            rid = record_id(name, prov)
            raw = dict(id=rid, source=name, kind=kind, provenance=prov, record=copy.deepcopy(row),
                       raw_sha256=digest(row))
            if ref_env:
                raw.update(referenced_record=copy.deepcopy(ref_env), reference_sha256=digest(ref_env['record']))
            item = dict(id=rid, source=name, domain=source['domain'], category=source.get('category', source['domain']),
                        source_repo=prov['repo'], revision=prov['revision'],
                        row_index=f'{prov["file"]}:rg{prov["row_group"]}:{prov["row_in_group"]}',
                        messages=out_msgs, prompt_sha256=digest(out_msgs), prompt_tokens=n_tokens,
                        extra=dict(source_kind=kind, raw_sha256=raw['raw_sha256'],
                                   selection_prompt_sha256=digest(prompt)))
            if out_msgs != prompt:
                item['extra']['rendering'] = 'codeforces_official_body_without_instructions'
            return None, (item, raw)

        log = []
        accepted, stats[name] = sel.run_source('opd', source, n_train + n_val, evaluate, log)
        stats[name].update(train=n_train, val=n_val)
        for i, (item, raw) in enumerate(accepted):  # first accepted -> train, the rest -> val
            item = dict(item, split='train' if i < n_train else 'val')
            (train if i < n_train else val).append(item)
            raws.append(raw)
        rejections += [dict(source=name, **r) for r in log]
    random.Random(sel.args.seed).shuffle(train)
    random.Random(sel.args.seed + 1).shuffle(val)
    out.mkdir(parents=True)
    files = dict(train=write_jsonl(out / 'opd_prompts.jsonl', train),
                 val=write_jsonl(out / 'opd_val_prompts.jsonl', val),
                 raw=write_jsonl(out / 'raw_records.jsonl.gz', raws),
                 rejections=write_jsonl(out / 'rejections.jsonl.gz', rejections))
    lengths = [r['prompt_tokens'] for r in train + val]
    return dict(stage='opd', rows=dict(train=len(train), val=len(val)), sources=stats, files=files,
                max_prompt_tokens=max(lengths), mean_prompt_tokens=round(sum(lengths) / len(lengths), 1),
                domains=dict(collections.Counter(r['domain'] for r in train)))


# ---------------------------------------------------------------- Block-AP
def build_blockap(sel, recipe, out):
    cfg = recipe['blockap']
    length = cfg['window_length']
    start_id = sel.tok.convert_tokens_to_ids('<|im_start|>')
    normalizer = sel.tok.backend_tokenizer.normalizer
    windows, raws, rejections, stats = [], [], [], {}
    for source in cfg['sources']:
        name, kind = source['name'], source['kind']

        def evaluate(row, prov, name=name, kind=kind, source=source):
            reason, ref_env = sel.identity(kind, row)
            if reason:
                return reason, None
            ref = ref_env and ref_env['record']
            try:
                prompt, dialogue = render.prompt_messages(kind, row, ref), render.output_dialogue(kind, row, ref)
                rendered = render.official_render(sel.tok, dialogue)
            except (ValueError, TypeError, KeyError) as exc:
                return f'unusable_record:{type(exc).__name__}', None
            if len(rendered) < length:  # a token has >= 1 character: cannot reach the window
                return f'length_below_{length}', None
            text = render.text_of(prompt)
            key = decontam.normalize(text)
            question = render.code_statement(row['dataset'], ref) if kind == 'code' else text
            reason = (rules.admission(prompt, dialogue)
                      or rules.common_reason(kind, [m['content'] for m in prompt + dialogue])
                      or sel.duplicate(key, question))
            if reason:
                return reason, None
            ids = sel.tok.encode(rendered, add_special_tokens=False)
            if len(ids) < length:
                return f'length_below_{length}', None
            try:
                audit = render.template_audit(dialogue, rendered)
            except ValueError:
                return 'official_template_text_loss', None
            expected = normalizer.normalize_str(rendered) if normalizer else rendered
            if sel.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != expected:
                raise ValueError(f'Tokenizer round trip changed text: {record_id(name, prov)}')
            turns = [m['content'] for m in prompt + dialogue if m['role'] == 'user']
            full = '\n'.join(m['content'] for m in dialogue)
            reason = sel.overlap(turns, [text, full], [text, *turns])
            if reason:
                return reason, None
            sel.commit(kind, row, key, question)
            rid = record_id(name, prov)
            raw = dict(id=rid, source=name, domain=source['domain'], kind=kind, provenance=prov,
                       record=copy.deepcopy(row), raw_sha256=digest(row), dialogue_sha256=digest(dialogue),
                       full_tokens=len(ids), used_head_tokens=length, template_audit=audit)
            if ref_env:
                raw.update(referenced_record=copy.deepcopy(ref_env), reference_sha256=digest(ref_env['record']))
            return None, (ids[:length], raw)

        target = source['windows']
        log = []
        accepted, stats[name] = sel.run_source('blockap', source, target, evaluate, log)
        for i, (ids, raw) in enumerate(accepted):
            if ids[0] != start_id or len(ids) != length:
                raise AssertionError('Window must be one full dialogue head starting at <|im_start|>')
            windows.append(dict(id=f'{name}:{i}', domain=source['domain'], input_ids=ids, segments=[dict(
                record_id=raw['id'], source_start=0, source_end=length, window_start=0, window_end=length)]))
            raws.append(raw)
        rejections += [dict(source=name, **r) for r in log]
    random.Random(sel.args.seed).shuffle(windows)
    out.mkdir(parents=True)
    files = dict(train=write_jsonl(out / 'train_tokens.jsonl.gz', windows),
                 raw=write_jsonl(out / 'raw_records.jsonl.gz', raws),
                 rejections=write_jsonl(out / 'rejections.jsonl.gz', rejections))
    return dict(stage='blockap', windows=len(windows), window_length=length, sources=stats, files=files,
                historical_reasoning_records=sum(bool(r['template_audit']['historical_reasoning']) for r in raws),
                empty_think_blocks=sum(r['template_audit']['empty_think_blocks'] for r in raws),
                domains=dict(collections.Counter(w['domain'] for w in windows)))


# ---------------------------------------------------------------- entrypoint
def load_tokenizer(recipe, override):
    from transformers import AutoTokenizer
    if override:
        path, pin = Path(override), dict(local=str(override))
    else:
        from huggingface_hub import snapshot_download
        pin = recipe['tokenizer']
        path = Path(snapshot_download(pin['repo'], revision=pin['revision'], allow_patterns=list(TOKENIZER_FILES)))
    hashes = {f: file_sha256(path / f) for f in TOKENIZER_FILES if (path / f).exists()}
    return AutoTokenizer.from_pretrained(path, local_files_only=True), dict(pin, sha256=hashes)


def environment():
    import langdetect  # noqa: F401
    import pyarrow
    import transformers
    try:
        rev = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', str(HERE)], cwd=HERE,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        rev, dirty = None, None
    return dict(python=platform.python_version(), host=platform.node(), transformers=transformers.__version__,
                pyarrow=pyarrow.__version__, git_revision=rev, git_dirty=dirty,
                code_sha256={p.name: file_sha256(p) for p in sorted(HERE.glob('*.py'))})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=('opd', 'blockap', 'all'), required=True)
    p.add_argument('--out', type=Path, required=True, help='new directory; never overwritten')
    p.add_argument('--benchmarks', type=Path, required=True, help='prepare_benchmark_texts.py output (JSONL)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--recipe', type=Path, default=HERE / 'sampling_recipe.json')
    p.add_argument('--cache', type=Path, default=Path(os.environ.get('HF_HOME', Path.home() / '.cache')) / 'qat_data')
    p.add_argument('--readonly-cache', type=Path, action='append', default=[],
                   help='existing cache roots searched first (never written)')
    p.add_argument('--tokenizer', help='local tokenizer directory instead of the recipe pin')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch', type=int, default=64, help='rows fetched per I/O batch (does not change data)')
    args = p.parse_args(argv)
    if args.out.exists():
        p.error(f'{args.out} exists; use a new output directory')
    recipe = json.loads(args.recipe.read_text())
    stages = ('opd', 'blockap') if args.stage == 'all' else (args.stage,)
    args.out.mkdir(parents=True)
    started = time.time()
    tokenizer, tok_pin = load_tokenizer(recipe, args.tokenizer)
    bench_rows = decontam.read_jsonl(args.benchmarks)
    bench = decontam.BenchmarkIndex(bench_rows)
    near_eval = decontam.load_near_matcher(bench_rows)
    conflicts = rules.ifevalg_conflicts(rules.fetch_ifevalg(args.cache).read_text())
    print('READY', len(bench_rows), 'benchmark texts', flush=True)
    build = dict(command=sys.argv if argv is None else ['prepare_training_data.py', *argv], seed=args.seed, stages=list(stages),
                 recipe=recipe, recipe_sha256=file_sha256(args.recipe), tokenizer=tok_pin,
                 benchmarks=dict(file=str(args.benchmarks), rows=len(bench_rows), sha256=file_sha256(args.benchmarks)),
                 ifevalg=dict(url=rules.IFEVALG_URL, sha256=rules.IFEVALG_SHA256), environment=environment())
    manifests = {}
    for stage in stages:  # each stage deduplicates independently (cross-stage overlap is allowed)
        sel = Selector(args, tokenizer, bench, near_eval, conflicts)
        builder = build_opd if stage == 'opd' else build_blockap
        manifests[stage] = builder(sel, recipe, args.out / stage)
    gates = decontam.gate(bench, args.out / 'opd' if 'opd' in stages else None,
                          args.out / 'blockap' if 'blockap' in stages else None)
    status = 'FAIL' if gates['status'] != 'PASS' else 'PASS'
    for stage, manifest in manifests.items():
        manifest.update(status=status, seed=args.seed, build='../BUILD.json', gates='../GATES.json')
        (args.out / stage / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n')
    (args.out / 'GATES.json').write_text(json.dumps(gates, indent=1, ensure_ascii=False) + '\n')
    build.update(status=status, seconds=round(time.time() - started), gates=dict(
        status=gates['status'], failures=gates['failures']))
    (args.out / 'BUILD.json').write_text(json.dumps(build, indent=1, ensure_ascii=False) + '\n')
    print('BUILD', status, json.dumps({s: m['files'] for s, m in manifests.items()}), flush=True)
    if status == 'FAIL':
        print('GATE FAILURES:', gates['failures'], file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
