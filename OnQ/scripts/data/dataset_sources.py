"""Provide pinned dataset discovery, cached row access and seeded sampling; imported by preparation scripts.

Pinned Hugging Face sources: shard catalogs, uniform random order and lazy row access.

Every source is a (repo, revision, prefix, columns) spec. The revision is a full
commit SHA, so the same seed always reads the same bytes. Rows are addressed by a
*global coordinate* over all row groups of all shards (sorted by file name); the
random order is a permutation of those coordinates, independent of I/O batching.

Cache layout (compatible with the previous build caches, which can be
passed read-only through ``--readonly-cache``):
  catalogs/<key>.json                         shard/row-group sizes
  cohorts/<digest>.json.gz                    coordinates whose column == value
  columns/<key+columns>/<group>.bounded.parquet  projected row group, 512-row chunks
  columns/<key+columns>/<group>.parquet          legacy single-chunk form (converted on use)"""

import bisect
import collections
import concurrent.futures
import gzip
import hashlib
import itertools
import json
import os
import random
import time
from pathlib import Path

import pyarrow.parquet as pq

CHUNK_ROWS = 512


def digest(value):
    """SHA256 of canonical JSON (sorted keys, compact separators)."""
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def source_seed(seed, name):
    """Independent, stable per-source seed derived from the global seed."""
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'big')


def random_order(size, seed):
    """Lazy uniform permutation of range(size) (sparse Fisher-Yates, O(drawn) memory)."""
    rng, swaps, remaining = random.Random(seed), {}, size
    while remaining:
        pick = rng.randrange(remaining)
        value = swaps.get(pick, pick)
        remaining -= 1
        if pick != remaining:
            swaps[pick] = swaps.get(remaining, remaining)
        swaps.pop(remaining, None)
        yield value


def retry(fn, attempts=5):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # network/Hub errors: bounded exponential backoff
            if attempt == attempts - 1:
                raise
            delay = 15 * 2 ** attempt
            print(f'SOURCE_RETRY {type(exc).__name__} in {delay}s', flush=True)
            time.sleep(delay)


class Cache:
    """One writable cache root plus optional read-only roots searched first."""

    def __init__(self, root, readonly=()):
        self.root = Path(root)
        self.readonly = [Path(p) for p in readonly]

    def find(self, rel):
        for base in [*self.readonly, self.root]:
            if (base / rel).exists():
                return base / rel
        return None

    def target(self, rel):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _tmp(path):
    return path.with_name(f'{path.name}.{os.getpid()}.tmp')


def _hub_open(spec, file):
    from huggingface_hub import HfFileSystem
    return HfFileSystem().open(f'datasets/{spec["repo"]}@{spec["revision"]}/{file}', 'rb',
                               block_size=1 << 20)


class ParquetSource:
    """All parquet shards under ``prefix`` of a pinned dataset revision."""

    def __init__(self, spec, cache, workers=4):
        if len(spec['revision']) != 40:
            raise ValueError(f'Pin a full commit SHA for {spec["repo"]}')
        self.spec, self.cache, self.workers = spec, cache, workers
        key = digest({k: spec[k] for k in ('repo', 'revision', 'prefix')})[:20]
        self.columns_dir = f'columns/{digest([key, spec["columns"]])[:20]}'
        rel = f'catalogs/{key}.json'
        found = cache.find(rel)
        if found:
            self.catalog = json.loads(found.read_text())
            self.catalog_sha256 = file_sha256(found)
        else:
            self.catalog = self._build_catalog()
            path = cache.target(rel)
            _tmp(path).write_text(json.dumps(self.catalog, indent=1))
            _tmp(path).replace(path)
            self.catalog_sha256 = file_sha256(path)
        self.groups, self.ends, offset = [], [], 0
        for shard in self.catalog['shards']:
            missing = set(spec['columns']) - set(shard['fields'])
            if missing:
                raise ValueError(f'{spec["repo"]}:{shard["file"]} lacks columns {sorted(missing)}')
            for g, n in enumerate(shard['groups']):
                self.groups.append(dict(file=shard['file'], group=g, start=offset, rows=n))
                offset += n
                self.ends.append(offset)
        self.size = offset

    def _build_catalog(self):
        from huggingface_hub import HfApi
        spec = self.spec
        info = retry(lambda: HfApi().dataset_info(spec['repo'], revision=spec['revision']))
        if info.sha != spec['revision']:
            raise ValueError(f'Pinned revision mismatch for {spec["repo"]}')
        files = sorted(s.rfilename for s in info.siblings
                       if s.rfilename.startswith(spec['prefix']) and s.rfilename.endswith('.parquet'))
        if not files:
            raise ValueError(f'No parquet shards under {spec["repo"]}:{spec["prefix"]}')

        def meta(file):
            def work():
                with _hub_open(spec, file) as f:
                    md = pq.ParquetFile(f).metadata
                    return dict(file=file, rows=md.num_rows, fields=md.schema.to_arrow_schema().names,
                                groups=[md.row_group(g).num_rows for g in range(md.num_row_groups)])
            return retry(work)

        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            shards = list(pool.map(meta, files))
        return dict(repo=spec['repo'], revision=spec['revision'], prefix=spec['prefix'], shards=shards)

    # ------------------------------------------------------------------ cohorts
    def cohort(self, column, values):
        """Sorted coordinates whose ``column`` equals one of ``values`` (None: all rows)."""
        if values is None:
            return None, {}
        per_value = {v: self._cohort_one(column, v) for v in values}
        if any(not rows for rows in per_value.values()):
            raise ValueError(f'Empty cohort in {self.spec["repo"]}: {column}')
        combined = sorted(set().union(*per_value.values()))
        if len(combined) != sum(map(len, per_value.values())):
            raise ValueError('Cohorts overlap')
        return combined, {v: len(r) for v, r in per_value.items()}

    def _cohort_one(self, column, value):
        c = self.catalog
        rel = f'cohorts/{digest([c["repo"], c["revision"], c["prefix"], column, value])[:20]}.json.gz'
        found = self.cache.find(rel)
        if found:
            with gzip.open(found, 'rt') as f:
                return json.load(f)
        coords, offset = [], 0
        for shard in c['shards']:
            def work():
                with _hub_open(self.spec, shard['file']) as f:
                    return pq.ParquetFile(f).read(columns=[column]).column(0).to_pylist()
            values = retry(work)
            if len(values) != shard['rows']:
                raise ValueError('Cohort column row count mismatch')
            coords.extend(offset + i for i, v in enumerate(values) if v == value)
            offset += shard['rows']
        path = self.cache.target(rel)
        with gzip.open(_tmp(path), 'wt') as f:
            json.dump(coords, f)
        _tmp(path).replace(path)
        return coords

    # ------------------------------------------------------------------ rows
    def provenance(self, coord):
        meta = self.groups[bisect.bisect_right(self.ends, coord)]
        return dict(repo=self.spec['repo'], revision=self.spec['revision'], file=meta['file'],
                    row_group=meta['group'], row_in_group=coord - meta['start'], global_row=coord)

    def rows(self, coords):
        """{coord: (row dict, provenance)} for the requested global coordinates."""
        by_group = collections.defaultdict(list)
        for c in coords:
            if not 0 <= c < self.size:
                raise IndexError(c)
            by_group[bisect.bisect_right(self.ends, c)].append(c)
        out = {}
        for g, members in by_group.items():
            reader = self._group(g)
            for c in members:
                out[c] = (reader(c - self.groups[g]['start']), self.provenance(c))
        return out

    def _group(self, g):
        """Row reader for one row group (cached as 512-row chunks; downloaded when absent)."""
        meta = self.groups[g]
        rel = f'{self.columns_dir}/{g:06d}'
        path = self.cache.find(f'{rel}.bounded.parquet') or self._store(g, self.cache.find(f'{rel}.parquet'))
        pf = pq.ParquetFile(path)
        ends = list(itertools.accumulate(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)))
        if not ends or ends[-1] != meta['rows']:
            raise ValueError(f'Cached row count mismatch: {path}')
        chunks = {}

        def read(i):
            k = bisect.bisect_right(ends, i)
            if k not in chunks:
                chunks[k] = pf.read_row_group(k, columns=self.spec['columns'])
            return chunks[k].slice(i - (ends[k - 1] if k else 0), 1).to_pylist()[0]
        return read

    def _store(self, g, legacy=None):
        """Write the projected row group as 512-row chunks (from a legacy cache file or the Hub)."""
        meta = self.groups[g]
        path = self.cache.target(f'{self.columns_dir}/{g:06d}.bounded.parquet')

        def work():
            tmp = _tmp(path)
            writer, count = None, 0
            with (open(legacy, 'rb') if legacy else _hub_open(self.spec, meta['file'])) as f:
                groups = None if legacy else [meta['group']]
                for batch in pq.ParquetFile(f).iter_batches(batch_size=CHUNK_ROWS, row_groups=groups,
                                                            columns=self.spec['columns']):
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, batch.schema, compression='zstd')
                    writer.write_batch(batch, row_group_size=CHUNK_ROWS)
                    count += batch.num_rows
            if writer is not None:
                writer.close()
            if count != meta['rows']:
                raise ValueError('Projected row count mismatch')
            tmp.replace(path)
        retry(work)
        return path


# OCR2 stores (dataset, split, index) references; the problem text lives in these originals.
ORIGINALS = {
    'taco': dict(repo='BAAI/TACO', revision='d593ed0a2becbbc952230bb89be09189bf1056dc',
                 prefix='ALL/train-', columns=['question', 'starter_code']),
    'code_contests': dict(repo='deepmind/code_contests', revision='802411c3010cb00d1b05bad57ca77365a3c699d6',
                          prefix='data/train-', columns=['name', 'description', 'input_file', 'output_file']),
    'open-r1/codeforces': dict(repo='open-r1/codeforces', revision='fbe3f6e903ee854eec2e69e9d96d0306cde59baf',
                               prefix='data/train-',
                               columns=['id', 'title', 'description', 'input_format', 'output_format',
                                        'interaction_format', 'examples', 'note', 'time_limit', 'memory_limit']),
}
APPS = dict(repo='codeparrot/apps', revision='21e74ddf8de1a21436da12e3e653065c5213e9d1', file='train.jsonl')


class CodeOriginals:
    """Resolve OCR2 rows to their original problem records (train split only)."""

    def __init__(self, cache, workers=4):
        self.cache, self.workers, self.pools, self.apps = cache, workers, {}, None

    def resolve(self, rows):
        """{(dataset, index): dict(record=..., provenance=...)} for the given OCR2 rows."""
        wanted = collections.defaultdict(set)
        for r in rows:
            wanted[r['dataset']].add(int(r['index']))
        out = {}
        for ds, indices in wanted.items():
            if ds == 'apps':
                if self.apps is None:
                    from huggingface_hub import hf_hub_download
                    path = retry(lambda: hf_hub_download(APPS['repo'], APPS['file'], repo_type='dataset',
                                                         revision=APPS['revision']))
                    with open(path) as f:
                        self.apps = [json.loads(line) for line in f]
                for i in indices:
                    out[ds, i] = dict(record={k: self.apps[i].get(k) for k in ('question', 'starter_code')},
                                      provenance=dict(repo=APPS['repo'], revision=APPS['revision'],
                                                      file=APPS['file'], row=i))
                continue
            if ds not in ORIGINALS:
                raise ValueError(f'Unmapped OCR2 original dataset {ds!r}')
            if ds not in self.pools:
                self.pools[ds] = ParquetSource(ORIGINALS[ds], self.cache, self.workers)
            for i, (record, prov) in self.pools[ds].rows(sorted(indices)).items():
                out[ds, i] = dict(record=record, provenance=prov)
        return out
