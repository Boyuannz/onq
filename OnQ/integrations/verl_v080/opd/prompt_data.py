"""Freeze two random passes over an already accepted prompt pool; never reselect it."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .schedule import response_cap, V4_SCHEDULE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand(pool, validation, seed=42):
    for rows, size in ((pool, 25600), (validation, 256)):
        if len(rows) != size or len({r['id'] for r in rows}) != size:
            raise ValueError('Incorrect prompt count or duplicate IDs')
        if len({r['prompt_sha256'] for r in rows}) != size:
            raise ValueError('Duplicate prompt hashes')
        for row in rows:
            if not row['messages'] or any(m['role'] not in ('system', 'user') for m in row['messages']):
                raise ValueError('OPD data must contain prompts, not supplied assistant answers')
    if ({r['prompt_sha256'] for r in pool} & {r['prompt_sha256'] for r in validation}
            or {r['id'] for r in pool} & {r['id'] for r in validation}):
        raise ValueError('Training/validation overlap')

    def convert(row, index, validation=False):
        return dict(data_source=row['domain'], prompt=row['messages'], ability='distillation',
                    reward_model=dict(style='rule', ground_truth=''),
                    extra_info=dict(index=index, row_id=index, prompt_id=row['id'],
                        prompt_sha256=row['prompt_sha256'], source=row['source'], domain=row['domain'],
                        validation=validation, plan_step=0 if validation else index // 128 + 1,
                        response_cap=16384 if validation else response_cap(index // 128 + 1, schedule=V4_SCHEDULE)))
    train = []
    for epoch in range(2):
        rows = list(pool)
        random.Random(seed + epoch).shuffle(rows)
        for row in rows:
            train.append(convert(row, len(train)))
    return train, [convert(r, i, True) for i, r in enumerate(validation)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    # JSON strings may contain literal Unicode line separators: splitlines()
    # would split a valid JSON object in the middle of a prompt.
    pool, val = ([json.loads(line) for line in file.open() if line.strip()]
                 for file in (a.train, a.validation))
    train_rows, val_rows = expand(pool, val)
    a.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('train.parquet', train_rows), ('validation.parquet', val_rows)):
        pq.write_table(pa.Table.from_pylist(rows), a.output / name)
    manifest = dict(version=1, prompts=25600, appearances=2, validation_prompts=256,
                    seed=42, schedule=V4_SCHEDULE, rows=len(train_rows),
                    source_counts=dict(Counter(r['source'] for r in pool)),
                    domain_counts=dict(Counter(r['domain'] for r in pool)),
                    original_train_sha256=sha(a.train), original_validation_sha256=sha(a.validation),
                    validation_ids=[r['id'] for r in val],
                    files={n: sha(a.output / n) for n in ('train.parquet', 'validation.parquet')})
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
