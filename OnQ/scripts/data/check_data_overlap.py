r"""Audit written training files for exact/contained benchmark overlap and duplicate records.

Test-set decontamination and duplicate gates for built OPD / Block-AP data.

Normalization is identical to QAOPD ``_norm``: ``re.sub(r'\s+', ' ', s).strip().lower()``.

Contamination hit (either):
  exact      a normalized user turn (or whole prompt) equals a normalized benchmark text;
  contained  a normalized benchmark text of >= 60 characters occurs inside the normalized
             training text (OPD: the prompt; Block-AP: the whole conversation incl. answers).
Benchmark texts shorter than 60 characters are checked for equality only.

Other gates: no duplicate normalized prompt inside a split, OPD train and val disjoint,
Block-AP one window per record. ``prepare_training_data.py`` applies the same checks during selection
(and additionally the rare-n-gram paraphrase matcher), so a finished build must report
zero hits; the gate re-reads the written files independently and fails otherwise.

This is verbatim matching. It does not detect paraphrases and is no proof of zero
contamination.

    python -m OnQ preparedata check --benchmarks HELDOUT_QUERIES.jsonl --opd OUT/opd --blockap OUT/blockap"""

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

MIN_CONTAINED = 60


def normalize(s):
    return re.sub(r'\s+', ' ', str(s)).strip().lower()


def read_jsonl(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return [json.loads(line) for line in f if line.strip()]


class BenchmarkIndex:
    """Exact and containment matcher over benchmark texts ({benchmark, id, text} rows)."""

    def __init__(self, rows, min_contained=MIN_CONTAINED):
        self.min = min_contained
        self.labels = collections.defaultdict(set)
        for r in rows:
            self.labels[normalize(r['text'])].add((r['benchmark'], str(r.get('id'))))
        self.labels.pop('', None)
        # Every occurrence of a pattern starts with its first `min` characters: exact anchor lookup.
        self.anchors = collections.defaultdict(list)
        for text in self.labels:
            if len(text) >= self.min:
                self.anchors[text[:self.min]].append(text)
        self.rows = len(rows)

    def _hit(self, kind, text):
        benchmark, item = sorted(self.labels[text])[0]
        return dict(match=kind, benchmark=benchmark, benchmark_id=item, benchmark_text=text[:120])

    def exact(self, text):
        t = normalize(text)
        return self._hit('exact', t) if t in self.labels else None

    def contained(self, text):
        t, k = normalize(text), self.min
        for i in range(len(t) - k + 1):
            for pattern in self.anchors.get(t[i:i + k], ()):
                if t.startswith(pattern, i):
                    return self._hit('contained', pattern)
        return None

    def check(self, turns, texts):
        """First hit: exact match of any user turn or whole text, else containment in any text."""
        texts = [texts] if isinstance(texts, str) else list(dict.fromkeys(texts))
        for t in [*turns, *texts]:
            hit = self.exact(t)
            if hit:
                return hit
        return next((hit for t in texts if (hit := self.contained(t))), None)


def load_near_matcher(rows):
    """Rare-n-gram / same-numbers paraphrase matcher shared with the Block-AP tooling."""
    from OnQ.scripts.data.text_quality import EvalIndex

    unique = {}
    for r in rows:  # aliases must not count one test item as several documents
        unique.setdefault(normalize(r['text']), r)
    return EvalIndex([(r['benchmark'], r['text']) for r in unique.values()])


# ---------------------------------------------------------------- views of built files
def opd_view(row):
    turns = [m['content'] for m in row['messages'] if m['role'] == 'user']
    return turns, '\n'.join(m['content'] for m in row['messages'] if m['role'] != 'assistant')


def blockap_view(raw):
    from OnQ.scripts.data.format_conversations import output_dialogue
    ref = raw.get('referenced_record')
    dialogue = output_dialogue(raw['kind'], raw['record'], ref['record'] if ref else None)
    turns = [m['content'] for m in dialogue if m['role'] == 'user']
    prompt = '\n'.join(m['content'] for m in dialogue if m['role'] != 'assistant')
    return turns, prompt, '\n'.join(m['content'] for m in dialogue)


def duplicates(keys):
    counts = collections.Counter(keys)
    return {k[:120]: n for k, n in counts.items() if n > 1}


def gate(index, opd_dir=None, blockap_dir=None):
    """Independent re-check of written outputs. Returns a report with status PASS/FAIL."""
    report, failures = {}, []
    if opd_dir:
        splits = {name: read_jsonl(Path(opd_dir) / f) for name, f in
                  (('train', 'opd_prompts.jsonl'), ('val', 'opd_val_prompts.jsonl'))}
        keys = {}
        for name, rows in splits.items():
            hits, keys[name] = [], []
            for r in rows:
                turns, prompt = opd_view(r)
                hit = index.check(turns, prompt)
                if hit:
                    hits.append(dict(id=r['id'], **hit))
                keys[name].append(normalize(prompt))
            dups = duplicates(keys[name])
            report[f'opd_{name}'] = dict(records=len(rows), contamination_hits=hits, duplicate_prompts=dups,
                                         duplicate_ids=len(rows) - len({r['id'] for r in rows}))
            failures += [f'opd_{name}: {len(hits)} contamination hits'] * bool(hits)
            failures += [f'opd_{name}: {len(dups)} duplicate prompts'] * bool(dups)
            failures += [f'opd_{name}: duplicate ids'] * bool(report[f'opd_{name}']['duplicate_ids'])
        overlap = set(keys['train']) & set(keys['val'])
        overlap_ids = {r['id'] for r in splits['train']} & {r['id'] for r in splits['val']}
        report['opd_train_val_overlap'] = dict(prompts=len(overlap), ids=len(overlap_ids))
        failures += ['opd train/val overlap'] * bool(overlap or overlap_ids)
    if blockap_dir:
        raws = read_jsonl(Path(blockap_dir) / 'raw_records.jsonl.gz')
        windows = read_jsonl(Path(blockap_dir) / 'train_tokens.jsonl.gz')
        hits, keys = [], []
        for raw in raws:
            turns, prompt, full = blockap_view(raw)
            hit = index.check(turns, [prompt, full])
            if hit:
                hits.append(dict(id=raw['id'], **hit))
            keys.append(normalize(prompt))
        used = collections.Counter(s['record_id'] for w in windows for s in w['segments'])
        dups = duplicates(keys)
        report['blockap'] = dict(records=len(raws), windows=len(windows), contamination_hits=hits,
                                 duplicate_prompts=dups,
                                 records_not_in_exactly_one_window=sum(used[r['id']] != 1 for r in raws)
                                 + len(set(used) - {r['id'] for r in raws}))
        failures += [f'blockap: {len(hits)} contamination hits'] * bool(hits)
        failures += [f'blockap: {len(dups)} duplicate prompts'] * bool(dups)
        failures += ['blockap: window/record mismatch'] * bool(report['blockap']['records_not_in_exactly_one_window'])
    report['benchmark_texts'] = dict(rows=index.rows, unique=len(index.labels),
                                     containment_patterns=sum(map(len, index.anchors.values())))
    report['status'] = 'FAIL' if failures else 'PASS'
    report['failures'] = failures
    return report


def summary(report):
    """Report with hit lists / duplicate maps replaced by their sizes."""
    def size(v):
        return len(v) if isinstance(v, (list, dict)) else v
    return {k: ({kk: size(vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in report.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--benchmarks', type=Path, required=True, help='prepare_benchmark_texts.py output (JSONL)')
    parser.add_argument('--opd', type=Path, help='directory with opd_prompts.jsonl / opd_val_prompts.jsonl')
    parser.add_argument('--blockap', type=Path, help='directory with raw_records.jsonl.gz / train_tokens.jsonl.gz')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if not (args.opd or args.blockap):
        parser.error('give --opd and/or --blockap')
    report = gate(BenchmarkIndex(read_jsonl(args.benchmarks)), args.opd, args.blockap)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.report:
        args.report.write_text(text + '\n')
    print(text if report['status'] == 'FAIL' else json.dumps(summary(report), indent=1))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
