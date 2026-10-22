"""Prepare five version-pinned native benchmarks for EvalScope 1.5.1."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi
from OnQ.scripts.data import prepare_benchmark_texts as sources

# Preserve source row order and the filenames consumed by the native adapters.
SINGLE = {
    'gsm8k': ('gsm8k', 'main_test.jsonl', 1319),
    'math500': ('math500', 'test.jsonl', 500),
    'gpqa': ('gpqa', 'default_train.jsonl', 198),
    'ifeval': ('ifeval', 'default_train.jsonl', 541),
}


def prepare(output, cache=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources.CACHE = str(cache) if cache else None
    manifest = {}

    def write(benchmark, filename, rows, count):
        if len(rows) != count:
            raise ValueError(f'{benchmark}/{filename}: expected {count} rows, got {len(rows)}')
        path = output / benchmark / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        return dict(rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    for benchmark, (key, filename, count) in SINGLE.items():
        rows = sources.load(key)[0]
        manifest[benchmark] = dict(source=sources.PINS[key], files={filename: write(benchmark, filename, rows, count)})

    repo, revision, _ = sources.PINS['mmlu_redux']
    files = sorted(name for name in HfApi().list_repo_files(repo, repo_type='dataset', revision=revision)
                   if name.endswith('/data-00000-of-00001.arrow'))
    if len(files) != 57:
        raise ValueError(f'Expected 57 MMLU-Redux subjects, got {len(files)}')
    manifest['mmlu_redux'] = dict(source=(repo, revision, files), files={})
    for name in files:
        subject = name.split('/')[0]
        filename = subject + '_test.jsonl'
        rows = sources.load('mmlu_redux', [name])[0]
        manifest['mmlu_redux']['files'][filename] = write('mmlu_redux', filename, rows, 100)
    (output / 'SOURCE.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--cache', type=Path, help='Hugging Face and archive cache directory')
    args = parser.parse_args()
    manifest = prepare(args.output, args.cache)
    print(json.dumps({key: sum(item['rows'] for item in value['files'].values()) for key, value in manifest.items()}))


if __name__ == '__main__':
    main()
