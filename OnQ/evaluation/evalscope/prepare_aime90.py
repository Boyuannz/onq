"""Prepare a local AIME 2024–2026 adapter input; publishes no benchmark rows."""
import argparse
import json
from pathlib import Path


def validate(rows):
    if len(rows) != 90 or len({str(x['id']) for x in rows}) != 90:
        raise ValueError('Expected 90 distinct AIME question IDs')
    if {int(x['year']) for x in rows} != {2024, 2025, 2026}:
        raise ValueError('Expected years 2024, 2025 and 2026')
    for year in (2024, 2025, 2026):
        if sum(int(x['year']) == year for x in rows) != 30:
            raise ValueError(f'Expected 30 questions for {year}')
    if any(not str(x.get('problem') or '').strip() or not str(x.get('answer') if x.get('answer') is not None else '').strip() for x in rows):
        raise ValueError('Each question needs problem and answer')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, nargs='+', required=True,
                   help='Official JSONL inputs with id, year, problem, answer fields')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(line) for f in a.input for line in f.read_text().splitlines() if line.strip()]
    validate(rows)
    rows = [dict(row, answer=str(row["answer"]).strip()) for row in rows]
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'default_test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


if __name__ == '__main__':
    main()
