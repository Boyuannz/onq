"""Consume preparation-script token windows without retokenizing or resampling."""
import gzip
import json
from pathlib import Path


def load_windows(path, tokenizer, train_size, val_size, seqlen):
    import torch
    if val_size != 0:
        raise ValueError('Token cache has training windows only; val_size must be zero')
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    windows, seen = [], set()
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            identity, ids = row['id'], row['input_ids']
            if identity in seen:
                raise ValueError('Duplicate calibration window ID')
            seen.add(identity)
            if (len(ids) != seqlen or any(type(i) is not int or not 0 <= i < len(tokenizer) for i in ids)):
                raise ValueError('Invalid token window length or vocabulary ID')
            x = torch.tensor([ids], dtype=torch.long)
            windows.append((x, x.clone()))
    if len(windows) != train_size:
        raise ValueError(f'Expected {train_size} windows, found {len(windows)}')
    return windows, []
