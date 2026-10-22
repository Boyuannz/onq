"""Read saved quantizers without assuming HF shard filenames start with model."""
import json
from pathlib import Path

from safetensors import safe_open


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate checkpoint index key: {key}')
        result[key] = value
    return result


def load_quantizer_state(root):
    """Load only quantizer tensors, preserving their saved CPU dtype and values.

    An HF index is authoritative: audit its shard headers for misplaced,
    unindexed or duplicate quantizers, then read the declared quantizer keys.
    Dense tensors are never materialized. Without an index, retain the legacy
    model*.safetensors layout. The caller still checks the complete expected
    quantizer key set, shapes and configuration before restoring any state.
    """
    root = Path(root)
    index = root / 'model.safetensors.index.json'
    mapping = None
    if index.exists():
        record = json.loads(index.read_text(), object_pairs_hook=_unique_json_object)
        mapping = record.get('weight_map') if isinstance(record, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Checkpoint index requires a nonempty weight_map')
        for key, filename in mapping.items():
            if (not isinstance(key, str) or not isinstance(filename, str)
                    or not filename.endswith('.safetensors')
                    or Path(filename).is_absolute() or '..' in Path(filename).parts):
                raise ValueError(f'Invalid checkpoint index entry: {key}')
        shard_names = sorted(set(mapping.values()))
    else:
        shard_names = [p.name for p in sorted(root.glob('model*.safetensors'))]
    state = {}
    for filename in shard_names:
        path = root / filename
        if not path.is_file():
            raise ValueError(f'Missing checkpoint shard: {filename}')
        with safe_open(path, framework='pt', device='cpu') as handle:
            keys = {key for key in handle.keys() if '.weight_quantizer.' in key}
            for key in sorted(keys):
                if key in state:
                    raise ValueError(f'Duplicate checkpoint quantizer key: {key}')
                if mapping is not None and mapping.get(key) != filename:
                    raise ValueError(f'Checkpoint quantizer index/shard mismatch: {key} in {filename}')
            if mapping is not None:
                expected = {key for key, shard in mapping.items()
                            if shard == filename and '.weight_quantizer.' in key}
                if expected - keys:
                    raise ValueError(f'Missing indexed quantizer keys in {filename}: {sorted(expected - keys)}')
            for key in sorted(keys):
                state[key] = handle.get_tensor(key)
    return state
