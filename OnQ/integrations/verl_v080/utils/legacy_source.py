"""Validate old bufferless QAT formats without relabeling their weight provenance."""
import json
from pathlib import Path

from OnQ.quantization.checkpoint import validate_stage2_start_checkpoint
from OnQ.quantization.checkpoint_checks import audit_checkpoint


def verified_legacy_metadata(path, *, bits, group_size):
    path = Path(path)
    files = [path/name for name in ('unified_calibration_manifest.json', 'unified_stage2_start_manifest.json')
             if (path/name).is_file()]
    if len(files) != 1:
        raise ValueError('Bufferless checkpoint needs one unambiguous registered provenance manifest')
    declared = json.loads(files[0].read_text())
    audit = audit_checkpoint(path, 'official_fakequant', bits, group_size)
    if audit['status'] != 'PASS':
        raise ValueError(f'Legacy tensor/source audit failed: {audit.get("errors")}')
    return validate_stage2_start_checkpoint(
        checkpoint=path, model=declared['model'], bits=bits, group_size=group_size,
        recipe=declared['calibration_recipe'], audit_manifest=audit)
