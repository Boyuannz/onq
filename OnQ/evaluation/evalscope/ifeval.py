"""Check IFEval scorer results and 541-item output coverage."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

METRICS = ("prompt_level_strict", "inst_level_strict",
           "prompt_level_loose", "inst_level_loose")
COUNT = 541


def validate_scores(values):
    if not isinstance(values, dict) or any(name not in values for name in METRICS):
        raise RuntimeError("IFEval scorer returned missing metrics; refusing partial scoring")
    for name in METRICS:
        value = values[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise RuntimeError(f"IFEval scorer returned an invalid {name}")


@contextmanager
def strict_scorer():
    from evalscope.benchmarks.ifeval.ifeval_adapter import IFEvalAdapter
    original = IFEvalAdapter.match_score

    def checked(self, *args, **kwargs):
        score = original(self, *args, **kwargs)
        # EvalScope 1.5.1 catches scorer exceptions and returns value={}.
        validate_scores(score.value)
        return score

    IFEvalAdapter.match_score = checked
    try:
        yield
    finally:
        IFEvalAdapter.match_score = original


def input_records(local_data):
    if local_data is None:
        return None
    path = Path(local_data) / "default_train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != COUNT or len({row["key"] for row in rows}) != COUNT:
        raise ValueError("IFEval requires all 541 distinct input keys")
    return rows


def _read_outputs(output, kind):
    paths = sorted((Path(output) / kind).rglob("ifeval_*.jsonl"))
    rows = [json.loads(line) for path in paths
            for line in path.read_text().splitlines() if line.strip()]
    indices = [row["index"] for row in rows]
    if (len(rows) != COUNT or any(type(i) is not int for i in indices)
            or set(indices) != set(range(COUNT))):
        raise RuntimeError(f"IFEval {kind} must contain each of the 541 indices exactly once")
    return {row["index"]: row for row in rows}


def audit_coverage(output, inputs=None):
    predictions = _read_outputs(output, "predictions")
    reviews = _read_outputs(output, "reviews")
    keys = set()
    for index in range(COUNT):
        prediction = predictions[index]
        review = reviews[index]["sample_score"]
        if review["sample_id"] != index:
            raise RuntimeError(f"IFEval sample ID mismatch at index {index}")
        validate_scores(review["score"]["value"])
        metadata = prediction["metadata"]
        scored_metadata = review["sample_metadata"]
        if not isinstance(metadata, dict) or not isinstance(scored_metadata, dict):
            raise RuntimeError(f"IFEval metadata missing at index {index}")
        fields = ("key", "prompt", "instruction_id_list", "kwargs")
        if any(metadata[field] != scored_metadata[field] for field in fields):
            raise RuntimeError(f"IFEval prediction/review identity mismatch at index {index}")
        if inputs is not None and any(metadata[field] != inputs[index][field] for field in fields):
            raise RuntimeError(f"IFEval output differs from prepared input at index {index}")
        keys.add(metadata["key"])
    if len(keys) != COUNT:
        raise RuntimeError("IFEval output contains duplicate question keys")
    files = sorted(path for kind in ("predictions", "reviews")
                   for path in (Path(output) / kind).rglob("ifeval_*.jsonl"))
    return {"status": "PASS", "predictions": COUNT, "scored": COUNT,
            "unique_question_keys": COUNT, "metrics": list(METRICS),
            "prepared_input_identity_checked": inputs is not None,
            "sha256": {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in files}}
