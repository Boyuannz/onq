"""Union targets are requested during actor forward, never during rollout."""

import functools
import json
import math
import re
import ssl
import urllib.request
from urllib.parse import urlparse

from .model_adapter import effective_model_adapter


def union_spec(config):
    return effective_model_adapter(config.actor_rollout_ref.model).get("union")


def validate_union_spec(spec):
    """Validate the accepted single-teacher protocol without contacting a server."""
    required = {
        "url",
        "token_file",
        "revision",
        "timeout",
        "teacher_topk",
        "student_topk",
    }
    # ca_file pins a private CA for an intranet HTTPS teacher.
    if not isinstance(spec, dict) or not required <= set(spec) <= required | {"ca_file"}:
        raise ValueError(
            "Union requires an explicit exact-candidate teacher spec: "
            + ", ".join(sorted(required))
        )
    for key in ("teacher_topk", "student_topk"):
        if type(spec[key]) is not int or spec[key] != 20:
            raise ValueError("Union requires teacher_topk=student_topk=20")
    if (
        type(spec["timeout"]) not in (int, float)
        or not math.isfinite(spec["timeout"])
        or spec["timeout"] <= 0
    ):
        raise ValueError("Union teacher timeout must be finite and positive")
    if not isinstance(spec["token_file"], str) or not spec["token_file"].startswith(
        "/"
    ):
        raise ValueError(
            "Union token_file must be an absolute path, never a token literal"
        )
    if not isinstance(spec["revision"], str) or not re.fullmatch(
        "[0-9a-f]{40}", spec["revision"]
    ):
        raise ValueError(
            "Union teacher revision must be an immutable 40-character commit"
        )
    if not isinstance(spec["url"], str):
        raise ValueError("Union teacher URL must be a string")
    url = urlparse(spec["url"])
    if "ca_file" in spec and (
        not isinstance(spec["ca_file"], str)
        or not spec["ca_file"].startswith("/")
        or url.scheme != "https"
    ):
        raise ValueError("Union ca_file must be an absolute path for an HTTPS teacher")
    if (
        url.scheme != "https"
        and not (
            url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
        )
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        raise ValueError(
            "Union teacher requires HTTPS or an SSH loopback tunnel at the server root"
        )
    return spec


class NoRolloutTeacherClient:
    async def generate(self, *args, **kwargs):
        raise RuntimeError(
            "Union must request its sole teacher in actor forward, not rollout"
        )


@functools.lru_cache(maxsize=None)
def _pinned_context(ca_file):
    return ssl.create_default_context(cafile=ca_file)


def tls_options(spec):
    """urlopen keyword arguments; system CAs unless the spec pins a private CA."""
    return {"context": _pinned_context(spec["ca_file"])} if spec.get("ca_file") else {}


def teacher_metadata(spec, *, required_tokens):
    with urllib.request.urlopen(
        spec["url"].rstrip("/") + "/metadata", timeout=10, **tls_options(spec)
    ) as response:
        result = json.load(response)
    if (
        result.get("revision") != spec["revision"]
        or result.get("temperature") != 1
        or result.get("dtype") != "bfloat16"
        or result.get("backend") != "vllm_0.12.0"
    ):
        raise ValueError("Union teacher identity or precision mismatch")
    from .union_protocol import validate_capacity
    validate_capacity(required_tokens, result.get('max_input_tokens'), result.get('max_model_len'))
    return dict(
        result,
        base_url=spec["url"],
        rollout_teacher_rpc=False,
        scoring_phase="actor_forward",
        legacy_rollout_endpoint_ignored=True,
    )
