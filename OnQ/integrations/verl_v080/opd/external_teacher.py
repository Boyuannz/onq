"""Transport adapter for native k1 OPD and a separately served dense vLLM."""
import asyncio
from contextlib import asynccontextmanager
import math
import os
import ipaddress
import json
import logging
import urllib.request
import ssl
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import aiohttp

from ..utils.version_guard import assert_pinned_upstream

assert_pinned_upstream()

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs
from verl.single_controller.ray import ResourcePoolManager
from verl.trainer.ppo.utils import Role

logger = logging.getLogger(__name__)


def settings(config):
    encoded = config.get('opd_external_teacher_json')
    if encoded is None:
        return None
    value = json.loads(encoded)
    required = {'base_url', 'model_name', 'token_file'}
    optional = {'ca_file', 'private_http_endpoint', 'request_timeout_seconds',
                'max_concurrent_requests', 'max_connection_retries', 'connection_keepalive_seconds'}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional:
        raise ValueError('Invalid external teacher transport configuration')
    validate_request_limit(value.get('max_concurrent_requests'))
    validate_connection_retries(value.get('max_connection_retries', 0))
    validate_connection_keepalive(value.get('connection_keepalive_seconds'))
    return value


def validate_request_limit(value):
    # None preserves the transport behavior of existing recorded experiments.
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError('Teacher max_concurrent_requests must be a positive integer or null')


def validate_connection_retries(value):
    if type(value) is not int or not 0 <= value <= 3:
        raise ValueError('Teacher max_connection_retries must be an integer in [0, 3]')


def validate_connection_keepalive(value):
    if value is not None and (type(value) not in (int, float)
                              or not math.isfinite(value) or value <= 0):
        raise ValueError('Teacher connection_keepalive_seconds must be finite and positive or null')


class ExternalTeacherResourcePoolManager(ResourcePoolManager):
    def get_resource_pool(self, role):
        if role == Role.TeacherModel:
            return None
        return super().get_resource_pool(role)


def externalize_teacher_pool(manager):
    if manager.resource_pool_dict or 'teacher_pool' not in manager.resource_pool_spec:
        raise ValueError('External teacher must be configured before allocating native pools')
    return ExternalTeacherResourcePoolManager(
        resource_pool_spec={k: v for k, v in manager.resource_pool_spec.items() if k != 'teacher_pool'},
        mapping=manager.mapping.copy(), max_colocate_count=manager.max_colocate_count)


class ExternalTeacherModelManager:
    """Supplies a native-client-compatible transport without a local teacher GPU."""
    def __init__(self, config, resource_pool):
        from verl.utils.config import omega_conf_to_dataclass

        if resource_pool is not None:
            raise ValueError('An external teacher must not reserve a local GPU pool')
        distillation = omega_conf_to_dataclass(config.distillation)
        if len(distillation.teacher_models) != 1:
            raise ValueError('External transport currently supports one dense teacher')
        self.key = next(iter(distillation.teacher_models))
        from ..utils.union_transport import union_spec, NoRolloutTeacherClient, teacher_metadata
        union = union_spec(config)
        if union is not None:
            from ..utils.union_protocol import required_input_tokens
            self.client = NoRolloutTeacherClient()
            self.endpoint_metadata = teacher_metadata(union, required_tokens=required_input_tokens(config))
            return
        self.client = ExternalTeacherClient(config, **settings(config))
        self.endpoint_metadata = self.client.validate_remote(
            config.data.max_prompt_length + config.data.max_response_length + 1,
            Path(distillation.teacher_models[self.key].model_path).name)

    def get_client(self):
        return {self.key: self.client}


def parse_teacher_reply(reply, prompt_ids, model_name, topk=0):
    if type(topk) is not int or topk < 0:
        raise ValueError('Invalid teacher top-k width')
    if reply.get('model') != model_name or len(reply.get('choices', [])) != 1:
        raise ValueError('Unexpected teacher model or number of choices')
    choice = reply['choices'][0]
    if choice.get('prompt_token_ids') != prompt_ids:
        raise ValueError('Teacher did not score the exact student prompt token IDs')
    scores = choice.get('prompt_logprobs')
    if not isinstance(scores, list) or len(scores) != len(prompt_ids) or scores[0] is not None:
        raise ValueError('Missing or misaligned teacher prompt logprobs')
    selected = [None]
    for token_id, entry in zip(prompt_ids[1:], scores[1:], strict=True):
        if not isinstance(entry, dict):
            raise ValueError('Missing sampled-token teacher logprob')
        if topk:
            if len(entry) not in (topk, topk + 1):
                raise ValueError('Teacher top-k width mismatch')
            converted, ranks = {}, set()
            for key, value in entry.items():
                rank = value.get('rank')
                logprob = float(value.get('logprob', float('nan')))
                if type(rank) is not int or rank < 1 or not math.isfinite(logprob) or logprob > 0:
                    raise ValueError('Invalid teacher top-k rank or logprob')
                if rank <= topk:
                    if rank in ranks:
                        raise ValueError('Duplicate teacher top-k rank')
                    ranks.add(rank)
                elif int(key) != token_id:
                    raise ValueError('Unexpected non-top-k teacher token')
                if int(key) < 0 or int(key) in converted:
                    raise ValueError('Invalid or duplicate teacher token ID')
                converted[int(key)] = SimpleNamespace(rank=rank, logprob=logprob)
            if ranks != set(range(1, topk + 1)):
                raise ValueError('Incomplete teacher top-k ranks')
            selected.append(converted)
            continue
        value = entry.get(str(token_id), entry.get(token_id))
        if not isinstance(value, dict) or 'logprob' not in value:
            raise ValueError('Teacher response omitted a sampled token')
        logprob = float(value['logprob'])
        if not math.isfinite(logprob):
            raise ValueError('Non-finite teacher logprob')
        selected.append({token_id: SimpleNamespace(logprob=logprob)})
    if choice.get('finish_reason') not in ('stop', 'length'):
        raise ValueError('Teacher request did not complete')
    fields = {}
    # Preserve native next-token shift and dummy-final-slot behavior.
    extract_prompt_logprobs(SimpleNamespace(prompt_logprobs=selected), topk, fields)
    return TokenOutput(token_ids=choice.get('token_ids', []), log_probs=None,
                       stop_reason='completed', extra_fields=fields)


class ExternalTeacherClient(LLMServerClient):
    def __init__(self, config, *, base_url, model_name, token_file, ca_file=None,
                 private_http_endpoint=None, request_timeout_seconds=180,
                 max_concurrent_requests=None, max_connection_retries=0,
                 connection_keepalive_seconds=None):
        super().__init__(config, load_balancer_handle=None)
        url = urlparse(base_url)
        private_allowed = False
        if private_http_endpoint is not None:
            if private_http_endpoint != base_url or url.scheme != 'http' or url.port is None:
                raise ValueError('Private HTTP authorization must match the exact IP endpoint and port')
            try:
                address = ipaddress.IPv4Address(url.hostname)
            except (ValueError, TypeError) as exc:
                raise ValueError('Private HTTP requires a literal RFC1918 IPv4 address') from exc
            private_allowed = any(address in ipaddress.IPv4Network(network)
                                  for network in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))
            if not private_allowed:
                raise ValueError('Private HTTP requires a literal RFC1918 IPv4 address')
        if not private_allowed and url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost', '::1')):
            raise ValueError('Teacher transport must use HTTPS or an SSH loopback tunnel')
        if url.username or url.password or url.query or url.fragment:
            raise ValueError('Do not embed credentials or query data in teacher URL')
        if url.path not in ('', '/'):
            raise ValueError('Teacher base URL must be the server root, without /v1')
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.token_file = str(token_file)
        self.ca_file = str(ca_file) if ca_file else None
        if (not isinstance(request_timeout_seconds, (int, float)) or
                isinstance(request_timeout_seconds, bool) or
                not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0):
            raise ValueError('Teacher request timeout must be finite and positive')
        self.request_timeout_seconds = float(request_timeout_seconds)
        validate_request_limit(max_concurrent_requests)
        self.max_concurrent_requests = max_concurrent_requests
        validate_connection_retries(max_connection_retries)
        self.max_connection_retries = max_connection_retries
        validate_connection_keepalive(connection_keepalive_seconds)
        self.connection_keepalive_seconds = connection_keepalive_seconds
        self.topk = 0
        if config is not None and config.distillation.distillation_loss.loss_mode == 'forward_kl_topk':
            self.topk = config.distillation.distillation_loss.topk
            if type(self.topk) is not int or self.topk != 20:
                raise ValueError('Native forward KL requires the reviewed teacher top20')
        self._session = None
        self._session_owner = None
        self._ssl_context = None
        self._request_semaphore = None
        self._request_semaphore_owner = None
        self._closed = False

    def __getstate__(self):
        # Ray copies the configuration into each agent worker. Never ship live
        # sockets, event loops or SSL contexts to another process.
        state = self.__dict__.copy()
        state.update(_session=None, _session_owner=None, _ssl_context=None,
                     _request_semaphore=None, _request_semaphore_owner=None)
        return state

    @asynccontextmanager
    async def _request_slot(self):
        """Bound each worker's complete HTTP scoring lifecycle, not GPU batches."""
        if self._closed:
            raise RuntimeError('Teacher client is shut down')
        if self.max_concurrent_requests is None:
            yield
            return
        owner = (os.getpid(), asyncio.get_running_loop())
        if self._request_semaphore is None:
            self._request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            self._request_semaphore_owner = owner
        elif self._request_semaphore_owner != owner:
            raise RuntimeError('Teacher client cannot change process or event loop; create a new client')
        async with self._request_semaphore:
            # Cleanup may start while this request is queued. Never reopen its
            # session after obtaining a slot released by a cancelled sibling.
            if self._closed:
                raise RuntimeError('Teacher client is shut down')
            yield

    def ssl_context(self):
        if self.ca_file and self._ssl_context is None:
            self._ssl_context = ssl.create_default_context(cafile=self.ca_file)
        return self._ssl_context

    def _get_session(self):
        if self._closed:
            raise RuntimeError('Teacher client is shut down')
        owner = (os.getpid(), asyncio.get_running_loop())
        if self._session is not None and not self._session.closed:
            if self._session_owner != owner:
                raise RuntimeError('Teacher client cannot change process or event loop; create a new client')
            return self._session
        connector_options = dict(limit=0, ssl=self.ssl_context() or True)
        if self.connection_keepalive_seconds is not None:
            connector_options['keepalive_timeout'] = self.connection_keepalive_seconds
        connector = aiohttp.TCPConnector(**connector_options)
        self._session = aiohttp.ClientSession(
            connector=connector, cookie_jar=aiohttp.DummyCookieJar(),
            timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds))
        self._session_owner = owner
        return self._session

    def begin_shutdown(self):
        # Synchronous admission barrier: late trajectories cannot reopen a pool
        # while the worker is awaiting cancellation/cleanup of other tasks.
        self._closed = True

    async def aclose(self):
        self.begin_shutdown()
        if self._session is not None and not self._session.closed:
            if self._session_owner != (os.getpid(), asyncio.get_running_loop()):
                raise RuntimeError('Teacher client must close in its owning event loop')
            await self._session.close()
        self._session = self._session_owner = None
        self._request_semaphore = self._request_semaphore_owner = None

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError('Teacher client is shut down')
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    def validate_remote(self, required_context, expected_revision):
        token = Path(self.token_file).read_text().strip()
        if not token:
            raise ValueError('Teacher authentication token is empty')
        request = urllib.request.Request(self.base_url + '/v1/models',
                                         headers={'Authorization': 'Bearer ' + token})
        with urllib.request.urlopen(request, timeout=20, context=self.ssl_context()) as response:
            data = json.load(response)
        matches = [m for m in data.get('data', []) if m.get('id') == self.model_name]
        if len(matches) != 1:
            raise ValueError('Expected teacher model is not uniquely served')
        model = matches[0]
        if model.get('max_model_len', 0) < required_context:
            raise ValueError('Teacher context is smaller than student prompt + response + 1')
        if Path(model.get('root', '')).name != expected_revision:
            raise ValueError('Teacher endpoint revision differs from the selected snapshot')
        return {k: model[k] for k in ('id', 'root', 'max_model_len')}

    async def generate(self, request_id, *, prompt_ids, sampling_params,
                       image_data=None, video_data=None, audio_data=None,
                       mm_processor_kwargs=None, **kwargs):
        if image_data or video_data or audio_data or mm_processor_kwargs or kwargs:
            raise ValueError('This teacher client supports text-only scoring')
        if sampling_params != {'max_tokens': 1, 'temperature': 1.0, 'prompt_logprobs': self.topk}:
            raise ValueError('External teacher requires native sampled-token scoring at temperature 1')
        if len(prompt_ids) < 2:
            raise ValueError('Teacher scoring requires a prompt plus a response token')
        token = Path(self.token_file).read_text().strip()
        if not token:
            raise ValueError('Teacher authentication token is empty')
        payload = dict(model=self.model_name, prompt=prompt_ids, max_tokens=1,
                       temperature=1.0, prompt_logprobs=self.topk, return_token_ids=True,
                       add_special_tokens=False, stream=False)
        headers = {'Authorization': 'Bearer ' + token, 'X-Request-Id': request_id}
        async with self._request_slot():
            # These scoring calls have no training side effects. Keep the exact
            # request and the same admission slot across bounded serial retries.
            # Closing the shared session here would abort healthy sibling calls;
            # aiohttp discards the broken connection when its context unwinds.
            max_attempts = self.max_connection_retries + 1
            for attempt in range(1, max_attempts + 1):
                try:
                    session = self._get_session()
                    async with session.post(self.base_url + '/v1/completions', json=payload, headers=headers) as response:
                        response.raise_for_status()
                        reply = await response.json()
                    break
                except aiohttp.ClientConnectionError as exc:
                    # ServerTimeoutError also inherits ClientConnectionError.
                    # Timeouts and TLS validation failures are not stale sockets.
                    if isinstance(exc, (asyncio.TimeoutError, aiohttp.ClientSSLError,
                                        aiohttp.ServerFingerprintMismatch)):
                        raise
                    retry = attempt < max_attempts and not self._closed
                    logger.warning('Teacher scoring connection failure request_id=%s '
                                   'attempt=%d max_attempts=%d retry=%s error_type=%s errno=%s',
                                   request_id, attempt, max_attempts, retry,
                                   type(exc).__name__, getattr(exc, 'errno', None))
                    if not retry:
                        raise
                    # Fixed bounded backoff also makes cancellation/shutdown
                    # observable before the next admission to the HTTP pool.
                    await asyncio.sleep(.25 * 2 ** (attempt - 1))
            # Target validation is deliberately outside the connection retry.
            # Invalid JSON, HTTP statuses and cancelled calls also never retry.
            result = parse_teacher_reply(reply, prompt_ids, self.model_name, self.topk)
            if attempt > 1:
                logger.warning('Teacher scoring connection recovered request_id=%s attempts=%d',
                               request_id, attempt)
            return result
