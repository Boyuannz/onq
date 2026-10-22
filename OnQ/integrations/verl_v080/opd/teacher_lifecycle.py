"""Worker-owned HTTP pool cleanup through native agent-loop extensions."""
import asyncio
from contextlib import contextmanager
import logging

import ray
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker

from .external_teacher import ExternalTeacherClient


class ExternalTeacherAgentLoopWorker(AgentLoopWorker):
    teacher_shutdown_timeout = 20

    def __init__(self, *args, **kwargs):
        self._teacher_closing = False
        self._trajectory_tasks = set()
        self._teacher_shutdown_task = None
        super().__init__(*args, **kwargs)
        self._union_actor_teacher = None

    async def _compute_teacher_logprobs(self, output, *args, **kwargs):
        if self._union_actor_teacher is None:
            from ..utils.union_transport import union_spec
            self._union_actor_teacher = union_spec(self.config) is not None
        if self._union_actor_teacher:
            if 'teacher_ids' in output.extra_fields or 'teacher_logprobs' in output.extra_fields:
                raise ValueError('Unexpected rollout teacher targets in Union actor-scoring mode')
            return
        return await super()._compute_teacher_logprobs(output, *args, **kwargs)

    async def generate_sequences(self, *args, **kwargs):
        if self._teacher_closing:
            raise RuntimeError('Teacher worker is shut down')
        return await super().generate_sequences(*args, **kwargs)

    async def _run_agent_loop(self, *args, **kwargs):
        if self._teacher_closing:
            raise RuntimeError('Teacher worker is shut down')
        task = asyncio.current_task()
        self._trajectory_tasks.add(task)
        try:
            return await super()._run_agent_loop(*args, **kwargs)
        finally:
            self._trajectory_tasks.discard(task)

    async def close_teacher_clients(self):
        if self._teacher_shutdown_task is None:
            self._teacher_closing = True
            clients = {id(c): c for c in (self.teacher_client or {}).values()
                       if isinstance(c, ExternalTeacherClient)}
            for client in clients.values():
                client.begin_shutdown()
            self._teacher_shutdown_task = asyncio.create_task(self._drain_and_close(clients))
        # Concurrent or cancelled cleanup RPCs must not cancel shared cleanup.
        await asyncio.shield(self._teacher_shutdown_task)

    async def _drain_and_close(self, clients):
        tasks = set(self._trajectory_tasks)
        pending = set()
        try:
            for task in tasks:
                task.cancel()
            if tasks:
                done, pending = await asyncio.wait(tasks, timeout=self.teacher_shutdown_timeout)
                await asyncio.gather(*done, return_exceptions=True)
        finally:
            await asyncio.gather(*(c.aclose() for c in clients.values()))
        if pending:
            raise RuntimeError(f'{len(pending)} rollout tasks did not stop during teacher shutdown')


class ExternalTeacherAgentLoopManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(ExternalTeacherAgentLoopWorker)
        super().__init__(*args, **kwargs)

    def close_teacher_clients(self):
        calls = [worker.close_teacher_clients.remote()
                 for worker in getattr(self, 'agent_loop_workers', ())]
        if calls:
            ray.get(calls, timeout=30)


@contextmanager
def teacher_clients_lifecycle(trainer, *, close_on_success=True):
    primary_error = None
    try:
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        manager = getattr(trainer, 'async_rollout_manager', None)
        close = getattr(manager, 'close_teacher_clients', None)
        if close is not None and (close_on_success or primary_error is not None):
            try:
                close()
            except Exception as exc:
                if primary_error is None:
                    raise
                note = f'Teacher HTTP pool cleanup failed: {exc!r}'
                if hasattr(primary_error, 'add_note'):
                    primary_error.add_note(note)
                else:  # Python 3.10 also supports the integration.
                    logging.getLogger(__name__).warning(note)
