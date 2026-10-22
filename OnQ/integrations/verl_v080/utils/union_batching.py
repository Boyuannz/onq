"""Bounded coalescing for frozen-teacher scoring; no CUDA/HTTP dependencies."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque


class TeacherOverloaded(RuntimeError):
    pass


class AsyncTeacherAdmission:
    """Bound outstanding work, leaving batching/scheduling to AsyncLLM.

    Each handler call is an independent request. This gate limits concurrent
    sequences and their total input tokens; it never constructs fixed batches
    or waits for a slow neighbour before admitting a replacement. Cancellation
    is local to the calling request. Shutdown drains work already admitted.
    """

    def __init__(self, handler, *, max_batch=1, max_tokens=32768, max_pending=32):
        if (type(max_batch) is not int or not 1 <= max_batch <= 32
                or type(max_tokens) is not int or max_tokens < 1
                or type(max_pending) is not int or max_pending < max_batch):
            raise ValueError("Invalid teacher admission bounds")
        self.handler = handler
        self.max_batch, self.max_tokens, self.max_pending = max_batch, max_tokens, max_pending
        self.pending = self.active = self.active_tokens = 0
        self.closed = False
        self.waiters = deque()
        self.condition = asyncio.Condition()

    async def submit(self, item, *, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.max_tokens:
            raise ValueError("Request exceeds teacher input token budget")
        ticket = object()
        async with self.condition:
            if self.closed:
                raise RuntimeError("Teacher admission is closed")
            if self.pending >= self.max_pending:
                raise TeacherOverloaded("Teacher pending-request limit reached")
            self.pending += 1
            self.waiters.append(ticket)
            try:
                await self.condition.wait_for(lambda: self.waiters[0] is ticket
                    and self.active < self.max_batch
                    and self.active_tokens + tokens <= self.max_tokens)
            except BaseException:
                self.waiters.remove(ticket)
                self.pending -= 1
                self.condition.notify_all()
                raise
            self.waiters.popleft()
            self.active += 1
            self.active_tokens += tokens
            self.condition.notify_all()
        try:
            return await self.handler(item)
        finally:
            async with self.condition:
                self.active -= 1
                self.active_tokens -= tokens
                self.pending -= 1
                self.condition.notify_all()

    async def close(self):
        async with self.condition:
            self.closed = True
            await self.condition.wait_for(lambda: self.pending == 0)


class TeacherBatcher:
    """One inference thread, bounded outstanding requests, stable reply order.

    The caller validates each request before admission. The synchronous handler
    must return one result per item, in the same order. No request is retried by
    this scheduler. A cancelled client cannot cancel another client's scoring.
    """

    def __init__(self, handler, *, max_batch=1, max_tokens=32768,
                 wait_ms=2.0, max_pending=32):
        if (type(max_batch) is not int or not 1 <= max_batch <= 32
                or type(max_tokens) is not int or max_tokens < 1
                or type(max_pending) is not int or max_pending < max_batch
                or not 0 <= wait_ms <= 100):
            raise ValueError("Invalid teacher batch bounds")
        self.handler = handler
        self.max_batch = max_batch
        self.max_tokens = max_tokens
        self.wait_seconds = wait_ms / 1000
        self.max_pending = max_pending
        self.pending = 0
        self.closed = False
        self.queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="union-teacher")
        self.task = None

    async def submit(self, item, *, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.max_tokens:
            raise ValueError("Request exceeds teacher batch token budget")
        if self.closed:
            raise RuntimeError("Teacher batcher is closed")
        if self.pending >= self.max_pending:
            raise TeacherOverloaded("Teacher pending-request limit reached")
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        future = asyncio.get_running_loop().create_future()
        self.pending += 1
        self.queue.put_nowait((item, tokens, future))
        return await future

    async def _run(self):
        loop = asyncio.get_running_loop()
        carry = None
        stopping = False
        while True:
            first = carry if carry is not None else await self.queue.get()
            carry = None
            if first is None:
                break
            if first[2].cancelled():
                self.pending -= 1
                continue
            batch, tokens = [first], first[1]
            deadline = loop.time() + self.wait_seconds
            while len(batch) < self.max_batch:
                try:
                    entry = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        entry = await asyncio.wait_for(self.queue.get(), remaining)
                    except asyncio.TimeoutError:
                        break
                if entry is None:
                    stopping = True
                    break
                if entry[2].cancelled():
                    self.pending -= 1
                    continue
                if tokens + entry[1] > self.max_tokens:
                    carry = entry
                    break
                batch.append(entry)
                tokens += entry[1]
            try:
                results = await loop.run_in_executor(
                    self.executor, self.handler, [x[0] for x in batch]
                )
                if len(results) != len(batch):
                    raise ValueError("Teacher batch response count mismatch")
                for (_, _, future), result in zip(batch, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
            finally:
                self.pending -= len(batch)
            if stopping:
                break

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.task is not None:
            self.queue.put_nowait(None)
            await self.task
        self.executor.shutdown(wait=True)
