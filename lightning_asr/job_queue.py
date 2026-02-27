from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Job[T]:
    job_id: str
    payload: T


class JobQueue[T]:
    def __init__(self, *, max_queue_size: int = 1000) -> None:
        self._q: queue.Queue[Job[T]] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._processor: Callable[[Job[T]], None] | None = None

    def start(self, *, processor: Callable[[Job[T]], None], daemon: bool = True) -> None:
        if self._thread is not None:
            return
        self._processor = processor
        self._thread = threading.Thread(target=self._run, daemon=daemon)
        self._thread.start()

    def submit(self, payload: T) -> str:
        job_id = uuid.uuid4().hex
        self._q.put(Job(job_id=job_id, payload=payload))
        return job_id

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._processor is not None:
                    self._processor(job)
            finally:
                self._q.task_done()
