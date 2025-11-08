from dataclasses import dataclass, field
import queue
import threading
from typing import Optional
import time

@dataclass
class Detection:
    t_capture: float
    cx: float
    cy: float
    conf: float

@dataclass
class MoveCommand:
    steps_x: int
    steps_y: int
    target_time: float

class LatestDetectionMailbox:
    def __init__(self):
        self._lock = threading.Lock()
        self._det: Optional[Detection] = None
        self._seq = 0

    def write(self, det: Detection):
        with self._lock:
            self._det = det
            self._seq += 1

    def read(self):
        with self._lock:
            return self._det, self._seq

class MoveQueue:
    def __init__(self, maxsize=64):
        self._q = queue.Queue(maxsize)

    def enqueue(self, cmd: MoveCommand):
        try:
            self._q.put_nowait(cmd)
        except queue.Full:
            # drop oldest or skip
            _ = self._q.get_nowait()
            self._q.put_nowait(cmd)

    def dequeue(self, timeout=0.01) -> Optional[MoveCommand]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None