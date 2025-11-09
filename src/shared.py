from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Detection:
    t_cap: float
    t_inf: float
    W: int
    H: int
    cx: float
    cy: float
    conf: float
    cls: int


@dataclass
class PredState:
    x: float  # angle error (rad)
    v: float  # angular velocity (rad/s)
    t: float  # last update time (s, monotonic)


@dataclass
class MicroMove:
    Nx: int   # steps for yaw (right +)
    Ny: int   # steps for pitch (up +)
    T: float  # planned duration (s)


class LatestDetectionMailbox:
    """
    Single-slot mailbox for the most recent detection with a sequence number.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._det: Optional[Detection] = None
        self._seq: int = 0

    def write(self, det: Detection) -> int:
        with self._lock:
            self._det = det
            self._seq += 1
            return self._seq

    def read(self) -> Tuple[Optional[Detection], int]:
        with self._lock:
            return self._det, self._seq


class SchedulerETA:
    """
    Lock-protected ETA value (seconds remaining in current burst).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._eta_s: float = 0.0

    def write(self, eta_s: float) -> None:
        with self._lock:
            self._eta_s = eta_s

    def read(self) -> float:
        with self._lock:
            return self._eta_s


def create_move_queue(maxsize: int = 4) -> "queue.Queue[MicroMove]":
    return queue.Queue(maxsize=maxsize)


