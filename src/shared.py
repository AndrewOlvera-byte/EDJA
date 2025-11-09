from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
import csv
import os
import time


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
    T: Optional[float] = None  # planned duration (s); None => scheduler auto-computes


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



# Minimal CSV logger for cross-thread event logging
class CsvEventLogger:
    def __init__(self, csv_path: str) -> None:
        self._lock = threading.Lock()
        self._csv_path = csv_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._csv_path) or ".", exist_ok=True)
        # Initialize with header if new
        need_header = not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0
        if need_header:
            with open(self._csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ts_iso", "thread", "event", "message"])

    def log(self, thread: str, event: str, message: str) -> None:
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        with self._lock:
            try:
                with open(self._csv_path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([ts_iso, thread, event, message])
            except Exception:
                # Best-effort logging; do not raise
                pass
