from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    from .shared import LatestDetectionMailbox, MicroMove, PredState, SchedulerETA  # type: ignore
except Exception:
    from shared import LatestDetectionMailbox, MicroMove, PredState, SchedulerETA  # type: ignore


@dataclass
class ControlConfig:
    tick_hz: float
    alpha: float
    beta: float
    kp: float
    kd: float
    ki: float
    deadband_steps: int
    micro_move_T_ms: float
    tau0_ms: float
    # FOV in radians
    fov_x_rad: float
    fov_y_rad: float
    conf_min: float
    # Stepper and scheduler limits
    steps_per_rev: int
    s_max_steps_s: float
    a_max_steps_s2: float


class ControlWorker:
    """
    α–β filter + latency-aware prediction, PID, and quantization to MicroMove.
    """

    def __init__(
        self,
        cfg: ControlConfig,
        det_mailbox: LatestDetectionMailbox,
        move_queue,
        eta: SchedulerETA,
        logger: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.det_mailbox = det_mailbox
        self.move_queue = move_queue
        self.eta = eta
        self.logger = logger
        now = time.perf_counter()
        self.state_x = PredState(0.0, 0.0, now)
        self.state_y = PredState(0.0, 0.0, now)
        self._seq_last = -1
        self._Ix = 0.0
        self._Iy = 0.0
        self._x_prev = 0.0
        self._y_prev = 0.0
        self._steps_per_rad = float(cfg.steps_per_rev) / (2.0 * math.pi)
        print(f"[Control] setup: tick_hz={cfg.tick_hz}, kp={cfg.kp}, kd={cfg.kd}, ki={cfg.ki}, deadband_steps={cfg.deadband_steps}")
        if self.logger is not None:
            try:
                self.logger.log("Control", "setup", f"tick_hz={cfg.tick_hz} kp={cfg.kp} kd={cfg.kd} ki={cfg.ki} deadband={cfg.deadband_steps}")
            except Exception:
                pass

    def _measurement_from_pixels(self, cx: float, cy: float, W: int, H: int) -> Tuple[float, float]:
        # Small-angle linear mapping around optical axis
        ex = ((cx - (W / 2.0)) / (W / 2.0)) * (self.cfg.fov_x_rad / 2.0)
        ey = -((cy - (H / 2.0)) / (H / 2.0)) * (self.cfg.fov_y_rad / 2.0)
        return ex, ey

    def _feasible_steps_for_T(self, T: float) -> int:
        """
        Maximum dominant steps achievable in duration T under S_max and A_max.
        """
        A = self.cfg.a_max_steps_s2
        Smax = self.cfg.s_max_steps_s
        if T <= 0:
            return 0
        # If cannot reach Smax within T (triangular)
        if T < (2.0 * Smax / A):
            steps_max = A * T * T / 4.0
        else:
            T_a = Smax / A
            steps_max = Smax * (T - T_a)
        return max(0, int(math.floor(steps_max)))

    def _quantize_and_cap(self, ux_rad: float, uy_rad: float, T: float) -> Optional[MicroMove]:
        Nx = int(round(ux_rad * self._steps_per_rad))
        Ny = int(round(uy_rad * self._steps_per_rad))
        if abs(Nx) < self.cfg.deadband_steps and abs(Ny) < self.cfg.deadband_steps:
            return None
        Nd = max(abs(Nx), abs(Ny))
        Nd_max = self._feasible_steps_for_T(T)
        if Nd > Nd_max and Nd_max > 0:
            scale = Nd_max / float(Nd)
            Nx = int(round(Nx * scale))
            Ny = int(round(Ny * scale))
            # Enforce deadband post-scale
            if abs(Nx) < self.cfg.deadband_steps and abs(Ny) < self.cfg.deadband_steps:
                return None
        # Let scheduler choose duration for this step vector
        return MicroMove(Nx=Nx, Ny=Ny, T=None)

    def run_loop(self, stop_event: Optional[Any] = None) -> None:
        tick_dt = 1.0 / max(1.0, self.cfg.tick_hz)
        tau0_s = self.cfg.tau0_ms / 1000.0
        T_burst = self.cfg.micro_move_T_ms / 1000.0

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                now = time.perf_counter()
                # Predict to now
                dt_x = max(1e-6, now - self.state_x.t)
                dt_y = max(1e-6, now - self.state_y.t)
                self.state_x.x += self.state_x.v * dt_x
                self.state_y.x += self.state_y.v * dt_y  # type: ignore[attr-defined]
                self.state_x.t = now
                self.state_y.t = now

                # Measurement update if new detection and confident
                det, seq = self.det_mailbox.read()
                if det is not None and seq != self._seq_last and det.conf >= self.cfg.conf_min:
                    zx, zy = self._measurement_from_pixels(det.cx, det.cy, det.W, det.H)
                    # Residuals
                    rx = zx - self.state_x.x
                    ry = zy - self.state_y.x  # type: ignore[attr-defined]
                    # α–β update
                    self.state_x.x += self.cfg.alpha * rx
                    self.state_y.x += self.cfg.alpha * ry  # type: ignore[attr-defined]
                    self.state_x.v += (self.cfg.beta / dt_x) * rx
                    self.state_y.v = getattr(self.state_y, "v", 0.0) + (self.cfg.beta / dt_y) * ry  # ensure v exists
                    self._seq_last = seq

                # Latency-aware prediction
                eta_s = self.eta.read()
                tau = tau0_s + eta_s
                xpred = self.state_x.x + self.state_x.v * tau
                ypred = getattr(self.state_y, "x", 0.0) + getattr(self.state_y, "v", 0.0) * tau

                # PID (D on measurement derivative)
                dx = (self.state_x.x - self._x_prev) / max(1e-6, dt_x)
                dy = (getattr(self.state_y, "x", 0.0) - self._y_prev) / max(1e-6, dt_y)
                self._x_prev = self.state_x.x
                self._y_prev = getattr(self.state_y, "x", 0.0)

                ux = self.cfg.kp * xpred + self._Ix - self.cfg.kd * dx
                uy = self.cfg.kp * ypred + self._Iy - self.cfg.kd * dy
                # Integrator with simple clamping
                self._Ix += self.cfg.ki * xpred * tick_dt
                self._Iy += self.cfg.ki * ypred * tick_dt
                clamp = 0.5  # rad
                self._Ix = max(-clamp, min(clamp, self._Ix))
                self._Iy = max(-clamp, min(clamp, self._Iy))

                # Quantize and cap
                mm = self._quantize_and_cap(ux, uy, T_burst)
                if mm is not None:
                    try:
                        self.move_queue.put_nowait(mm)
                        t_label = mm.T if mm.T is not None else "auto"
                        print(f"[Control] enqueued MicroMove Nx={mm.Nx} Ny={mm.Ny} T={t_label}")
                        if self.logger is not None:
                            try:
                                self.logger.log("Control", "enqueue_mm", f"Nx={mm.Nx} Ny={mm.Ny} T={t_label}")
                            except Exception:
                                pass
                    except Exception:
                        # Queue full: drop this command (latest will replace soon)
                        print("[Control] move_queue full, dropping MicroMove")
                        if self.logger is not None:
                            try:
                                self.logger.log("Control", "queue_full", "drop MicroMove")
                            except Exception:
                                pass

                # Sleep to maintain tick rate
                next_tick = time.perf_counter() + tick_dt
                to_sleep = max(0.0, next_tick - time.perf_counter())
                if to_sleep > 0:
                    time.sleep(to_sleep)
        finally:
            # Nothing to cleanup
            pass


