from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    from .shared import LatestDetectionMailbox, MicroMove, PredState, SchedulerETA
except Exception:
    from shared import LatestDetectionMailbox, MicroMove, PredState, SchedulerETA


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
        # Track last issued detection seq to pace commands
        self._seq_last_issued = -1
        # Dynamic detection latency estimate (s)
        self._tau_det_s = 0.0
        # Detection fps tracking for burst sync
        self._det_dt_ema = 0.13  # seconds
        self._det_t_last = None
        self._last_det_conf = cfg.conf_min
        print(f"[Control] setup: tick_hz={cfg.tick_hz}, kp={cfg.kp}, kd={cfg.kd}, ki={cfg.ki}, deadband_steps={cfg.deadband_steps}")
        if self.logger is not None:
            try:
                self.logger.log("Control", "setup", f"tick_hz={cfg.tick_hz} kp={cfg.kp} kd={cfg.kd} ki={cfg.ki} deadband={cfg.deadband_steps}")
            except Exception:
                pass

    def _measurement_from_pixels(self, cx: float, cy: float, W: int, H: int) -> Tuple[float, float]:
        # Small-angle linear mapping around optical axis
        ex = ((cx - (W / 2.0)) / (W / 2.0)) * (self.cfg.fov_x_rad / 2.0)
        ey = ((cy - (H / 2.0)) / (H / 2.0)) * (self.cfg.fov_y_rad / 2.0)
        return ex, ey

    def _feasible_steps_for_T(self, T: float) -> int:
        """
        Maximum dominant steps achievable in duration T under S_max and A_max
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
        # Axis-specific deadband
        dbx = self.cfg.deadband_steps
        dby = max(self.cfg.deadband_steps, int(round(self.cfg.deadband_steps * 1.5)))
        if abs(Nx) < dbx and abs(Ny) < dby:
            return None
        Nd = max(abs(Nx), abs(Ny))
        Nd_max = self._feasible_steps_for_T(T)
        if Nd > Nd_max and Nd_max > 0:
            scale = Nd_max / float(Nd)
            Nx = int(round(Nx * scale))
            Ny = int(round(Ny * scale))
            # Enforce deadband post-scale
            if abs(Nx) < dbx and abs(Ny) < dby:
                return None
        # Let scheduler choose duration for this step vector
        return MicroMove(Nx=Nx, Ny=Ny, T=None)

    def run_loop(self, stop_event: Optional[Any] = None) -> None:
        tick_dt = 1.0 / max(1.0, self.cfg.tick_hz)
        tau0_s = self.cfg.tau0_ms / 1000.0
        T_burst_fallback = self.cfg.micro_move_T_ms / 1000.0

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                now = time.perf_counter()

                # Predict
                dt_x = max(1e-6, now - self.state_x.t)
                dt_y = max(1e-6, now - self.state_y.t)

                self.state_x.x += self.state_x.v * dt_x
                self.state_y.x += self.state_y.v * dt_y

                self.state_x.t = now
                self.state_y.t = now

                # Measurement update if new detection and confident
                det, seq = self.det_mailbox.read()
                if det is not None and seq != self._seq_last and det.conf >= self.cfg.conf_min:
                    # Estimate detection/inference latency
                    # Prefer inference completion time as reference
                    self._tau_det_s = max(0.0, now - float(det.t_inf))

                    # Update detection period EMA for sync
                    if self._det_t_last is not None:
                        dt_det = max(1e-3, float(det.t_cap) - float(self._det_t_last))
                        self._det_dt_ema = (self._det_dt_ema * 0.8) + (dt_det * 0.2)

                    self._det_t_last = float(det.t_cap)
                    zx, zy = self._measurement_from_pixels(det.cx, det.cy, det.W, det.H)

                    # Residuals
                    rx = zx - self.state_x.x
                    ry = zy - self.state_y.x

                    # α–β update
                    alpha_y = self.cfg.alpha * 0.8
                    beta_y = self.cfg.beta * 0.8

                    self.state_x.x += self.cfg.alpha * rx
                    self.state_y.x += alpha_y * ry
                    self.state_x.v += (self.cfg.beta / dt_x) * rx
                    self.state_y.v = getattr(self.state_y, "v", 0.0) + (beta_y / dt_y) * ry

                    self._seq_last = seq
                    self._last_det_conf = float(det.conf)

                # Loss-of-detection handling
                lost_timeout_s = 0.25
                det_age = float("inf") if self._det_t_last is None else (now - float(self._det_t_last))
                lost = det_age > lost_timeout_s
                if lost:
                    self.state_x.v = 0.0
                    self.state_y.v = 0.0
                    self._Ix = 0.0
                    self._Iy = 0.0

                # Latency-aware prediction
                eta_s = self.eta.read()
                # Base tau
                tau = tau0_s + self._tau_det_s + eta_s

                xpred = self.state_x.x + self.state_x.v * tau
                ypred = getattr(self.state_y, "x", 0.0) + getattr(self.state_y, "v", 0.0) * tau

                # PID with confidence-scaled gains
                dx = (self.state_x.x - self._x_prev) / max(1e-6, dt_x)
                dy = (getattr(self.state_y, "x", 0.0) - self._y_prev) / max(1e-6, dt_y)

                self._x_prev = self.state_x.x
                self._y_prev = getattr(self.state_y, "x", 0.0)

                # Map confidence -> scale
                conf_norm = (self._last_det_conf - self.cfg.conf_min) / max(1e-3, (1.0 - self.cfg.conf_min))
                conf_scale = max(0.5, min(1.0, conf_norm * 0.5 + 0.5))

                kp_eff = self.cfg.kp * conf_scale
                kd_eff = self.cfg.kd * conf_scale

                # Axis-specific gains
                kp_x, kd_x = kp_eff, kd_eff
                kp_y, kd_y = kp_eff * 0.75, kd_eff * 1.6

                ux = kp_x * xpred + self._Ix - kd_x * dx
                uy = kp_y * ypred + self._Iy - kd_y * dy

                # Integrator with simple clamping
                self._Ix += self.cfg.ki * xpred * tick_dt
                self._Iy += self.cfg.ki * ypred * tick_dt

                clamp = 0.5  # rad

                self._Ix = max(-clamp, min(clamp, self._Ix))
                self._Iy = max(-clamp, min(clamp, self._Iy))

                # Dynamic burst duration aligned to detection cadence
                T_burst = max(0.06, min(0.22, self._det_dt_ema if self._det_t_last is not None else T_burst_fallback))

                # Quantize and cap; set explicit burst duration aligned to detection fps
                mm = None if lost else self._quantize_and_cap(ux, uy, T_burst)

                # End-of-burst gating only to prevent ping-pong
                big_err = False

                if mm is not None:
                    try:
                        big_err = (abs(int(mm.Nx)) >= 20) or (abs(int(mm.Ny)) >= 20)
                    except Exception:
                        big_err = False
                should_issue = (eta_s <= 0.03) or big_err

                if mm is not None and should_issue:
                    # Use fixed T so scheduler paces to the frame cadence
                    mm = type(mm)(Nx=mm.Nx, Ny=mm.Ny, T=T_burst)
                    try:
                        self.move_queue.put_nowait(mm)
                        self._seq_last_issued = self._seq_last
                        t_label = mm.T if mm.T is not None else "auto"
                        print(f"[Control] enqueued MicroMove Nx={mm.Nx} Ny={mm.Ny} T={t_label}")
                        if self.logger is not None:
                            try:
                                self.logger.log("Control", "enqueue_mm", f"Nx={mm.Nx} Ny={mm.Ny} T={t_label}")
                            except Exception:
                                pass
                    except Exception:
                        # Queue full: replace oldest so latest command wins
                        try:
                            _ = self.move_queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            self.move_queue.put_nowait(mm)
                            self._seq_last_issued = self._seq_last
                            print(f"[Control] replaced queued MicroMove Nx={mm.Nx} Ny={mm.Ny} T={t_label}")
                            if self.logger is not None:
                                try:
                                    self.logger.log("Control", "replace_mm", f"Nx={mm.Nx} Ny={mm.Ny} T={t_label}")
                                except Exception:
                                    pass
                        except Exception:
                            if self.logger is not None:
                                try:
                                    self.logger.log("Control", "queue_full", "drop after replace attempt")
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


