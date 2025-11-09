from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import math

import RPi.GPIO as GPIO
import logging

try:
    from .shared import MicroMove, SchedulerETA  # type: ignore
except Exception:
    from shared import MicroMove, SchedulerETA  # type: ignore


_STEP_SEQUENCE = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
]


class StepperAxis:
    def __init__(self, pins: Tuple[int, int, int, int], cw_positive: bool) -> None:
        self.pins = tuple(pins)
        self.cw_positive = cw_positive
        self._idx = 0
        # Initialize pins low
        for p in self.pins:
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, GPIO.LOW)

    def step_once(self, positive: bool) -> None:
        # Map positive (math +) to electrical CW/CCW
        cw = positive if self.cw_positive else (not positive)
        self._idx = (self._idx - 1) % 8 if cw else (self._idx + 1) % 8
        seq = _STEP_SEQUENCE[self._idx]
        for pin, val in zip(self.pins, seq):
            GPIO.output(pin, GPIO.HIGH if val else GPIO.LOW)


@dataclass
class SchedulerConfig:
    tick_hz: float
    s_max_steps_s: float
    a_max_steps_s2: float
    yaw_pins: Tuple[int, int, int, int]
    pitch_pins: Tuple[int, int, int, int]
    yaw_cw_positive: bool
    pitch_cw_positive: bool
    gpio_mode_bcm: bool = True


class SchedulerWorker:
    """
    Time-based step scheduler for two axes. Consumes MicroMove bursts and
    emits steps so both axes finish together in T.
    """

    def __init__(self, cfg: SchedulerConfig, move_queue, eta: SchedulerETA, logger: Optional[Any] = None) -> None:
        # Python logging setup (lightweight; only configure root if not already configured)
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
            )
        self._log = logging.getLogger("edja.scheduler")
        self.cfg = cfg
        self.queue = move_queue
        self.eta = eta
        self.logger = logger
        self._setup_gpio()
        self.yaw = StepperAxis(cfg.yaw_pins, cfg.yaw_cw_positive)
        self.pitch = StepperAxis(cfg.pitch_pins, cfg.pitch_cw_positive)
        print(f"[Scheduler] setup: tick_hz={cfg.tick_hz}, Smax={cfg.s_max_steps_s}, Amax={cfg.a_max_steps_s2}, BCM={cfg.gpio_mode_bcm}")
        self._log.info("setup tick_hz=%s Smax=%s Amax=%s BCM=%s", cfg.tick_hz, cfg.s_max_steps_s, cfg.a_max_steps_s2, cfg.gpio_mode_bcm)
        if self.logger is not None:
            try:
                self.logger.log("Scheduler", "setup", f"tick_hz={cfg.tick_hz} Smax={cfg.s_max_steps_s} Amax={cfg.a_max_steps_s2} BCM={cfg.gpio_mode_bcm}")
            except Exception:
                pass

    def _setup_gpio(self) -> None:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM if self.cfg.gpio_mode_bcm else GPIO.BOARD)

    def _solve_trapezoid(self, Nd: int, T: float) -> Tuple[float, float, float, int]:
        """
        Given dominant steps Nd and target duration T, return (S_pk, T_a, T_c, Nd_feasible)
        honoring S_max and A_max. If Nd exceeds feasible steps for T, returns scaled Nd_feasible.
        """
        A = self.cfg.a_max_steps_s2
        Smax = self.cfg.s_max_steps_s
        Nd_req = max(0, int(Nd))
        if Nd_req == 0 or T <= 0:
            return 0.0, 0.0, 0.0, 0

        # First try ideal trapezoid ignoring Smax
        # Discriminant for T_a from Nd = A*T_a*(T - T_a)
        disc = T * T - 4.0 * Nd_req / max(1e-9, A)
        if disc < 0.0:
            # Too many steps for given A,T -> treat as triangular at limit
            T_a = T / 2.0
        else:
            T_a = 0.5 * (T - disc ** 0.5)
        T_a = max(0.0, min(T / 2.0, T_a))
        S_pk = A * T_a

        # Enforce Smax
        if S_pk > Smax:
            S_pk = Smax
            T_a = S_pk / A

        T_c = max(0.0, T - 2.0 * T_a)

        # Feasible step area with these S_pk, T_a within fixed T:
        Nd_feasible = int(round(S_pk * (T_c + T_a)))

        return S_pk, T_a, T_c, Nd_feasible

    def _t_min_for_steps(self, Nd: int) -> float:
        """
        Minimal feasible duration to perform Nd dominant steps under Smax/Amax.
        Triangular if Nd <= (Smax^2)/A, otherwise trapezoidal.
        """
        if Nd <= 0:
            return 0.0
        A = self.cfg.a_max_steps_s2
        Smax = self.cfg.s_max_steps_s
        Nd_tri = (Smax * Smax) / max(1e-9, A)
        if Nd <= Nd_tri:
            return 2.0 * math.sqrt(Nd / max(1e-9, A))
        return (Nd / max(1e-9, Smax)) + (Smax / max(1e-9, A))

    def _run_burst(self, cmd: MicroMove) -> None:
        Nx = int(cmd.Nx)
        Ny = int(cmd.Ny)
        # Determine duration: use provided T or compute minimal feasible
        Nx_abs_init = abs(Nx)
        Ny_abs_init = abs(Ny)
        Nd_init = max(Nx_abs_init, Ny_abs_init)
        if cmd.T is None:
            T = self._t_min_for_steps(Nd_init)
        else:
            T = float(cmd.T)
        print(f"[Scheduler] run_burst: Nx={Nx} Ny={Ny} T={T:.4f}s")
        self._log.info("run_burst Nx=%s Ny=%s T=%.4f", Nx, Ny, T)
        if self.logger is not None:
            try:
                self.logger.log("Scheduler", "run_burst", f"Nx={Nx} Ny={Ny} T={T:.4f}")
            except Exception:
                pass

        Nx_abs = Nx_abs_init
        Ny_abs = Ny_abs_init
        Nd = max(Nx_abs, Ny_abs)
        if Nd == 0 or T <= 0:
            self.eta.write(0.0)
            return

        S_pk, T_a, T_c, Nd_feasible = self._solve_trapezoid(Nd, T)
        if Nd_feasible <= 0:
            self.eta.write(0.0)
            return
        # Scale vector if requested exceeds feasible
        if Nd > Nd_feasible:
            scale = Nd_feasible / float(Nd)
            Nx_abs = int(round(Nx_abs * scale))
            Ny_abs = int(round(Ny_abs * scale))
            Nd = max(Nx_abs, Ny_abs)
            # Recompute as Nd changed
            S_pk, T_a, T_c, _ = self._solve_trapezoid(Nd, T)

        # Per-axis scale factors
        kx = (Nx_abs / Nd) if Nd > 0 else 0.0
        ky = (Ny_abs / Nd) if Nd > 0 else 0.0
        dir_x_pos = Nx >= 0
        dir_y_pos = Ny >= 0

        # Tick loop
        tick_dt = 1.0 / max(1.0, self.cfg.tick_hz)
        steps_x_done = 0
        steps_y_done = 0
        acc_x = 0.0
        acc_y = 0.0
        start = time.perf_counter()
        now = start
        end = start + T

        while True:
            now = time.perf_counter()
            if now >= end:
                self.eta.write(0.0)
                break

            t = now - start
            # Dominant-axis rate S(t)
            if t < T_a:
                S_dom = self.cfg.a_max_steps_s2 * t
            elif t < (T_a + T_c):
                S_dom = S_pk
            elif t < (T_a + T_c + T_a):
                t_d = (T_a + T_c + T_a) - t
                S_dom = self.cfg.a_max_steps_s2 * max(0.0, t_d)
            else:
                S_dom = 0.0

            Sx = kx * S_dom
            Sy = ky * S_dom

            # Accumulate fractional steps
            acc_x += Sx * tick_dt
            acc_y += Sy * tick_dt

            # Emit steps when accumulators cross integers
            while acc_x >= 1.0 and steps_x_done < Nx_abs:
                self.yaw.step_once(dir_x_pos)
                steps_x_done += 1
                acc_x -= 1.0
            while acc_y >= 1.0 and steps_y_done < Ny_abs:
                self.pitch.step_once(dir_y_pos)
                steps_y_done += 1
                acc_y -= 1.0

            # ETA update
            self.eta.write(max(0.0, end - now))

            # Sleep until next tick
            next_tick = now + tick_dt
            to_sleep = max(0.0, next_tick - time.perf_counter())
            if to_sleep > 0:
                time.sleep(to_sleep)

        # If any remainder steps not completed (due to quantization), flush them at safe rate
        flush_x = max(0, Nx_abs - steps_x_done)
        flush_y = max(0, Ny_abs - steps_y_done)
        if flush_x > 0 or flush_y > 0:
            # Interleave remaining steps to keep axes visually in sync
            while flush_x > 0 or flush_y > 0:
                if flush_x > 0:
                    self.yaw.step_once(dir_x_pos)
                    flush_x -= 1
                if flush_y > 0:
                    self.pitch.step_once(dir_y_pos)
                    flush_y -= 1
                # small delay to avoid blasting coils too fast
                time.sleep(min(tick_dt, 0.002))
            self._log.info("flush_remaining steps completed (interleaved)")
            print("[Scheduler] flush_remaining steps completed (interleaved)")
        self.eta.write(0.0)
        if self.logger is not None:
            try:
                self.logger.log("Scheduler", "burst_done", f"Nx_done={steps_x_done}/{Nx_abs} Ny_done={steps_y_done}/{Ny_abs}")
            except Exception:
                pass

    def run_loop(self, stop_event: Optional[Any] = None) -> None:
        try:
            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        self._log.info("stop_event set; exiting run_loop")
                        break
                    try:
                        # Small timeout so we can update ETA to 0 when idle
                        cmd: MicroMove = self.queue.get(timeout=0.05)
                        t_label = cmd.T if cmd.T is not None else "auto"
                        print(f"[Scheduler] got cmd: Nx={cmd.Nx} Ny={cmd.Ny} T={t_label}", flush=True)
                        self._log.info("got cmd Nx=%s Ny=%s T=%s", cmd.Nx, cmd.Ny, str(t_label))
                    except Exception:
                        self.eta.write(0.0)
                        # Optional heartbeat to confirm loop is alive without spamming console:
                        # self._log.debug("idle (queue empty)")
                        continue
                    self._run_burst(cmd)
                    print("[Scheduler] burst finished", flush=True)
                    self._log.info("burst finished")
            finally:
                # Ensure outputs low
                for p in list(self.cfg.yaw_pins) + list(self.cfg.pitch_pins):
                    try:
                        GPIO.output(p, GPIO.LOW)
                    except Exception:
                        pass
                try:
                    GPIO.cleanup()
                except Exception:
                    pass
        except Exception as e:
            import traceback
            print("[Scheduler] crashed:", e)
            traceback.print_exc()
            self._log.error("Scheduler crashed: %s", e, exc_info=True)
            try:
                GPIO.cleanup()
            except Exception:
                pass


