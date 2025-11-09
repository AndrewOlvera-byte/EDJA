from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

# Allow running as `python src/control_scheduler_test.py`
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared import (  # noqa: E402
    LatestDetectionMailbox,
    SchedulerETA,
    create_move_queue,
    Detection,
)
import scheduler_worker as SW  # noqa: E402
import control_worker as CW  # noqa: E402


# Optional GPIO shim for non-Pi environments
try:
    import RPi.GPIO as _GPIO  # type: ignore  # noqa: F401
except Exception:
    class _FakeGPIO:
        BCM = 0
        BOARD = 1
        OUT = 0
        LOW = 0
        HIGH = 1

        def setwarnings(self, *_args, **_kwargs):
            pass

        def setmode(self, *_args, **_kwargs):
            pass

        def setup(self, *_args, **_kwargs):
            pass

        def output(self, *_args, **_kwargs):
            pass

        def cleanup(self, *_args, **_kwargs):
            pass

    # Use shim inside scheduler module (where GPIO is referenced)
    SW.GPIO = _FakeGPIO()  # type: ignore[attr-defined]


class CountingAxis:
    """
    Test double for SW.StepperAxis that records step count and timings.
    """

    def __init__(self, pins, cw_positive):
        self.pins = tuple(pins)
        self.cw_positive = bool(cw_positive)
        self.steps = 0
        self.signs = []  # math-positive direction used
        self.times = []

    def step_once(self, positive: bool) -> None:
        self.steps += 1
        self.signs.append(bool(positive))
        self.times.append(time.perf_counter())


def _measurement_from_pixels(cx: float, cy: float, W: int, H: int, fov_x_rad: float, fov_y_rad: float):
    ex = ((cx - (W / 2.0)) / (W / 2.0)) * (fov_x_rad / 2.0)
    ey = -((cy - (H / 2.0)) / (H / 2.0)) * (fov_y_rad / 2.0)
    return ex, ey


def _feasible_steps_for_T(T: float, Smax: float, A: float) -> int:
    if T <= 0:
        return 0
    if T < (2.0 * Smax / A):  # triangular case
        return max(0, int((A * T * T) // 4))
    Ta = Smax / A
    return max(0, int(Smax * (T - Ta)))


def main() -> None:
    # Shared containers
    mailbox = LatestDetectionMailbox()
    move_queue = create_move_queue(maxsize=4)
    eta = SchedulerETA()

    # Monkeypatch StepperAxis to our counting axis (before creating SchedulerWorker)
    SW.StepperAxis = CountingAxis  # type: ignore[attr-defined]

    # Scheduler configuration (mirrors defaults)
    sched_cfg = SW.SchedulerConfig(
        tick_hz=1500.0,
        s_max_steps_s=500.0,
        a_max_steps_s2=4000.0,
        yaw_pins=(23, 24, 25, 5),
        pitch_pins=(17, 18, 27, 22),
        yaw_cw_positive=True,
        pitch_cw_positive=False,
        gpio_mode_bcm=True,
    )
    sched = SW.SchedulerWorker(cfg=sched_cfg, move_queue=move_queue, eta=eta, logger=None)

    # Control configuration tuned for determinism in this script
    fov_x_deg, fov_y_deg = 54.0, 41.0
    ctrl_cfg = CW.ControlConfig(
        tick_hz=150.0,
        alpha=1.0,   # use measurement directly
        beta=0.0,
        kp=1.8,
        kd=0.0,
        ki=0.0,
        deadband_steps=1,
        micro_move_T_ms=200.0,  # longer burst for clearer step timing
        tau0_ms=0.0,            # remove extra prediction for this check
        fov_x_rad=fov_x_deg * 3.141592653589793 / 180.0,
        fov_y_rad=fov_y_deg * 3.141592653589793 / 180.0,
        conf_min=0.25,
        steps_per_rev=4096,
        s_max_steps_s=sched_cfg.s_max_steps_s,
        a_max_steps_s2=sched_cfg.a_max_steps_s2,
    )
    ctrl = CW.ControlWorker(cfg=ctrl_cfg, det_mailbox=mailbox, move_queue=move_queue, eta=eta, logger=None)

    # Run control in its own thread so it can enqueue a MicroMove
    stop = threading.Event()
    t_control = threading.Thread(target=ctrl.run_loop, kwargs={"stop_event": stop}, daemon=True)
    t_control.start()

    # Create one fake detection that is off-center in both axes
    W, H = 640, 640
    cx_px, cy_px = 0.75 * W, 0.35 * H  # right and above center => ex > 0, ey > 0
    now = time.perf_counter()
    det = Detection(
        t_cap=now,
        t_inf=now,
        W=W,
        H=H,
        cx=cx_px,
        cy=cy_px,
        conf=0.9,
        cls=0,
    )
    mailbox.write(det)

    # Wait for Control to enqueue a MicroMove
    mm = None
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        try:
            mm = move_queue.get(timeout=0.05)
            break
        except Exception:
            pass

    assert mm is not None, "Control did not enqueue a MicroMove within 1s"
    print(f"[Script] MicroMove from control: Nx={mm.Nx} Ny={mm.Ny} T={mm.T:.3f}s")

    # Validate direction and approximate magnitude against the same logic as Control
    ex, ey = _measurement_from_pixels(cx_px, cy_px, W, H, ctrl_cfg.fov_x_rad, ctrl_cfg.fov_y_rad)
    steps_per_rad = ctrl_cfg.steps_per_rev / (2.0 * 3.141592653589793)
    Nx_ideal = int(round(ctrl_cfg.kp * ex * steps_per_rad))
    Ny_ideal = int(round(ctrl_cfg.kp * ey * steps_per_rad))
    Nd = max(abs(Nx_ideal), abs(Ny_ideal))
    T = ctrl_cfg.micro_move_T_ms / 1000.0
    Nd_max = _feasible_steps_for_T(T, ctrl_cfg.s_max_steps_s, ctrl_cfg.a_max_steps_s2)
    if Nd > Nd_max and Nd_max > 0:
        scale = Nd_max / float(Nd)
        Nx_cap = int(round(Nx_ideal * scale))
        Ny_cap = int(round(Ny_ideal * scale))
    else:
        Nx_cap, Ny_cap = Nx_ideal, Ny_ideal

    # Direction checks (unless axis gets zeroed by deadband)
    if Nx_cap != 0 and mm.Nx != 0:
        assert (mm.Nx > 0) == (Nx_cap > 0), "Yaw sign does not point toward center"
    if Ny_cap != 0 and mm.Ny != 0:
        assert (mm.Ny > 0) == (Ny_cap > 0), "Pitch sign does not point toward center"

    # Magnitude checks within small tolerance (allow rounding error)
    if Nx_cap != 0 and mm.Nx != 0:
        assert abs(mm.Nx - Nx_cap) <= 2, f"Yaw magnitude off: got {mm.Nx}, expect ~{Nx_cap}"
    if Ny_cap != 0 and mm.Ny != 0:
        assert abs(mm.Ny - Ny_cap) <= 2, f"Pitch magnitude off: got {mm.Ny}, expect ~{Ny_cap}"

    # Ratio (proportionality) check if both are non-zero
    if all(v != 0 for v in (mm.Nx, mm.Ny, Nx_cap, Ny_cap)):
        ratio = abs(mm.Nx / mm.Ny)
        expect_ratio = abs(Nx_cap / Ny_cap)
        assert abs(ratio - expect_ratio) <= 0.25 * max(1.0, expect_ratio), "XY ratio deviates more than tolerance"

    # Execute a single burst with the scheduler and verify simultaneity
    t0 = time.perf_counter()
    sched._run_burst(mm)  # type: ignore[attr-defined]  # call internal for deterministic single-burst run
    t1 = time.perf_counter()

    # Read counters from monkeypatched axes
    yaw_axis = sched.yaw  # type: ignore[attr-defined]
    pitch_axis = sched.pitch  # type: ignore[attr-defined]
    assert yaw_axis.steps == abs(mm.Nx), f"Yaw steps {yaw_axis.steps} != {abs(mm.Nx)}"
    assert pitch_axis.steps == abs(mm.Ny), f"Pitch steps {pitch_axis.steps} != {abs(mm.Ny)}"

    # Simultaneity window and duration close to planned T
    if yaw_axis.times and pitch_axis.times:
        win_start = min(yaw_axis.times[0], pitch_axis.times[0])
        win_end = max(yaw_axis.times[-1], pitch_axis.times[-1])
        dur = win_end - win_start
        # Host timing jitter tolerance
        assert abs(dur - T) <= max(0.05, 0.3 * T), f"Burst duration {dur:.3f}s deviates from T={T:.3f}s"

    # ETA returns to 0 at the end of burst
    assert eta.read() == 0.0, "ETA not zero after burst completion"

    # Cleanup
    stop.set()
    try:
        t_control.join(timeout=1.0)
    except Exception:
        pass
    try:
        SW.GPIO.cleanup()  # type: ignore[attr-defined]
    except Exception:
        pass

    print("[Script] OK: Control direction/magnitude correct and Scheduler ran XY concurrently.")


if __name__ == "__main__":
    main()


