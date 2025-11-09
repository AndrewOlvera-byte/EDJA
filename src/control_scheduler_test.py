from __future__ import annotations

import sys
import time
import threading
import queue
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


class SpyQueue:
    """
    Wraps a queue to capture the last enqueued MicroMove while delegating to the real queue.
    """

    def __init__(self, q):
        self._q = q
        self.last_mm = None
        self.last_put_ts = None

    def put_nowait(self, item):
        self.last_mm = item
        try:
            self.last_put_ts = time.perf_counter()
        except Exception:
            self.last_put_ts = None
        return self._q.put_nowait(item)

    def get(self, timeout=None):
        return self._q.get(timeout=timeout)


import time


class TimedStepper(SW.StepperAxis): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times = []

    def step_once(self, positive: bool) -> None:
        super().step_once(positive)
        try:
            self.times.append(time.perf_counter())
        except Exception:
            pass


SW.StepperAxis = TimedStepper 


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

def _t_min_for_steps(Nd: int, Smax: float, A: float) -> float:
    if Nd <= 0:
        return 0.0
    Nd_tri = (Smax * Smax) / max(1e-9, A)
    if Nd <= Nd_tri:
        return 2.0 * (Nd / max(1e-9, A)) ** 0.5
    return (Nd / max(1e-9, Smax)) + (Smax / max(1e-9, A))


def main() -> None:
    # Shared containers
    mailbox = LatestDetectionMailbox()
    move_queue = SpyQueue(create_move_queue(maxsize=16))
    eta = SchedulerETA()

    # Scheduler configuration (mirrors defaults)
    sched_cfg = SW.SchedulerConfig(
        tick_hz=1500.0,
        s_max_steps_s=500.0,
        a_max_steps_s2=4000.0,
        yaw_pins=(17, 18, 27, 22),
        pitch_pins=(23, 24, 25, 5),
        yaw_cw_positive=True,
        pitch_cw_positive=False,
        gpio_mode_bcm=True,
    )
    sched = SW.SchedulerWorker(cfg=sched_cfg, move_queue=move_queue, eta=eta, logger=None)
    # Start scheduler loop in background
    sched_stop = threading.Event()
    t_sched = threading.Thread(target=sched.run_loop, kwargs={"stop_event": sched_stop}, daemon=True)
    t_sched.start()
    print("[Script] Scheduler thread started")

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
    ctrl_stop = threading.Event()
    t_control = threading.Thread(target=ctrl.run_loop, kwargs={"stop_event": ctrl_stop}, daemon=True)
    t_control.start()
    print("[Script] Control thread started")

    # Create one fake detection that is off-center in both axes
    W, H = 640, 640
    cx_px, cy_px = 0.9 * W, 0.2 * H  # right and above center => ex > 0, ey > 0
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
    print(f"[Script] Wrote fake detection: cx={cx_px:.1f}, cy={cy_px:.1f}, conf={det.conf}")

    # Wait for Control to enqueue a MicroMove (do not consume from the queue here)
    deadline = time.perf_counter() + 1.0
    while move_queue.last_mm is None and time.perf_counter() < deadline:
        time.sleep(0.01)
    mm = move_queue.last_mm
    assert mm is not None, "Control did not enqueue a MicroMove within 1s"

    # Stop control immediately to avoid additional bursts getting enqueued/consumed
    ctrl_stop.set()
    try:
        t_control.join(timeout=1.0)
    except Exception:
        pass

    # Scheduler will auto-compute T for this vector
    print(f"[Script] MicroMove from control: Nx={mm.Nx} Ny={mm.Ny} T={'auto' if getattr(mm, 'T', None) is None else mm.T}")

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

    # Predict scheduler duration for this step vector (auto-T)
    Nd_final = max(abs(mm.Nx), abs(mm.Ny))
    T_pred = _t_min_for_steps(Nd_final, ctrl_cfg.s_max_steps_s, ctrl_cfg.a_max_steps_s2)

    # Wait for scheduler to become active (ETA > 0), then complete (ETA back to 0)
    saw_active = False
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < (T_pred + 1.0):
        e = eta.read()
        if e > 0.0:
            saw_active = True
        time.sleep(0.01)
    time.sleep(0.1)  # small tail window
    assert saw_active, "Scheduler never became active (ETA stayed 0)"
    assert eta.read() == 0.0, "ETA not zero after burst completion"
    print("[Script] Scheduler burst completed (ETA returned to 0)")


    if mm.Nx != 0 and Nx_cap != 0:
        print(f"[Script] Direction check yaw OK: sign={('+' if mm.Nx>0 else '-')} target={('+' if Nx_cap>0 else '-')}")
    if mm.Ny != 0 and Ny_cap != 0:
        print(f"[Script] Direction check pitch OK: sign={('+' if mm.Ny>0 else '-')} target={('+' if Ny_cap>0 else '-')}")

    # Timing window using timestamps recorded after the control's last put
    y_times = getattr(sched.yaw, "times", []) or []
    p_times = getattr(sched.pitch, "times", []) or []
    t_mark = getattr(move_queue, "last_put_ts", None)
    if t_mark is not None:
        y_times = [t for t in y_times if t >= t_mark]
        p_times = [t for t in p_times if t >= t_mark]
    if y_times and p_times:
        win_start = min(y_times[0], p_times[0])
        win_end   = max(y_times[-1], p_times[-1])
        dur = win_end - win_start
        assert abs(dur - T_pred) <= max(0.05, 0.3 * T_pred), f"Burst duration {dur:.3f}s deviates from T_pred={T_pred:.3f}s"

    # Cleanup
    sched_stop.set()
    try:
        t_sched.join(timeout=1.0)
    except Exception:
        pass
    try:
        SW.GPIO.cleanup()
    except Exception:
        pass

    print("[Script] OK: Control enqueued, Scheduler consumed and actuated XY concurrently.")


if __name__ == "__main__":
    main()


