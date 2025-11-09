from __future__ import annotations

import sys
import time
from pathlib import Path

import RPi.GPIO as GPIO
import threading
import traceback
import sys as _sys
import faulthandler

# Enable fault handler for hard crashes
faulthandler.enable()

# Ensure thread exceptions are printed
def _thread_excepthook(args):
    print(f"[Thread-EXC] {args.thread.name} crashed: {args.exc_value}", flush=True)
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = _thread_excepthook

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared import MicroMove, SchedulerETA, create_move_queue
from scheduler_worker import SchedulerConfig, SchedulerWorker


def main() -> None:
    q = create_move_queue(maxsize=4)
    eta = SchedulerETA()
    sched = SchedulerWorker(
        cfg=SchedulerConfig(
            tick_hz=1500.0,
            s_max_steps_s=500.0,
            a_max_steps_s2=4000.0,
            yaw_pins=(17, 18, 27, 22),
            pitch_pins=(23, 24, 25, 5),
            yaw_cw_positive=True,
            pitch_cw_positive=False,
            gpio_mode_bcm=True,
        ),
        move_queue=q,
        eta=eta,
    )
    stop = False

    def run_sched():
        try:
            sched.run_loop()
        except Exception as e:
            print("[Scheduler] crashed:", e, flush=True)
            traceback.print_exc()

    t = threading.Thread(target=run_sched, daemon=True)
    t.start()

    # Scripted burst sequence
    try:
        moves = [
            MicroMove(Nx=200, Ny=0, T=0.5),
            MicroMove(Nx=0, Ny=200, T=0.5),
            MicroMove(Nx=-200, Ny=0, T=0.5),
            MicroMove(Nx=0, Ny=-200, T=0.5),
        ]
        for m in moves:
            print("Enqueue:", m, flush=True)
            q.put(m)
            # Poll ETA
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < m.T + 0.2:
                print(f"ETA: {eta.read():.3f}s", end="\r", flush=True)
                time.sleep(0.05)
            print()
        time.sleep(0.5)
        # Give the scheduler thread a moment to print any crash info
        t.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            GPIO.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()


