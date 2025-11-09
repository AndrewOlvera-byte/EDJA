import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import RPi.GPIO as GPIO

# Allow running as `python src/App.py` by adding this directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared import LatestDetectionMailbox, SchedulerETA, create_move_queue  # noqa: E402
from vision_worker import VisionConfig, VisionWorker  # noqa: E402
from scheduler_worker import SchedulerConfig, SchedulerWorker  # noqa: E402
from control_worker import ControlConfig, ControlWorker  # noqa: E402
from shared import CsvEventLogger  # noqa: E402


class App:
    def __init__(self, config_path: str = "config/config.yaml"):
        # Hardcoded configuration (no YAML)
        self.cfg = self._default_config()

        # Shared containers
        self.mailbox = LatestDetectionMailbox()
        self.move_queue = create_move_queue(maxsize=1)
        self.eta = SchedulerETA()
        # Logger
        self.logger = None
        try:
            log_cfg = self.cfg.get("logging", {})
            if bool(log_cfg.get("enabled", True)):
                from pathlib import Path as _P
                csv_path = str(self._resolve_path(log_cfg.get("csv_path", "logs/edja_events.csv")))
                self.logger = CsvEventLogger(csv_path)
                print(f"[App] CSV logging enabled at {csv_path}")
        except Exception:
            self.logger = None

        # Workers
        self.vision = VisionWorker(
            cfg=VisionConfig(
                onnx_path=str(self._resolve_path(self.cfg["vision"]["onnx_path"])),
                providers=tuple(self.cfg["vision"].get("providers", ["CPUExecutionProvider"])),
                input_size=tuple(self.cfg["vision"].get("input_size", [640, 640])),
                conf_min=float(self.cfg["vision"].get("conf_min", 0.25)),
                target_cls=int(self.cfg["vision"].get("target_cls", 0)),
            ),
            mailbox=self.mailbox,
            logger=self.logger,
        )

        self.scheduler = SchedulerWorker(
            cfg=SchedulerConfig(
                tick_hz=float(self.cfg["scheduler"]["tick_hz"]),
                s_max_steps_s=float(self.cfg["scheduler"]["s_max_steps_s"]),
                a_max_steps_s2=float(self.cfg["scheduler"]["a_max_steps_s2"]),
                yaw_pins=tuple(self.cfg["stepper"]["yaw_pins"]),
                pitch_pins=tuple(self.cfg["stepper"]["pitch_pins"]),
                yaw_cw_positive=bool(self.cfg["stepper"]["yaw_cw_positive"]),
                pitch_cw_positive=bool(self.cfg["stepper"]["pitch_cw_positive"]),
                gpio_mode_bcm=(self.cfg.get("gpio", {}).get("mode", "BCM").upper() == "BCM"),
            ),
            move_queue=self.move_queue,
            eta=self.eta,
            logger=self.logger,
        )

        # Control config requires FOV radians and limits
        fov_x_deg = float(self.cfg["vision"]["fov_deg"]["x"])
        fov_y_deg = float(self.cfg["vision"]["fov_deg"]["y"])
        self.control = ControlWorker(
            cfg=ControlConfig(
                tick_hz=float(self.cfg["control"]["tick_hz"]),
                alpha=float(self.cfg["control"]["alpha"]),
                beta=float(self.cfg["control"]["beta"]),
                kp=float(self.cfg["control"]["kp"]),
                kd=float(self.cfg["control"]["kd"]),
                ki=float(self.cfg["control"]["ki"]),
                deadband_steps=int(self.cfg["control"]["deadband_steps"]),
                micro_move_T_ms=float(self.cfg["control"]["micro_move_T_ms"]),
                tau0_ms=float(self.cfg["control"]["tau0_ms"]),
                fov_x_rad=fov_x_deg * 3.141592653589793 / 180.0,
                fov_y_rad=fov_y_deg * 3.141592653589793 / 180.0,
                conf_min=float(self.cfg["vision"]["conf_min"]),
                steps_per_rev=int(self.cfg["stepper"]["steps_per_rev"]),
                s_max_steps_s=float(self.cfg["scheduler"]["s_max_steps_s"]),
                a_max_steps_s2=float(self.cfg["scheduler"]["a_max_steps_s2"]),
            ),
            det_mailbox=self.mailbox,
            move_queue=self.move_queue,
            eta=self.eta,
            logger=self.logger,
        )

        self._stop_event = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._run_with_logs,
                args=("Vision", self.vision.run_loop),
                kwargs={
                    "stop_event": self._stop_event,
                    # Show window by default on Pi for debugging
                    "fps_overlay": bool(self.cfg.get("vision", {}).get("show_window", True)),
                },
                daemon=True,
            ),
            threading.Thread(
                target=self._run_with_logs,
                args=("Scheduler", self.scheduler.run_loop),
                kwargs={"stop_event": self._stop_event},
                daemon=True,
            ),
            threading.Thread(
                target=self._run_with_logs,
                args=("Control", self.control.run_loop),
                kwargs={"stop_event": self._stop_event},
                daemon=True,
            ),
        ]

    def _run_with_logs(self, name: str, fn, **kwargs) -> None:
        print(f"[{name}] starting")
        if self.logger is not None:
            try:
                self.logger.log(name, "start", f"kwargs={list(kwargs.keys())}")
            except Exception:
                pass
        try:
            fn(**kwargs)
            print(f"[{name}] exited")
            if self.logger is not None:
                try:
                    self.logger.log(name, "exit", "normal")
                except Exception:
                    pass
        except Exception as e:
            import traceback
            print(f"[{name}] crashed: {e}")
            traceback.print_exc()
            if self.logger is not None:
                try:
                    self.logger.log(name, "crash", f"{type(e).__name__}: {e}")
                except Exception:
                    pass

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = (BASE_DIR / ".." / path).resolve()
        return path

    def _default_config(self) -> Dict[str, Any]:
        return {
            "vision": {
                "onnx_path": "src/yolo11n.onnx",
                "providers": ["CPUExecutionProvider"],
                "input_size": [640, 640],
                "conf_min": 0.10,
                "target_cls": 0,
                "fov_deg": {"x": 54.0, "y": 41.0},
                "show_window": True,
            },
            "control": {
                # Pace control near camera cadence; add damping and velocity estimate
                "tick_hz": 90,
                "alpha": 0.6,
                "beta": 0.08,
                "kp": 0.8,
                "kd": 0.22,
                "ki": 0.0,
                "deadband_steps": 8,
                # Burst duration will be dynamically matched to detection fps; this is fallback
                "micro_move_T_ms": 120.0,
                "tau0_ms": 60.0,
            },
            "scheduler": {
                # Lower max speed/accel for smoother movement
                "tick_hz": 1200,
                "s_max_steps_s": 140,
                "a_max_steps_s2": 700,
            },
            "stepper": {
                "steps_per_rev": 4096,
                # Use the pin mapping previously in config.yaml
                "yaw_pins": [17, 18, 27, 22],
                "pitch_pins": [23, 24, 25, 5],
                "yaw_cw_positive": True,
                "pitch_cw_positive": False,
            },
            "gpio": {"mode": "BCM"},
            "logging": {
                "enabled": True,
                "csv_path": "logs/edja_events.csv",
            },
        }

    def start(self) -> None:
        try:
            for t in self._threads:
                t.start()
            # Keep main thread alive; respond to Ctrl+C
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            for t in self._threads:
                try:
                    t.join(timeout=2.0)
                except Exception:
                    pass
            try:
                GPIO.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    app = App()
    app.start()
    