import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import RPi.GPIO as GPIO
import yaml

# Allow running as `python src/App.py` by adding this directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared import LatestDetectionMailbox, SchedulerETA, create_move_queue  # noqa: E402
from vision_worker import VisionConfig, VisionWorker  # noqa: E402
from scheduler_worker import SchedulerConfig, SchedulerWorker  # noqa: E402
from control_worker import ControlConfig, ControlWorker  # noqa: E402


class App:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = str(config_path)
        self.cfg = self._load_config(self.config_path)

        # Shared containers
        self.mailbox = LatestDetectionMailbox()
        self.move_queue = create_move_queue(maxsize=4)
        self.eta = SchedulerETA()

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
        )

        self._stop_event = threading.Event()
        self._threads = [
            threading.Thread(target=self.vision.run_loop, kwargs={"stop_event": self._stop_event, "fps_overlay": False}, daemon=True),
            threading.Thread(target=self.scheduler.run_loop, kwargs={"stop_event": self._stop_event}, daemon=True),
            threading.Thread(target=self.control.run_loop, kwargs={"stop_event": self._stop_event}, daemon=True),
        ]

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
                "conf_min": 0.25,
                "target_cls": 0,
                "fov_deg": {"x": 54.0, "y": 41.0},
            },
            "control": {
                "tick_hz": 150,
                "alpha": 0.6,
                "beta": 0.2,
                "kp": 1.8,
                "kd": 0.2,
                "ki": 0.0,
                "deadband_steps": 1,
                "micro_move_T_ms": 40,
                "tau0_ms": 60,
            },
            "scheduler": {
                "tick_hz": 1500,
                "s_max_steps_s": 500,
                "a_max_steps_s2": 4000,
            },
            "stepper": {
                "steps_per_rev": 4096,
                "yaw_pins": [23, 24, 25, 5],
                "pitch_pins": [17, 18, 27, 22],
                "yaw_cw_positive": True,
                "pitch_cw_positive": False,
            },
            "gpio": {"mode": "BCM"},
        }

    def _load_config(self, path_str: str) -> Dict[str, Any]:
        cfg = self._default_config()
        path = Path(path_str)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    user_cfg = yaml.safe_load(f) or {}
                # Shallow-merge dicts
                for k, v in user_cfg.items():
                    if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
            except Exception:
                pass
        return cfg

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
    