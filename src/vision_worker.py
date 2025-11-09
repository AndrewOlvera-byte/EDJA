from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    from picamera2 import Picamera2
    _PICAM_AVAILABLE = True
except Exception:
    _PICAM_AVAILABLE = False

from shared import Detection, LatestDetectionMailbox
try:
    # When running as package (python -m src.App)
    from .vision_detect_loop_test import preprocess_for_onnx, parse_onnx_detections, draw_detections
except Exception:
    # When running as script with src on sys.path
    from vision_detect_loop_test import preprocess_for_onnx, parse_onnx_detections, draw_detections


@dataclass
class VisionConfig:
    onnx_path: str
    providers: Tuple[str, ...]
    input_size: Tuple[int, int]
    conf_min: float
    target_cls: int


class VisionWorker:
    """
    Capture frames, run ONNX inference, pick target, write latest Detection.
    """

    def __init__(self, cfg: VisionConfig, mailbox: LatestDetectionMailbox, logger: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.mailbox = mailbox
        self.logger = logger
        self._session = self._create_session()
        self._input_name = self._session.get_inputs()[0].name
        self._camera = self._open_camera()
        # Setup summary print
        cam_kind = "Picamera2" if _PICAM_AVAILABLE else "OpenCV-USB"
        print(f"[Vision] setup: onnx='{self.cfg.onnx_path}', providers={self.cfg.providers}, input_size={self.cfg.input_size}, camera={cam_kind}")
        if self.logger is not None:
            try:
                self.logger.log("Vision", "setup", f"onnx={self.cfg.onnx_path}, providers={self.cfg.providers}, input={self.cfg.input_size}, camera={cam_kind}")
            except Exception:
                pass

    def _create_session(self) -> ort.InferenceSession:
        return ort.InferenceSession(
            self.cfg.onnx_path,
            providers=list(self.cfg.providers),
        )

    def _open_camera(self) -> Any:
        if _PICAM_AVAILABLE:
            cam = Picamera2()
            cam.preview_configuration.main.size = (640, 480)
            cam.preview_configuration.main.format = "RGB888"
            cam.configure("preview")
            cam.start()
            return cam
        # Fallback to OpenCV USB camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def _capture_rgb(self) -> Tuple[np.ndarray, int, int]:
        if _PICAM_AVAILABLE and hasattr(self._camera, "capture_array"):
            frame_rgb = self._camera.capture_array()  # RGB
        else:
            ok, frame_bgr = self._camera.read()
            if not ok or frame_bgr is None:
                raise RuntimeError("Failed to capture frame from camera")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = frame_rgb.shape[:2]
        return frame_rgb, W, H

    def _pick_target(self, dets_xyxy: np.ndarray, ratio: float, pad_x: float, pad_y: float, W: int, H: int) -> Optional[Tuple[float, float, float, int]]:
        """
        dets_xyxy: (M, 6) [x1,y1,x2,y2,score,cls] in model input space (e.g., 640x640).
        Returns (cx_px, cy_px, conf, cls) in original frame pixel coordinates.
        """
        if dets_xyxy.size == 0:
            return None
        # Filter by class
        classes = dets_xyxy[:, 5].astype(np.int32)
        mask = classes == int(self.cfg.target_cls)
        if not np.any(mask):
            return None
        cand = dets_xyxy[mask]
        # Argmax by confidence
        idx = int(np.argmax(cand[:, 4]))
        x1, y1, x2, y2, conf, cls = cand[idx].tolist()
        cx_model = (x1 + x2) * 0.5
        cy_model = (y1 + y2) * 0.5
        # Map from model input (letterboxed) to original frame pixels
        cx_px = (cx_model - float(pad_x)) / max(1e-6, float(ratio))
        cy_px = (cy_model - float(pad_y)) / max(1e-6, float(ratio))
        cx_px = float(max(0.0, min(float(W - 1), cx_px)))
        cy_px = float(max(0.0, min(float(H - 1), cy_px)))
        return float(cx_px), float(cy_px), float(conf), int(cls)

    def run_loop(self, stop_event: Optional[Any] = None, fps_overlay: bool = False) -> None:
        """
        Main loop. If stop_event is provided, stops when set().
        """
        input_w, input_h = self.cfg.input_size
        fps_smooth: Optional[float] = None
        window_name = "Vision (FPS)"
        if fps_overlay:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 720)
            except Exception:
                # If window creation fails (e.g., no display), continue headless
                fps_overlay = False
        last_print_t = time.perf_counter()

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                t_cap = time.perf_counter()
                frame_rgb, W, H = self._capture_rgb()

                tensor, ratio, pad_x, pad_y = preprocess_for_onnx(frame_rgb, size=(input_w, input_h))
                outputs = self._session.run(None, {self._input_name: tensor})
                t_inf = time.perf_counter()

                # Parse and find target
                dets = parse_onnx_detections(outputs, conf_threshold=self.cfg.conf_min)

                picked = self._pick_target(dets, ratio, pad_x, pad_y, W, H)

                if picked is not None:
                    cx_px, cy_px, conf, cls = picked
                    det = Detection(
                        t_cap=t_cap,
                        t_inf=t_inf,
                        W=W,
                        H=H,
                        cx=cx_px,
                        cy=cy_px,
                        conf=conf,
                        cls=cls,
                    )
                    self.mailbox.write(det)

                # Periodic print/log
                now = time.perf_counter()
                if (now - last_print_t) >= 1.0:
                    if picked is not None:
                        print(f"[Vision] frame {W}x{H}, dets={int(dets.shape[0])}, picked: cls={cls} conf={conf:.2f} at ({int(cx_px)},{int(cy_px)})")
                        if self.logger is not None:
                            try:
                                self.logger.log("Vision", "detection", f"W={W} H={H} N={int(dets.shape[0])} cls={cls} conf={conf:.3f} cx={cx_px:.1f} cy={cy_px:.1f}")
                            except Exception:
                                pass
                    else:
                        print(f"[Vision] frame {W}x{H}, dets=0, no target")
                        if self.logger is not None:
                            try:
                                self.logger.log("Vision", "no_target", f"W={W} H={H} N=0")
                            except Exception:
                                pass
                    last_print_t = now

                # Optional FPS/detections overlay (supports both Picamera2 and OpenCV)
                if fps_overlay:
                    dt = max(1e-6, time.perf_counter() - t_cap)
                    fps_instant = 1.0 / dt
                    if fps_smooth is None:
                        fps_smooth = fps_instant
                    else:
                        fps_smooth = fps_smooth * 0.9 + fps_instant * 0.1
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    # Draw only the tracked target (person) using letterbox mapping
                    if dets is not None and dets.size > 0 and picked is not None:
                        try:
                            # Filter to target class and select same max-conf target
                            classes = dets[:, 5].astype(np.int32)
                            dets_person = dets[classes == int(self.cfg.target_cls)]
                            if dets_person.size > 0:
                                idx = int(np.argmax(dets_person[:, 4]))
                                target_det = dets_person[idx][None, :]
                                draw_detections(frame_bgr, target_det, ratio, pad_x, pad_y)
                        except Exception:
                            pass
                    cv2.putText(
                        frame_bgr,
                        f"FPS: {fps_smooth:.2f}",
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    # Rotate only for display after overlays so boxes rotate with the image
                    display_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
                    cv2.imshow(window_name, display_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if fps_overlay:
                cv2.destroyAllWindows()
            # Stop camera if picamera2
            if _PICAM_AVAILABLE and hasattr(self._camera, "stop"):
                try:
                    self._camera.stop()
                except Exception:
                    pass


