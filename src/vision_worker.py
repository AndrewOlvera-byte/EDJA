from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    from picamera2 import Picamera2  # type: ignore
    _PICAM_AVAILABLE = True
except Exception:
    _PICAM_AVAILABLE = False

from .shared import Detection, LatestDetectionMailbox
try:
    # When running as package (python -m src.App)
    from .vision_detect_loop_test import preprocess_for_onnx, parse_onnx_detections  # type: ignore
except Exception:
    # When running as script with src on sys.path
    from vision_detect_loop_test import preprocess_for_onnx, parse_onnx_detections  # type: ignore


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

    def __init__(self, cfg: VisionConfig, mailbox: LatestDetectionMailbox) -> None:
        self.cfg = cfg
        self.mailbox = mailbox
        self._session = self._create_session()
        self._input_name = self._session.get_inputs()[0].name
        self._camera = self._open_camera()

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
            H, W = frame_rgb.shape[:2]
            return frame_rgb, W, H
        else:
            ok, frame_bgr = self._camera.read()
            if not ok or frame_bgr is None:
                raise RuntimeError("Failed to capture frame from camera")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            H, W = frame_rgb.shape[:2]
            return frame_rgb, W, H

    def _pick_target(self, dets_xyxy: np.ndarray, scale_x: float, scale_y: float) -> Optional[Tuple[float, float, float, int]]:
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
        # Scale to frame pixels
        cx_px = cx_model * scale_x
        cy_px = cy_model * scale_y
        return float(cx_px), float(cy_px), float(conf), int(cls)

    def run_loop(self, stop_event: Optional[Any] = None, fps_overlay: bool = False) -> None:
        """
        Main loop. If stop_event is provided, stops when set().
        """
        input_w, input_h = self.cfg.input_size
        fps_smooth: Optional[float] = None
        window_name = "Vision (FPS)"
        if fps_overlay and not _PICAM_AVAILABLE:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 720)

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                t_cap = time.perf_counter()
                frame_rgb, W, H = self._capture_rgb()

                tensor = preprocess_for_onnx(frame_rgb, size=(input_w, input_h))
                outputs = self._session.run(None, {self._input_name: tensor})
                t_inf = time.perf_counter()

                # Parse and find target
                dets = parse_onnx_detections(outputs, conf_threshold=self.cfg.conf_min)

                scale_x = W / float(input_w)
                scale_y = H / float(input_h)
                picked = self._pick_target(dets, scale_x, scale_y)

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

                # Optional FPS overlay (OpenCV only)
                if fps_overlay and not _PICAM_AVAILABLE:
                    dt = max(1e-6, time.perf_counter() - t_cap)
                    fps_instant = 1.0 / dt
                    if fps_smooth is None:
                        fps_smooth = fps_instant
                    else:
                        fps_smooth = fps_smooth * 0.9 + fps_instant * 0.1
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
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
                    cv2.imshow(window_name, frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if fps_overlay and not _PICAM_AVAILABLE:
                cv2.destroyAllWindows()
            # Stop camera if picamera2
            if _PICAM_AVAILABLE and hasattr(self._camera, "stop"):
                try:
                    self._camera.stop()
                except Exception:
                    pass


