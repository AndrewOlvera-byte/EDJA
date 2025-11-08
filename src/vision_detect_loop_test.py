import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2


# Configuration
MODEL_DIR = Path("src/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ONNX_PATH = MODEL_DIR / "yolo11n.onnx"   # expected local ONNX path
MODEL_PT_PATH = MODEL_DIR / "yolo11n.pt"       # local .pt for export / caching
INPUT_SIZE = (640, 640)                        # YOLOv11n default inference size
WINDOW_NAME = "YOLOv11n ONNX (FPS)"
CONFIDENCE_THRESHOLD = 0.25                    # not used unless you add postprocessing


def _find_downloaded_pt() -> Path | None:
    """
    Search common Ultralytics/Torch cache locations for a downloaded yolo11n .pt.
    Returns the first match if found.
    """
    home = Path.home()
    candidates = []
    candidates += list((home / ".cache" / "ultralytics").rglob("yolo11n*.pt"))
    candidates += list((home / ".cache" / "torch" / "hub").rglob("yolo11n*.pt"))
    return candidates[0] if candidates else None


def ensure_onnx_model_exists() -> Path:
    """
    Ensure an ONNX model exists at MODEL_ONNX_PATH. If missing, try to:
    1) Ensure a local .pt at MODEL_PT_PATH (download via Ultralytics if online).
    2) Export ONNX from the local .pt and place it at MODEL_ONNX_PATH.
    """
    if MODEL_ONNX_PATH.exists():
        return MODEL_ONNX_PATH

    try:
        from ultralytics import YOLO
    except Exception:
        # If Ultralytics isn't present, we can't download or export.
        raise FileNotFoundError(
            f"Missing {MODEL_ONNX_PATH}. Install Ultralytics to auto-download/export once online:\n"
            "  pip install ultralytics\n"
        )

    # 1) Ensure local PT exists; if not, trigger download to cache and copy into repo
    if not MODEL_PT_PATH.exists():
        print("Local PT not found. Attempting to download 'yolo11n.pt' via Ultralytics...")
        # Trigger download (to cache); this requires internet
        _ = YOLO("yolo11n.pt")
        cached = _find_downloaded_pt()
        if not cached:
            raise FileNotFoundError(
                "Ultralytics did not populate a cached yolo11n.pt. Ensure internet connectivity and retry."
            )
        import shutil
        shutil.copy2(cached, MODEL_PT_PATH)
        print(f"Cached PT found and copied to {MODEL_PT_PATH}")

    # 2) Export ONNX from local PT
    print(f"Exporting ONNX from {MODEL_PT_PATH} ...")
    model = YOLO(str(MODEL_PT_PATH))
    model.export(format="onnx", imgsz=INPUT_SIZE[0], opset=12, dynamic=False, simplify=True, nms=True)
    # Ultralytics typically writes 'yolo11n.onnx' in CWD
    exported = Path("yolo11n.onnx")
    if exported.exists():
        MODEL_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
        exported.replace(MODEL_ONNX_PATH)
        print(f"Exported ONNX moved to {MODEL_ONNX_PATH}")
        return MODEL_ONNX_PATH

    raise FileNotFoundError("Export finished but yolo11n.onnx not found. Check Ultralytics export logs.")


def preprocess_for_onnx(image_rgb: np.ndarray, size=(640, 640)) -> np.ndarray:
    """
    Minimal preprocessing for YOLOv11n ONNX:
    - Resize to (size,size) (no letterbox for simplicity)
    - Normalize to [0, 1]
    - Convert HWC RGB -> NCHW
    """
    resized = cv2.resize(image_rgb, size, interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
    tensor = np.expand_dims(tensor, axis=0).copy()  # add batch dim, ensure contiguous
    return tensor


def main():
    # 1) Ensure model is present (export if possible)
    onnx_path = ensure_onnx_model_exists()

    # 2) Init ONNX Runtime
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # 3) Init Pi Camera
    picam2 = Picamera2()
    # Preview config provides fast frames; format RGB888 for convenient NumPy array
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    # 4) UI
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    # 5) Main loop
    fps_smooth = None
    try:
        while True:
            loop_start = time.perf_counter()

            # Capture frame as RGB (H, W, 3)
            frame_rgb = picam2.capture_array()

            # Prepare model input
            input_tensor = preprocess_for_onnx(frame_rgb, size=INPUT_SIZE)

            # Run inference (we do not postprocess here; goal is to stress the model and show FPS)
            _ = session.run(None, {input_name: input_tensor})

            # Compute FPS
            dt = time.perf_counter() - loop_start
            fps_instant = 1.0 / dt if dt > 0 else 0.0
            if fps_smooth is None:
                fps_smooth = fps_instant
            else:
                # Exponential moving average for stability
                fps_smooth = fps_smooth * 0.9 + fps_instant * 0.1

            # Convert to BGR for display and overlay FPS
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

            cv2.imshow(WINDOW_NAME, frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        # Provide a concise hint on setup
        print(str(e))
        print(
            "Hints:\n"
            f"- Place ONNX at {MODEL_ONNX_PATH}\n"
            f"- Or place PT at {MODEL_PT_PATH} and ensure 'pip install ultralytics' is done\n"
            "- Dependencies: sudo apt install -y python3-picamera2 python3-opencv; pip install onnxruntime ultralytics\n"
        )

