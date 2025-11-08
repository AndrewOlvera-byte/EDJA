import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2


# Configuration (use this script's directory only; no subfolders)
BASE_DIR = Path(__file__).resolve().parent
PT_PATH = BASE_DIR / "yolo11n.pt"
ONNX_PATH = BASE_DIR / "yolo11n.onnx"
# Migrate from old layout if present
OLD_MODELS_DIR = BASE_DIR / "models"
INPUT_SIZE = (640, 640)                        # YOLOv11n default inference size
WINDOW_NAME = "YOLOv11n ONNX (FPS)"
CONFIDENCE_THRESHOLD = 0.25                    # not used unless you add postprocessing
NMS_IOU_THRESHOLD = 0.50                       # IoU threshold for NMS when parsing raw outputs


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
    Ensure an ONNX model exists at ONNX_PATH. If missing, try to:
    1) Ensure a local .pt at PT_PATH (download via Ultralytics if online).
    2) Export ONNX from the local .pt and place it at ONNX_PATH.
    """
    # Migrate from old 'models' subdir if files exist there
    import shutil
    old_pt = OLD_MODELS_DIR / "yolo11n.pt"
    if old_pt.exists() and not PT_PATH.exists():
        shutil.copy2(old_pt, PT_PATH)
        print(f"Found legacy PT at {old_pt}, copied to {PT_PATH}")
    old_onnx = OLD_MODELS_DIR / "yolo11n.onnx"
    if old_onnx.exists() and not ONNX_PATH.exists():
        old_onnx.replace(ONNX_PATH)
        print(f"Found legacy ONNX at {old_onnx}, moved to {ONNX_PATH}")

    if ONNX_PATH.exists():
        return ONNX_PATH

    try:
        from ultralytics import YOLO
    except Exception:
        # If Ultralytics isn't present, we can't download or export.
        raise FileNotFoundError(
            f"Missing {ONNX_PATH}. Install Ultralytics to auto-download/export once online:\n"
            "  pip install ultralytics\n"
        )

    # 1) Ensure local PT exists; if not, trigger download to cache and copy into BASE_DIR
    if not PT_PATH.exists():
        print("Local PT not found. Attempting to download 'yolo11n.pt' via Ultralytics...")
        # Trigger download (to cache); this requires internet
        _ = YOLO("yolo11n.pt")
        # Try to resolve an absolute path for the weight from the model if available, else scan caches
        cached = None
        try:
            # Some Ultralytics versions expose model.ckpt_path / model.model if loaded from disk
            # Fallback to cache scan if not available
            cached = _find_downloaded_pt()
        except Exception:
            cached = _find_downloaded_pt()
        if not cached:
            raise FileNotFoundError(
                "Ultralytics did not populate a cached yolo11n.pt. Ensure internet connectivity and retry."
            )
        shutil.copy2(cached, PT_PATH)
        print(f"Cached PT found and copied to {PT_PATH}")

    # 2) Export ONNX from local PT
    print(f"Exporting ONNX from {PT_PATH} ...")
    model = YOLO(str(PT_PATH))
    prev_cwd = os.getcwd()
    try:
        # Ensure export drops output into BASE_DIR to avoid 'runs/' or CWD confusion
        os.chdir(str(BASE_DIR))
        exported_path = model.export(
            format="onnx",
            imgsz=INPUT_SIZE[0],
            opset=12,
            dynamic=False,
            simplify=True,
        )
    finally:
        os.chdir(prev_cwd)

    # Normalize return and ensure file is at ONNX_PATH
    exported_candidates = []
    if isinstance(exported_path, (list, tuple)):
        exported_candidates = [Path(p) for p in exported_path]
    elif exported_path:
        exported_candidates = [Path(exported_path)]
    # Also check the expected filename in BASE_DIR
    exported_candidates += [BASE_DIR / "yolo11n.onnx"]

    for exported in exported_candidates:
        if exported and exported.exists():
            if exported.resolve() != ONNX_PATH.resolve():
                exported.replace(ONNX_PATH)
            print(f"Exported ONNX available at {ONNX_PATH}")
            return ONNX_PATH

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


def non_max_suppression(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    A simple NMS implementation. Returns indices of boxes to keep.
    boxes_xyxy: (N, 4) in xyxy format.
    scores: (N,)
    """
    if boxes_xyxy.size == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]

    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep_indices = []
    while order.size > 0:
        i = order[0]
        keep_indices.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-7)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep_indices, dtype=np.int32)


def parse_onnx_detections(outputs: list, conf_threshold: float) -> np.ndarray:
    """
    Parse ONNX outputs into detections in the ONNX input coordinate space (640x640).
    Returns detections as (M, 6): [x1, y1, x2, y2, score, class_id].

    Handles two common cases:
    - Fused NMS export: output already Nx6 (or 1xNx6).
    - Raw predictions: (1, N, 4+num_classes) or (1, 4+num_classes, N).
    """
    if not outputs:
        return np.zeros((0, 6), dtype=np.float32)

    out0 = outputs[0]
    arr = np.array(out0)
    squeezed = np.squeeze(arr)

    # Case A: Fused NMS → detections already in [x1,y1,x2,y2,score,class]
    if squeezed.ndim == 2 and squeezed.shape[-1] in (6, 7):
        # Some exports may include batch index making it 7 columns; drop if present
        if squeezed.shape[-1] == 7:
            # Heuristic: if first column appears to be all zeros (batch idx), drop it
            if np.all((squeezed[:, 0] == 0) | (squeezed[:, 0] == 1)):
                squeezed = squeezed[:, 1:]
        dets = squeezed.astype(np.float32)
        if dets.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        # Filter by confidence
        conf_mask = dets[:, 4] >= conf_threshold
        return dets[conf_mask]

    # Case B: Raw predictions → (1, N, 4+classes) or (1, 4+classes, N)
    preds = arr
    if preds.ndim == 3 and preds.shape[0] == 1:
        # Ensure shape is (1, N, C)
        if preds.shape[1] in (84, 85) and preds.shape[2] > preds.shape[1]:
            preds = np.transpose(preds, (0, 2, 1))
        preds = preds[0]  # (N, C)
    elif preds.ndim == 2:
        # Already (N, C)
        preds = preds
    else:
        return np.zeros((0, 6), dtype=np.float32)

    if preds.shape[1] <= 4:
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xywh = preds[:, 0:4]
    class_scores = preds[:, 4:]
    if class_scores.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    conf_mask = scores >= conf_threshold
    if not np.any(conf_mask):
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xywh = boxes_xywh[conf_mask]
    scores = scores[conf_mask]
    class_ids = class_ids[conf_mask]

    # Convert xywh -> xyxy
    x, y, w, h = boxes_xywh.T
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Apply NMS
    keep = non_max_suppression(boxes_xyxy, scores, NMS_IOU_THRESHOLD)
    if keep.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    dets = np.concatenate([boxes_xyxy[keep], scores[keep, None], class_ids[keep, None].astype(np.float32)], axis=1)
    return dets.astype(np.float32)


def draw_detections(frame_bgr: np.ndarray, dets_xyxy: np.ndarray, scale_x: float, scale_y: float) -> None:
    """
    Draw rectangles and labels onto frame_bgr using detections in model (input) space.
    dets_xyxy: (M, 6) with [x1,y1,x2,y2,score,class_id]
    """
    if dets_xyxy.size == 0:
        return
    h, w = frame_bgr.shape[:2]
    for det in dets_xyxy:
        x1, y1, x2, y2, score, cls_id = det.tolist()
        # Scale from model input space (640x640) to original frame size
        rx1 = int(max(0, min(w - 1, x1 * scale_x)))
        ry1 = int(max(0, min(h - 1, y1 * scale_y)))
        rx2 = int(max(0, min(w - 1, x2 * scale_x)))
        ry2 = int(max(0, min(h - 1, y2 * scale_y)))

        color = (0, 255, 255) if int(cls_id) % 2 == 0 else (255, 0, 0)
        cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), color, 2)
        label = f"{int(cls_id)}:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (rx1, max(0, ry1 - th - 6)), (rx1 + tw + 4, ry1), color, -1)
        cv2.putText(frame_bgr, label, (rx1 + 2, ry1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


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

            # Run inference and parse detections
            outputs = session.run(None, {input_name: input_tensor})
            dets = parse_onnx_detections(outputs, conf_threshold=CONFIDENCE_THRESHOLD)

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

            # Draw detections (scale from model input size to original frame size)
            h, w = frame_bgr.shape[:2]
            scale_x = w / float(INPUT_SIZE[0])
            scale_y = h / float(INPUT_SIZE[1])
            draw_detections(frame_bgr, dets, scale_x, scale_y)

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

