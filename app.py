# app.py (cleaned / dev-safe version)
import os
import time
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

# External helpers (left as-is — uncomment SendMessage import when you want alerts)
from NearestHospital import get_nearest_hospitals
# from SendMessage import send_message    # <-- uncomment this and the call below to enable chat alerts
from currentLocation import get_current_location
from temporal_module import Detection, TemporalAccidentVerifier
from vanet_layer import (
    AccidentAlert,
    build_accident_alert,
    demo_http_broadcast,
    publish_mqtt_alert,
)

# =========================
# Runtime Configuration
# =========================
# Use environment variables to toggle external behavior.
# By default both are OFF so the pipeline runs locally without external failures.
ENABLE_ALERTS = os.getenv("ENABLE_ALERTS", "0") == "1"
ENABLE_VANET = os.getenv("ENABLE_VANET", "0") == "1"

# Load trained YOLO model (Edge AI Detection Module)
# NOTE: If best_model.pt is large or missing, model creation can be slow or raise.
# Keep this here for convenience, or move to lazy load by wrapping in a function.
try:
    model = YOLO("best_model.pt")
except Exception as e:
    print(f"⚠️ Warning: failed to load YOLO model 'best_model.pt': {e}")
    model = None

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
NO_ACCIDENT_THRESHOLD = 30  # Frames without accident before resetting detection


def save_accident_frame(accident_frame, output_dir="accident_frames") -> str:
    """Saves the accident frame and returns its file path."""
    os.makedirs(output_dir, exist_ok=True)
    frame_filename = os.path.join(output_dir, f"accident_{int(time.time())}.jpg")
    cv2.imwrite(frame_filename, accident_frame)
    print(f"✅ Accident frame saved: {frame_filename}")
    return frame_filename


def _yolo_to_temporal_detections(results) -> List[Detection]:
    """
    Convert YOLO results to a list of Temporal Detection objects.
    Keeps only class 0 (vehicle/person — dependent on your dataset) and above confidence threshold.
    """
    detections: List[Detection] = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0]) if box.cls is not None else 0
            conf = float(box.conf[0])
            if cls != 0 or conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf))
    return detections


def _handle_confirmed_accident(
    frame,
    temporal_confidence: float,
    output_dir: str = "accident_frames",
) -> AccidentAlert:
    """
    Persist frame, prepare alert object, and (optionally) invoke external alerting.
    This function is intentionally defensive: external calls are guarded by flags and try/except.
    """
    frame_filename = save_accident_frame(frame, output_dir=output_dir)

    # Safe location fetch (fallback coordinate used if retrieval fails)
    try:
        location = get_current_location() or "28.5439375,77.3304876"
    except Exception:
        location = "28.5439375,77.3304876"

    # Nearest hospitals (if module exists). Keep call — if it raises, catch and continue.
    try:
        hospitals = get_nearest_hospitals(location)
    except Exception as exc:
        print(f"⚠️ get_nearest_hospitals() failed: {exc}")
        hospitals = "unknown"

    # Human-readable message (for chat or logs)
    message = (
        f"Accident detected at location: {location} "
        f"(temporal_confidence={temporal_confidence:.2f}). "
        f"Message sent to {hospitals}"
    )

    # Build the VANET alert object (kept in all modes; printing is safe)
    try:
        alert = build_accident_alert(location, confidence=temporal_confidence)
    except Exception as exc:
        # If the alert builder fails, create a minimal fallback object (dict-like)
        print(f"⚠️ build_accident_alert() failed: {exc}")
        class _FallbackAlert:
            def __init__(self, loc, conf):
                self.loc = loc
                self.conf = conf
            def to_json(self):
                return {"type": "accident", "location": self.loc, "confidence": self.conf}
            def __repr__(self):
                return str(self.to_json())
        alert = _FallbackAlert(location, temporal_confidence)

    payload = alert.to_json() if hasattr(alert, "to_json") else alert
    print("📡 VANET/V2X alert:", payload)

    # =========================
    # Optional Alert Layer
    # =========================

    # Alerts (chat / notification)
    if ENABLE_ALERTS:
        try:
            # send_message(message, frame_filename)  # <-- Uncomment when SendMessage import is enabled
            print("🔔 ALERT: send_message would run here (enabled).")
        except Exception as exc:
            print(f"⚠️ Chat alert skipped: {exc}")
    else:
        # For dev clarity: print short log that alerts are disabled
        print("🔕 ALERTS disabled (ENABLE_ALERTS != 1)")

    # VANET publish (MQTT / HTTP demo)
    if ENABLE_VANET:
        try:
            publish_mqtt_alert(alert)
            demo_http_broadcast(alert)
            print("✅ VANET alerts published.")
        except Exception as exc:
            print(f"⚠️ VANET broadcast skipped: {exc}")
    else:
        print("🔕 VANET disabled (ENABLE_VANET != 1)")

    return alert


def detect_accident(video_path, output_dir="accident_frames"):
    """
    End-to-end pipeline:
    CCTV → YOLOv8 detection → Temporal verification → VANET alert + hospital notification (optional).

    Notes:
    - model must be available (model != None), otherwise function will early-exit.
    - This loop displays a live window; press 'q' to stop.
    """
    if model is None:
        raise RuntimeError("YOLO model not loaded. Put 'best_model.pt' in project root or set model variable.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    verifier = TemporalAccidentVerifier()
    no_accident_frames = 0
    accident_active = False

    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()

        # Run detection (Ultralytics returns a Results object that supports indexing)
        try:
            results = model(frame)
        except Exception as exc:
            print(f"⚠️ Model inference failed for frame: {exc}")
            continue

        temporal_detections = _yolo_to_temporal_detections(results)
        is_accident, temporal_conf = verifier.update(temporal_detections)

        accident_detected_this_frame = False
        for det in temporal_detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                frame,
                f"Acc ({det.confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            accident_detected_this_frame = True

        if is_accident and not accident_active:
            accident_active = True
            try:
                _handle_confirmed_accident(frame, temporal_confidence=temporal_conf, output_dir=output_dir)
                print(f"✅ Temporal accident trigger (confidence={temporal_conf:.2f})")
            except Exception as exc:
                print(f"⚠️ _handle_confirmed_accident() failed: {exc}")

        if accident_detected_this_frame:
            no_accident_frames = 0
        else:
            no_accident_frames += 1

        if no_accident_frames >= NO_ACCIDENT_THRESHOLD:
            accident_active = False
            verifier.reset()

        # Visual layout (left: original, right: processed)
        frame_resized = cv2.resize(frame, (640, 360))
        original_resized = cv2.resize(original_frame, (640, 360))

        cv2.putText(
            original_resized,
            "Original Video",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame_resized,
            "Processed + Temporal Verification",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if accident_active:
            cv2.putText(
                frame_resized,
                "ACCIDENT CONFIRMED",
                (50, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        combined_frame = np.hstack((original_resized, frame_resized))
        cv2.imshow(
            "SaferoadAI | Left: Original | Right: Det + Temporal Verification",
            combined_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sample_video = "sample_videos/video1.mp4"
    detect_accident(sample_video)