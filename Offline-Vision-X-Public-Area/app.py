"""Offline Vision-X local prototype. Run with: python app.py"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
from assistant import (
    AssistantConfigurationError,
    AssistantRequestError,
    GemmaSceneAssistant,
    scene_summary,
)
sys.path.insert(0, str(ROOT / "python"))
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def config(name: str) -> dict:
    candidates = (ROOT / "config" / name, ROOT / name)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Configuration file not found: {name}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load configuration {path}: {exc}") from exc


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / max(1, aw*ah + bw*bh - inter)


class Tracker:
    """Greedy IoU tracker; IDs are session-local and may switch on crossings."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = int(config("tracker.yaml").get("max_lost_frames", 12))

    name = "Greedy IoU"

    def update(self, detections, frame=None, timestamp_s=None, latency_ms=0.0):
        ids = list(self.tracks)
        pairs = sorted(((iou(self.tracks[k]["bbox"], d["bbox"]), k, j)
                        for k in ids for j, d in enumerate(detections)), reverse=True)
        used_t, used_d = set(), set()
        for score, k, j in pairs:
            if score < .15 or k in used_t or j in used_d: continue
            t, d = self.tracks[k], detections[j]
            old = t["bbox"]
            ox, oy = old[0]+old[2]/2, old[1]+old[3]/2
            nx, ny = d["bbox"][0]+d["bbox"][2]/2, d["bbox"][1]+d["bbox"][3]/2
            t.update(d); t["velocity_px_s"] = [round((nx-ox)*self.fps, 1), round((ny-oy)*self.fps, 1)]
            t["missed"] = 0; t["age"] += 1; used_t.add(k); used_d.add(j)
        for k in ids:
            if k not in used_t: self.tracks[k]["missed"] += 1
        for j, d in enumerate(detections):
            if j not in used_d:
                k = self.next_id; self.next_id += 1
                self.tracks[k] = {**d, "id": k, "age": 1, "missed": 0, "velocity_px_s": [0,0], "born": time.time()}
        self.tracks = {k:v for k,v in self.tracks.items() if v["missed"] <= self.max_lost}
        return [dict(v, state="LOST" if v["missed"] else ("NEW" if v["age"] == 1 else "TRACKING")) for v in self.tracks.values()]


class PublicAreaDemoCapture:
    """Generated frames for an explicitly labeled software-only dashboard demo."""
    def __init__(self, width: int, height: int, fps: int):
        self.width, self.height = width, height
        self.interval = 1 / max(1, fps)
        self.started = time.monotonic()
        self.released = False

    def isOpened(self):
        return not self.released

    def set(self, *_args):
        return True

    def read(self):
        if self.released or cv2 is None:
            return False, None
        time.sleep(self.interval)
        import numpy as np
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        elapsed = time.monotonic() - self.started
        x = 80 + min(320, int(max(0, elapsed - 1) * 160))
        center_x = x + 30
        color = (185, 205, 225)
        cv2.circle(frame, (center_x, 155), 14, color, -1)
        cv2.rectangle(frame, (center_x - 13, 170), (center_x + 13, 220), color, -1)
        cv2.line(frame, (center_x - 8, 218), (center_x - 18, 250), color, 5)
        cv2.line(frame, (center_x + 8, 218), (center_x + 18, 250), color, 5)
        cv2.putText(frame, "SIMULATED PUBLIC-AREA DEMO", (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 210, 80), 2)
        return True, frame

    def release(self):
        self.released = True


class VisionApp:
    def __init__(self, device=0, mode="public_area", demo=False):
        self.device, self.mode = device, mode
        self.demo_mode = bool(demo)
        if self.demo_mode and self.mode != "public_area":
            raise ValueError("The simulated demo is only available in public_area mode.")
        self.cap = None
        self.lock = threading.RLock()
        self.frame = None
        self.tracks = []
        self.detection_count = 0
        self.events = deque(maxlen=200)
        self.gemma_assistant = GemmaSceneAssistant.from_environment()
        self.industrial = None
        try:
            from monitor import IndustrialConfig, IndustrialMonitor
            self.industrial = IndustrialMonitor(IndustrialConfig.load(ROOT / "industrial.yaml"))
        except (ImportError, OSError, ValueError, TypeError) as exc:
            self.events.append({"time": now(), "type": "industrial_config_error", "message": str(exc)})
        self.public_area = None
        self.public_area_error = ""
        try:
            from monitor import PublicAreaConfig, PublicAreaMonitor
            self.public_area_config = PublicAreaConfig.load(ROOT / "public_area.yaml")
            self.public_area = PublicAreaMonitor(self.public_area_config)
            if self.demo_mode:
                self.public_area_config = PublicAreaConfig.from_mapping({
                    "zones": [
                        {"name": "demo_entrance", "polygon": [[0, 0], [.35, 0], [.35, .65], [0, .65]]},
                        {"name": "demo_restricted", "sensitive": True,
                         "polygon": [[.62, 0], [1, 0], [1, .65], [.62, .65]]},
                    ],
                    "loiter_seconds": 5,
                    "transition_patterns": [{
                        "name": "demo_entrance_to_restricted",
                        "zones": ["demo_entrance", "demo_restricted"],
                    }],
                    "transition_window_seconds": 30,
                })
                self.public_area = PublicAreaMonitor(self.public_area_config)
        except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
            self.public_area_config = None
            self.public_area_error = str(exc)
        self.fps = 0.0; self.latency_ms = 0.0; self.processing_ms = 0.0
        self.camera_error = "OpenCV is not installed" if cv2 is None else "Camera not started"
        self.running = False
        self.tracker = Tracker(); self.tracker.fps = 15
        self.hog = None
        self.yolo_detector = None
        self.yolox_detector = None
        self.custom_detector = None
        self.detector_error = ""
        self.tracker_error = ""
        self.pipeline_error = ""
        self.lead_projector = None
        from tracking import BoxMOTBotSortTracker, LatencyKalmanProjector
        self.lead_projector = LatencyKalmanProjector()
        tracker_cfg = config("tracker.yaml")
        tracker_backend = tracker_cfg.get("backend", "greedy_iou")
        if tracker_backend not in ("greedy_iou", "boxmot_botsort"):
            raise ValueError(f"Unsupported tracker backend: {tracker_backend}")
        if tracker_backend == "boxmot_botsort":
            detector_cfg = config("detector.yaml")
            class_names = []
            if detector_cfg.get("backend") == "ultralytics_yolo":
                try:
                    from detection import UltralyticsYoloDetector
                    detector = UltralyticsYoloDetector(
                        str(ROOT / detector_cfg.get("model", "models/yolo.pt")),
                        float(detector_cfg.get("confidence_threshold", .25)),
                        int(detector_cfg.get("image_size", 640)),
                        str(detector_cfg.get("device", "cpu")),
                        int(detector_cfg.get("max_detections", 300)),
                    )
                    class_names = [detector.class_names[key] for key in sorted(detector.class_names)]
                    self.yolo_detector = detector
                except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
                    self.detector_error = str(exc)
            elif detector_cfg.get("backend") == "opencv_zoo_yolox":
                try:
                    from detection import OpenCVZooYoloXDetector
                    detector = OpenCVZooYoloXDetector(
                        str(ROOT / detector_cfg.get(
                            "model", "models/object_detection_yolox_2022nov.onnx"
                        )),
                        float(detector_cfg.get("confidence_threshold", .35)),
                        float(detector_cfg.get("nms_iou_threshold", .5)),
                    )
                    self.yolox_detector = detector
                    class_names = [detector.class_names[key] for key in sorted(detector.class_names)]
                except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
                    self.detector_error = str(exc)
            camera_fps = int(config("camera.yaml").get("fps", 30))
            try:
                self.tracker = BoxMOTBotSortTracker(class_names, max(1, camera_fps))
            except (ImportError, RuntimeError, ValueError, TypeError) as exc:
                self.tracker = None
                self.tracker_error = str(exc)
        det_cfg = config("detector.yaml")
        detector_backend = det_cfg.get("backend", "opencv_hog")
        if detector_backend not in (
            "opencv_hog", "opencv_zoo_yolox", "visionx_custom_grid", "ultralytics_yolo",
        ):
            raise ValueError(f"Unsupported detector backend: {detector_backend}")
        if cv2 is not None and detector_backend == "opencv_hog":
            try:
                self.hog = cv2.HOGDescriptor(); self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except AttributeError as exc:
                self.detector_error = f"OpenCV build lacks the HOG people baseline: {exc}"
        if detector_backend == "opencv_zoo_yolox":
            if self.yolox_detector is None:
                try:
                    from detection import OpenCVZooYoloXDetector
                    self.yolox_detector = OpenCVZooYoloXDetector(
                        str(ROOT / det_cfg.get(
                            "model", "models/object_detection_yolox_2022nov.onnx"
                        )),
                        float(det_cfg.get("confidence_threshold", .35)),
                        float(det_cfg.get("nms_iou_threshold", .5)),
                    )
                except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
                    self.detector_error = str(exc)
            self.hog = None
        elif detector_backend == "ultralytics_yolo":
            if self.yolo_detector is None:
                try:
                    from detection import UltralyticsYoloDetector
                    self.yolo_detector = UltralyticsYoloDetector(
                        str(ROOT / det_cfg.get("model", "models/yolo.pt")),
                        float(det_cfg.get("confidence_threshold", .25)),
                        int(det_cfg.get("image_size", 640)),
                        str(det_cfg.get("device", "cpu")),
                        int(det_cfg.get("max_detections", 300)),
                    )
                except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
                    self.detector_error = str(exc)
            self.hog = None
        elif detector_backend == "visionx_custom_grid":
            try:
                from detection import CustomGridDetector
                classes_path = ROOT / det_cfg.get("classes", "python/datasets/classes.txt")
                classes = [x.strip() for x in classes_path.read_text(encoding="utf-8").splitlines() if x.strip()]
                self.custom_detector = CustomGridDetector(str(ROOT / det_cfg.get("checkpoint", "models/detector.pt")), classes, float(det_cfg.get("confidence_threshold", .35)), float(det_cfg.get("nms_iou_threshold", .45)), str(det_cfg.get("device", "cpu")))
                self.hog = None
            except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
                self.detector_error = str(exc)
                self.hog = None
        if self.yolo_detector is not None or self.yolox_detector is not None:
            self.hog = None
        if self.yolo_detector is not None:
            detector_labels = set(self.yolo_detector.class_names.values())
        elif self.custom_detector is not None:
            detector_labels = set(self.custom_detector.classes)
        else:
            detector_labels = set()
        configured_gestures = set(self.public_area_config.gesture_classes) if self.public_area_config else set()
        self.supported_gesture_classes = {
            label for label in detector_labels
            if str(label).casefold() in configured_gestures
        }
        self.homography = None
        self._load_homography()

    def _load_homography(self):
        if cv2 is None:
            return
        c = config("localization.yaml")
        pts = c.get("image_points", [])
        world = c.get("world_points_m", [])
        if len(pts) == len(world) and len(pts) >= 4:
            import numpy as np
            self.homography = cv2.getPerspectiveTransform(np.array(pts[:4], dtype="float32"), np.array(world[:4], dtype="float32"))

    def start(self):
        self.running = True
        if cv2 is None: return
        cv2.setNumThreads(1)
        cam = config("camera.yaml")
        if self.demo_mode:
            self.cap = PublicAreaDemoCapture(
                int(cam.get("width", 640)), int(cam.get("height", 480)),
                int(cam.get("fps", 15)),
            )
        else:
            self.cap = cv2.VideoCapture(self.device)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam.get("width", 640)))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("height", 480)))
        if not self.cap.isOpened():
            self.camera_error = f"Could not open camera device {self.device}"; self.cap.release(); self.cap = None; return
        self.camera_error = ""
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        prev = time.perf_counter()
        while self.running and self.cap:
            tick = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                self.camera_error = "Camera read failed"; time.sleep(.1); continue
            capture_time = time.perf_counter()
            detections = []
            try:
                if self.demo_mode:
                    elapsed = time.monotonic() - self.cap.started
                    x = 80 + min(320, int(max(0, elapsed - 1) * 160))
                    detections = [{
                        "class_id": 0, "class_name": "person", "confidence": 0.91,
                        "bbox": [x, 140, 60, 120],
                    }]
                elif self.yolo_detector is not None or self.yolox_detector is not None:
                    detector = self.yolo_detector or self.yolox_detector
                    detected = detector.detect(frame, time.time())
                    detections = [dict(
                        class_id=d.class_id, class_name=d.class_name,
                        confidence=round(float(d.confidence), 3),
                        bbox=[float(v) for v in d.bounding_box],
                    ) for d in detected]
                elif self.custom_detector is not None:
                    detections = [dict(class_id=d.class_id, class_name=d.class_name, confidence=round(float(d.confidence),3), bbox=[int(v) for v in d.bounding_box]) for d in self.custom_detector.detect(frame, time.time())]
                elif self.mode in ("human", "public_area") and self.hog is not None:
                    rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
                    detections = [dict(class_id=0, class_name="person", confidence=round(float(weights[i]), 3), bbox=list(map(int, r))) for i,r in enumerate(rects)]
                if self.tracker is None:
                    tracks = []
                else:
                    self.tracker.fps = 1 / max(.001, time.perf_counter()-prev)
                    tracks = self.tracker.update(
                        detections,
                        frame=frame,
                        timestamp_s=capture_time,
                    )
                prev = time.perf_counter()
                self.processing_ms = (time.perf_counter() - capture_time) * 1000
                self.latency_ms = self.processing_ms
                for track in tracks:
                    projection = self.lead_projector.project(
                        str(track["id"]), track["bbox"], capture_time, self.latency_ms,
                    )
                    track["lead_bbox"] = projection["bbox"]
                    track["lead_center_px"] = projection["center"]
                    track["velocity_px_s"] = projection["velocity_px_s"]
                    track["projection_horizon_ms"] = projection["horizon_ms"]
                self.pipeline_error = ""
            except Exception as exc:
                self.pipeline_error = f"{type(exc).__name__}: {exc}"
                if not self.events or self.events[-1].get("message") != self.pipeline_error:
                    self.events.append({"time": now(), "type": "perception_error", "message": self.pipeline_error})
                tracks = []
                detections = []
            if self.mode == "industrial" and self.industrial is not None:
                raw = []
                for track in tracks:
                    center_x = track["bbox"][0] + track["bbox"][2] / 2
                    foot_y = track["bbox"][1] + track["bbox"][3]
                    x_m, y_m = self.localize(center_x, foot_y)
                    raw.append({
                        "id": track["id"], "class_name": track["class_name"],
                        "x_m": x_m, "y_m": y_m, "state": track["state"],
                    })
                self.industrial.update(raw, timestamp=time.time())
                self.events = deque(self.industrial.events(), maxlen=200)
            elif self.mode == "public_area" and self.public_area is not None:
                self.public_area.update(
                    tracks, frame.shape, timestamp=time.time(),
                    supported_gesture_classes=self.supported_gesture_classes,
                )
                self.events = deque(self.public_area.events(), maxlen=200)
                if cv2 is not None:
                    import numpy as np
                    height, width = frame.shape[:2]
                    for zone in self.public_area_config.zones:
                        points = np.array([
                            [round(x * width), round(y * height)] for x, y in zone.polygon
                        ], dtype=np.int32)
                        color = (50, 170, 255) if zone.name in self.public_area_config.sensitive_zones else (180, 180, 70)
                        cv2.polylines(frame, [points], True, color, 2)
                        x, y = points[0]
                        cv2.putText(frame, zone.name, (int(x), max(16, int(y) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            for t in tracks:
                x,y,w,h=t["bbox"]
                x, y, w, h = map(int, (x, y, w, h))
                cv2.rectangle(frame,(x,y),(x+w,y+h),(70,220,150),2)
                lead = t.get("lead_bbox")
                if lead:
                    lx, ly, lw, lh = map(int, lead)
                    cv2.rectangle(frame,(lx,ly),(lx+lw,ly+lh),(255,190,70),1)
                label = f"{t['class_name']}-{int(t['id']):03d} {t['confidence']:.2f}"
                cv2.putText(frame,label,(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(70,220,150),2)
            self.fps=.9*self.fps+.1/(time.perf_counter()-tick) if self.fps else 1/(time.perf_counter()-tick)
            with self.lock:
                self.frame=frame; self.tracks=tracks; self.detection_count=len(detections); self.camera_error=""

    def track_payload(self):
        with self.lock:
            result=[]
            for t in self.tracks:
                x,y,w,h=t["bbox"]; px=x+w/2; py=y+h
                coord=self.localize(px,py)
                lead_bbox=t.get("lead_bbox",t["bbox"])
                lead_coord=self.localize(lead_bbox[0]+lead_bbox[2]/2,lead_bbox[1]+lead_bbox[3])
                vx,vy=t["velocity_px_s"]
                prefix = "PUBLIC" if self.mode == "public_area" else "DRONE" if self.mode == "human" else "FACTORY"
                result.append({"id":f"{prefix}-{int(t['id']):03d}","class":t["class_name"],"class_name":t["class_name"],"confidence":t["confidence"],"bbox":t["bbox"],"x_m":coord[0],"y_m":coord[1],"world_x":coord[0],"world_y":coord[1],"lead_x_m":lead_coord[0],"lead_y_m":lead_coord[1],"lead_x_px":round(lead_bbox[0]+lead_bbox[2]/2,1),"lead_y_px":round(lead_bbox[1]+lead_bbox[3],1),"projection_horizon_ms":round(t.get("projection_horizon_ms",self.latency_ms),1),"speed_px_s":round(math.hypot(vx,vy),1),"direction":self.direction(vx,vy),"state":t["state"],"age_frames":t["age"],"last_seen":now(),"position_kind":"calibrated camera plane" if coord[0] is not None else "image pixels","advisory_only":True})
            return result

    def localize(self,x,y):
        if cv2 is None or self.homography is None: return None,None
        import numpy as np
        p=cv2.perspectiveTransform(np.array([[[x,y]]],dtype="float32"),self.homography)[0][0]
        return round(float(p[0]),2),round(float(p[1]),2)

    @staticmethod
    def direction(vx,vy):
        if math.hypot(vx,vy)<8:return "STATIONARY"
        return ("S" if vy>0 else "N")+ ("E" if vx>0 else "W")

    def status(self):
        online=bool(not self.demo_mode and self.cap and self.cap.isOpened() and not self.camera_error)
        det_name = ("SIMULATED person fixture (not a live detector)" if self.demo_mode else
                    "local YOLO detector" if self.yolo_detector else
                    "OpenCV Zoo YOLOX-s COCO person detector" if self.yolox_detector else
                    "custom trained detector" if self.custom_detector else
                    "OpenCV HOG people baseline" if self.hog else
                    (self.detector_error or "Unavailable (install opencv-python)"))
        detector_active = (
            self.yolo_detector is not None or self.yolox_detector is not None
            or self.custom_detector is not None
            or (self.hog is not None and self.mode in ("human", "public_area"))
        )
        if self.mode == "industrial" and not detector_active:
            det_name = self.detector_error or "No industrial detector configured"
        if self.mode == "public_area" and not detector_active:
            det_name = self.detector_error or "No people detector available"
        tracker_name = getattr(self.tracker, "name", "unavailable")
        calibrated = self.homography is not None
        assistant_status = self.gemma_assistant.status()
        gesture_ready = bool(self.supported_gesture_classes)
        return {
            "product":"OFFLINE VISION-X",
            "mode":self.mode,
            "hud_mode":"PUBLIC AREA" if self.mode=="public_area" else "DRONE" if self.mode=="human" else "FACTORY",
            "camera_online":online,
            "camera_status":"online" if online else "unavailable",
            "camera_error":"" if self.demo_mode else self.camera_error,
            "camera_message":"Generated demo frames · no camera input" if self.demo_mode else self.camera_error or "Camera connected",
            "detector":det_name,
            "detector_error":self.detector_error,
            "tracker":tracker_name,
            "tracker_error":self.tracker_error,
            "pipeline_error":self.pipeline_error,
            "model_status":"SIMULATED FIXTURE (NOT LIVE)" if self.demo_mode else "LOCAL YOLO MODEL" if self.yolo_detector else "OPENCV ZOO YOLOX-S COCO MODEL" if self.yolox_detector else "CUSTOM TRAINED MODEL" if self.custom_detector else ("BASELINE" if self.hog and self.mode in ("human","public_area") else "NO DETECTOR"),
            "fps":round(self.fps,1),
            "processing_ms":round(self.processing_ms,1),
            "latency_ms":round(self.latency_ms,1),
            "lead_horizon_ms":round(min(1000.0,self.latency_ms),1),
            "telemetry_mode":"visual-only advisory",
            "control_enabled":False,
            "active_tracks":len(self.tracks),
            "detection_count":self.detection_count,
            "localization":"calibrated homography" if calibrated else "not calibrated",
            "localization_status":"Calibrated camera plane" if calibrated else "Positions unavailable · camera calibration required",
            "coordinate_unit":"METERS" if calibrated else "PIXELS",
            "room_width_m":config("localization.yaml").get("room_width_m",10),
            "room_height_m":config("localization.yaml").get("room_height_m",8),
            "assistant":assistant_status,
            "demo_mode":self.demo_mode,
            "public_area":{
                "enabled":self.mode=="public_area",
                "demo_mode":self.demo_mode,
                "monitor_available":self.public_area is not None,
                "configuration_error":self.public_area_error,
                "coordinate_space":"normalized_image",
                "configured_zones":len(self.public_area_config.zones) if self.public_area_config else 0,
                "sensitive_zones":list(self.public_area_config.sensitive_zones) if self.public_area_config else [],
                "gesture_status":(
                    "Experimental configured detector labels: " + ", ".join(sorted(self.supported_gesture_classes))
                    if gesture_ready else
                    "Unavailable: no configured gesture label is supported by the selected detector."
                ),
                "gesture_supported_labels":sorted(self.supported_gesture_classes),
                "gesture_configuration":list(self.public_area_config.gesture_classes) if self.public_area_config else [],
                "face_or_gaze_analysis":False,
                "notifications_enabled":False,
            },
            "external_ai_configured":assistant_status["available"],
            "offline":not assistant_status["available"],
        }


class Handler(BaseHTTPRequestHandler):
    app: VisionApp
    def log_message(self,*args): pass
    def send_json(self,obj,status=200):
        data=json.dumps(obj).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/api/status": return self.send_json(self.app.status())
        if path=="/api/assistant/status": return self.send_json(self.app.gemma_assistant.status())
        if path=="/api/tracks": return self.send_json(self.app.track_payload())
        if path=="/api/events": return self.send_json(list(self.app.events))
        if path=="/api/mode": return self.send_json({"mode":self.app.mode})
        if path=="/video.mjpg":
            if cv2 is None or self.app.cap is None or self.app.camera_error:
                self.send_error(503, "CAMERA UNAVAILABLE"); return
            self.send_response(200); self.send_header("Cache-Control","no-store"); self.send_header("Content-Type","multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            while True:
                if self.app.camera_error or self.app.cap is None: break
                with self.app.lock: frame=self.app.frame.copy() if self.app.frame is not None else None
                if frame is None:
                    time.sleep(.25); continue
                ok,buf=cv2.imencode(".jpg",frame)
                if not ok: continue
                try: self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"); time.sleep(.05)
                except (BrokenPipeError,ConnectionResetError): break
            return
        if path.startswith("/dashboard/"):
            name=Path(path.removeprefix("/dashboard/")).name
            if name in {"app.js","styles.css"}:
                file=ROOT/"dashboard"/name
                if not file.is_file():
                    file=ROOT/name
                if file.exists():
                    data=file.read_bytes(); self.send_response(200); self.send_header("Content-Type","text/javascript; charset=utf-8" if name.endswith(".js") else "text/css; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
        if path in ("/",""):
            file=ROOT/"dashboard"/"index.html"
            if not file.is_file():
                file=ROOT/"index.html"
            if file.exists():
                data=file.read_bytes(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/mode":
            if urlparse(self.path).path == "/api/assistant/scene":
                return self.do_assistant_scene()
            self.send_error(404); return
        try:
            length=int(self.headers.get("Content-Length","0")); body=json.loads(self.rfile.read(length) or b"{}")
            mode=body.get("mode")
            if mode not in ("human","industrial","public_area"):
                self.send_error(400,"mode must be human, industrial, or public_area"); return
            self.app.mode=mode
            status = self.app.status()
            self.send_json({"mode":mode,"hud_mode":status["hud_mode"],"model_status":status["model_status"],"tracker":status["tracker"],"telemetry_mode":status["telemetry_mode"],"control_enabled":False})
        except (ValueError,TypeError): self.send_error(400,"invalid JSON")

    def do_assistant_scene(self):
        origin = self.headers.get("Origin")
        if not origin or urlparse(origin).netloc != self.headers.get("Host"):
            return self.send_json({"error":"Scene assistant requests must come from this dashboard origin."},403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1024:
                return self.send_json({"error":"Request body is too large."},413)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError, json.JSONDecodeError):
            return self.send_json({"error":"Invalid JSON request."},400)
        confirmed = body.get("confirm_external_processing") is True
        include_frame = body.get("include_frame") is True
        frame_jpeg = None
        if include_frame:
            if not self.app.gemma_assistant.allow_frames:
                return self.send_json({"error":"Frame upload is disabled by server configuration."},403)
            if cv2 is None:
                return self.send_json({"error":"OpenCV is required to prepare an optional frame."},503)
            with self.app.lock:
                frame = self.app.frame.copy() if self.app.frame is not None else None
            if frame is None:
                return self.send_json({"error":"No camera frame is available to submit."},409)
            height, width = frame.shape[:2]
            scale = min(1.0, 640 / max(width, height))
            if scale < 1.0:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return self.send_json({"error":"Could not encode the optional camera frame."},500)
            frame_jpeg = encoded.tobytes()
        summary = scene_summary(
            self.app.status(),
            self.app.track_payload(),
            list(self.app.events),
        )
        try:
            text = self.app.gemma_assistant.summarize(
                summary,
                confirmed_external_processing=confirmed,
                image_jpeg=frame_jpeg,
            )
        except AssistantConfigurationError as exc:
            return self.send_json({"error":str(exc)},503)
        except AssistantRequestError as exc:
            return self.send_json({"error":str(exc)},502)
        return self.send_json({
            "text":text,
            "model":self.app.gemma_assistant.status()["model"],
            "frame_included":frame_jpeg is not None,
            "advisory_only":True,
        })


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--host",default=None); ap.add_argument("--port",type=int,default=8765); ap.add_argument("--camera",type=int,default=None); ap.add_argument("--mode",choices=["human","industrial","public_area"],default=None); ap.add_argument("--demo",action="store_true",help="run generated public-area frames and detections; not real camera input")
    args=ap.parse_args(); nc=config("network.yaml"); cc=config("camera.yaml")
    mode=args.mode or config("system.yaml").get("mode","public_area")
    app=VisionApp(args.camera if args.camera is not None else int(cc.get("device_id",0)), mode, demo=args.demo)
    Handler.app=app; app.start()
    host=args.host or nc.get("host","0.0.0.0")
    server=ThreadingHTTPServer((host,args.port),Handler)
    source = "SIMULATED demo frames (not live video)" if app.demo_mode else app.camera_error or "camera online"
    print(f"Offline Vision-X: http://127.0.0.1:{args.port}  (LAN bind {host}; {source})")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        app.running=False; server.server_close()
        if app.cap: app.cap.release()

if __name__=="__main__": main()
