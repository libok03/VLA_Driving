from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


YOLO_CLASSES = {
    0: 0,   # person
    1: 1,   # bicycle
    2: 2,   # car
    3: 2,   # motorcycle
    5: 2,   # bus
    7: 2,   # truck
    9: 3,   # traffic light
    11: 3,  # stop sign
}


@dataclass
class PerceptionConfig:
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    yolo_enabled: bool = True
    lane_roi_y: float = 0.55
    canny_low: int = 60
    canny_high: int = 160
    dim: int = 32


class PerceptionExtractor:
    """Convert a camera frame into compact driving features."""

    def __init__(self, cfg: dict[str, Any] | None = None, dim: int = 32) -> None:
        cfg = cfg or {}
        self.cfg = PerceptionConfig(
            yolo_model=str(cfg.get("yolo_model", "yolov8n.pt")),
            yolo_conf=float(cfg.get("yolo_conf", 0.35)),
            yolo_enabled=bool(cfg.get("yolo_enabled", True)),
            lane_roi_y=float(cfg.get("lane_roi_y", 0.55)),
            canny_low=int(cfg.get("canny_low", 60)),
            canny_high=int(cfg.get("canny_high", 160)),
            dim=int(cfg.get("dim", dim)),
        )
        self._yolo = None
        self._yolo_load_failed = False

    @property
    def dim(self) -> int:
        return self.cfg.dim

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        image_rgb = self._ensure_rgb(image_rgb)
        lane = self._lane_features(image_rgb)
        objects = self._object_features(image_rgb)
        features = np.concatenate([lane, objects]).astype(np.float32)
        fitted = np.zeros(self.dim, dtype=np.float32)
        fitted[: min(self.dim, features.shape[0])] = features[: self.dim]
        return fitted

    def _lane_features(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        roi_top = int(np.clip(self.cfg.lane_roi_y, 0.0, 0.95) * height)
        roi = image_rgb[roi_top:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.cfg.canny_low, self.cfg.canny_high)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=35,
            minLineLength=max(20, width // 12),
            maxLineGap=25,
        )

        left: list[tuple[float, float, float]] = []
        right: list[tuple[float, float, float]] = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                if abs(dx) < 1.0:
                    continue
                slope = dy / dx
                if abs(slope) < 0.35:
                    continue
                length = float(np.hypot(dx, dy))
                x_bottom = self._x_at_y(float(x1), float(y1), float(x2), float(y2), float(roi.shape[0] - 1))
                if slope < 0:
                    left.append((x_bottom, slope, length))
                else:
                    right.append((x_bottom, slope, length))

        left_x, left_slope, left_conf = self._weighted_lane(left, default_x=width * 0.25)
        right_x, right_slope, right_conf = self._weighted_lane(right, default_x=width * 0.75)
        lane_center = (left_x + right_x) * 0.5
        offset = (lane_center - width * 0.5) / max(width * 0.5, 1.0)
        heading = np.arctan((left_slope + right_slope) * 0.5) / (np.pi * 0.5)
        lane_width = (right_x - left_x) / max(width, 1.0)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        return np.asarray(
            [
                np.clip(offset, -1.0, 1.0),
                np.clip(heading, -1.0, 1.0),
                np.clip(lane_width, 0.0, 2.0),
                left_conf,
                right_conf,
                np.clip(edge_density * 10.0, 0.0, 1.0),
                np.clip(left_slope / 5.0, -1.0, 1.0),
                np.clip(right_slope / 5.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _object_features(self, image_rgb: np.ndarray) -> np.ndarray:
        detections = self._detect_objects(image_rgb)
        height, width = image_rgb.shape[:2]
        traffic_state = self._traffic_light_state_features(image_rgb, detections)
        class_bins = np.zeros(4, dtype=np.float32)
        closest_area = np.zeros(4, dtype=np.float32)
        closest_x = np.zeros(4, dtype=np.float32)
        closest_y = np.zeros(4, dtype=np.float32)
        best_conf = np.zeros(4, dtype=np.float32)
        total_area = 0.0

        for det in detections:
            group = YOLO_CLASSES.get(int(det["class_id"]), 3)
            x1, y1, x2, y2 = det["bbox"]
            box_w = max(float(x2 - x1), 0.0)
            box_h = max(float(y2 - y1), 0.0)
            area = (box_w * box_h) / max(float(width * height), 1.0)
            cx = (((x1 + x2) * 0.5) / max(width, 1)) * 2.0 - 1.0
            cy = ((y1 + y2) * 0.5) / max(height, 1)
            conf = float(det["confidence"])
            class_bins[group] += 1.0
            total_area += area
            if area > closest_area[group]:
                closest_area[group] = area
                closest_x[group] = float(cx)
                closest_y[group] = float(cy)
                best_conf[group] = conf

        class_bins = np.clip(class_bins / 8.0, 0.0, 1.0)
        return np.concatenate(
            [
                traffic_state,
                class_bins,
                np.clip(closest_area * 10.0, 0.0, 1.0),
                np.clip(closest_x, -1.0, 1.0),
                np.clip(closest_y, 0.0, 1.0),
                np.clip(best_conf, 0.0, 1.0),
                np.asarray([np.clip(total_area * 5.0, 0.0, 1.0)], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _traffic_light_state_features(
        self,
        image_rgb: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        best_crop = None
        best_conf = 0.0
        for det in detections:
            if int(det["class_id"]) != 9:
                continue
            conf = float(det["confidence"])
            if conf <= best_conf:
                continue
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            best_crop = image_rgb[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
            best_conf = conf

        if best_crop is None or best_crop.size == 0:
            best_crop, best_conf = self._find_traffic_light_crop_by_color(image_rgb)

        color_scores = self._traffic_light_color_scores(best_crop)
        state = self._state_from_color_scores(color_scores)
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[state] = 1.0
        confidence = float(np.clip(max(best_conf, color_scores.max()), 0.0, 1.0)) if state != 0 else 0.0
        return np.concatenate([color_scores, one_hot, np.asarray([confidence], dtype=np.float32)])

    def _find_traffic_light_crop_by_color(self, image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image_rgb.shape[:2]
        roi_bottom = max(int(height * 0.55), 1)
        roi = image_rgb[:roi_bottom, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        color_mask = self._traffic_color_mask(hsv)
        kernel = np.ones((3, 3), dtype=np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.dilate(color_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes: list[tuple[int, int, int, int, float]] = []
        best_box: tuple[int, int, int, int] | None = None
        best_score = 0.0

        image_area = float(max(width * height, 1))
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 8:
                continue
            area = float(w * h)
            if area < image_area * 0.0002 or area > image_area * 0.08:
                continue
            aspect = w / max(float(h), 1.0)
            if not (0.45 <= aspect <= 1.8):
                continue

            pad = int(max(w, h) * 0.35)
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, width)
            y2 = min(y + h + pad, roi_bottom)
            crop_mask = color_mask[y1:y2, x1:x2]
            color_ratio = float(np.count_nonzero(crop_mask)) / max(float(crop_mask.size), 1.0)
            circularity = self._contour_circularity(contour)
            score = color_ratio * 0.7 + circularity * 0.3
            candidate_boxes.append((x1, y1, x2, y2, score))
            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)

        if not candidate_boxes:
            return roi, 0.0

        strong_boxes = [box for box in candidate_boxes if box[4] >= max(best_score * 0.45, 0.05)]
        if len(strong_boxes) >= 2:
            x1 = min(box[0] for box in strong_boxes)
            y1 = min(box[1] for box in strong_boxes)
            x2 = max(box[2] for box in strong_boxes)
            y2 = max(box[3] for box in strong_boxes)
        else:
            assert best_box is not None
            x1, y1, x2, y2 = best_box
        return roi[y1:y2, x1:x2], float(np.clip(best_score, 0.0, 1.0))

    @staticmethod
    def _classify_traffic_light_color(image_rgb: np.ndarray) -> int:
        return PerceptionExtractor._state_from_color_scores(
            PerceptionExtractor._traffic_light_color_scores(image_rgb)
        )

    @staticmethod
    def _traffic_light_color_scores(image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.size == 0:
            return np.zeros(4, dtype=np.float32)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        red, yellow, green = PerceptionExtractor._traffic_color_masks(hsv)
        counts = np.asarray(
            [np.count_nonzero(red), np.count_nonzero(yellow), np.count_nonzero(green)],
            dtype=np.float32,
        )
        min_pixels = max(float(image_rgb.shape[0] * image_rgb.shape[1]) * 0.002, 3.0)
        area = max(float(image_rgb.shape[0] * image_rgb.shape[1]), 1.0)
        color_scores = np.clip(counts / area, 0.0, 1.0)
        total_score = float(np.clip(counts.sum() / area, 0.0, 1.0))
        if float(counts.max()) < min_pixels:
            return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray(
            [color_scores[0], color_scores[1], color_scores[2], total_score],
            dtype=np.float32,
        )

    @staticmethod
    def _state_from_color_scores(scores: np.ndarray) -> int:
        if scores.size < 4 or float(scores[3]) <= 0.0:
            return 0
        return int(np.argmax(scores[:3])) + 1

    @staticmethod
    def _traffic_color_mask(hsv: np.ndarray) -> np.ndarray:
        masks = PerceptionExtractor._traffic_color_masks(hsv)
        return masks[0] | masks[1] | masks[2]

    @staticmethod
    def _traffic_color_masks(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        red = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)) | cv2.inRange(
            hsv, (170, 80, 80), (180, 255, 255)
        )
        yellow = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
        green = cv2.inRange(hsv, (45, 60, 60), (95, 255, 255))
        return red, yellow, green

    @staticmethod
    def _contour_circularity(contour: np.ndarray) -> float:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0.0 or perimeter <= 1e-6:
            return 0.0
        return float(np.clip((4.0 * np.pi * area) / (perimeter * perimeter), 0.0, 1.0))

    def _detect_objects(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        model = self._load_yolo()
        if model is None:
            return []
        results = model.predict(image_rgb, conf=self.cfg.yolo_conf, verbose=False)
        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                detections.append(
                    {
                        "bbox": box.xyxy[0].detach().cpu().numpy().astype(float).tolist(),
                        "class_id": int(box.cls[0].detach().cpu().item()),
                        "confidence": float(box.conf[0].detach().cpu().item()),
                    }
                )
        return detections

    def _load_yolo(self):
        if not self.cfg.yolo_enabled or self._yolo_load_failed:
            return None
        if self._yolo is not None:
            return self._yolo
        try:
            from ultralytics import YOLO

            self._yolo = YOLO(self.cfg.yolo_model)
        except Exception:
            self._yolo_load_failed = True
            return None
        return self._yolo

    @staticmethod
    def _ensure_rgb(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[2] == 4:
            return image[:, :, :3]
        return image

    @staticmethod
    def _x_at_y(x1: float, y1: float, x2: float, y2: float, target_y: float) -> float:
        if abs(y2 - y1) < 1e-3:
            return (x1 + x2) * 0.5
        return x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)

    @staticmethod
    def _weighted_lane(lines: list[tuple[float, float, float]], default_x: float) -> tuple[float, float, float]:
        if not lines:
            return float(default_x), 0.0, 0.0
        weights = np.asarray([line[2] for line in lines], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-6)
        xs = np.asarray([line[0] for line in lines], dtype=np.float32)
        slopes = np.asarray([line[1] for line in lines], dtype=np.float32)
        confidence = float(np.clip(len(lines) / 8.0, 0.0, 1.0))
        return float((xs * weights).sum()), float((slopes * weights).sum()), confidence
