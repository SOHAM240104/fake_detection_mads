from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
try:
    import scipy.ndimage as ndi
except Exception:  # pragma: no cover - fallback path for minimal envs
    ndi = None


@dataclass
class RegionTrack:
    frames: List[int]
    boxes: List[Tuple[int, int, int, int]]  # (x_min, y_min, x_max, y_max)
    mean_cam: List[float]


def cam_to_binary_masks(cam: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """
    Threshold CAM volume into binary masks per frame.

    Args:
        cam: (T, H, W) CAM values.
        percentile: percentile per frame used as threshold.

    Returns:
        masks: (T, H, W) boolean.
    """
    cam = np.asarray(cam, dtype=float)
    if cam.ndim != 3:
        raise ValueError("cam must be 3D (T, H, W)")
    T, H, W = cam.shape
    masks = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        c = cam[t]
        thr = np.percentile(c, percentile)
        masks[t] = c >= thr
    return masks


def _extract_boxes_for_frame(mask: np.ndarray, cam_frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
    if ndi is not None:
        labeled, num = ndi.label(mask)
    else:
        import cv2

        n, labeled = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        num = int(max(0, n - 1))
    boxes_and_scores: List[Tuple[Tuple[int, int, int, int], float]] = []
    for label in range(1, num + 1):
        ys, xs = np.where(labeled == label)
        if ys.size == 0:
            continue
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        box = (int(x_min), int(y_min), int(x_max), int(y_max))
        mean_cam = float(cam_frame[ys, xs].mean())
        boxes_and_scores.append((box, mean_cam))
    return boxes_and_scores


def _iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def track_regions_iou(masks: np.ndarray, cam: np.ndarray, iou_threshold: float = 0.3) -> List[RegionTrack]:
    """
    Track CAM regions across frames via IoU-based association.

    Args:
        masks: (T, H, W) boolean masks.
        cam: (T, H, W) CAM values aligned with masks.
        iou_threshold: IoU threshold for association between frames.

    Returns:
        List of RegionTrack objects.
    """
    masks = np.asarray(masks, dtype=bool)
    cam = np.asarray(cam, dtype=float)
    if masks.shape != cam.shape:
        raise ValueError("masks and cam must have the same shape")
    T, _, _ = masks.shape

    tracks: List[RegionTrack] = []

    prev_boxes: List[Tuple[int, int, int, int]] = []
    prev_track_idx: List[int] = []

    for t in range(T):
        frame_boxes = _extract_boxes_for_frame(masks[t], cam[t])
        if not frame_boxes:
            prev_boxes = []
            prev_track_idx = []
            continue

        boxes_t = [b for b, _ in frame_boxes]
        scores_t = [s for _, s in frame_boxes]

        assigned = [False] * len(boxes_t)
        current_track_idx = [-1] * len(boxes_t)

        # Try to match to existing tracks from t-1.
        for j, (b_prev, track_idx) in enumerate(zip(prev_boxes, prev_track_idx)):
            best_iou = 0.0
            best_k = -1
            for k, b in enumerate(boxes_t):
                if assigned[k]:
                    continue
                iou = _iou(b_prev, b)
                if iou > best_iou:
                    best_iou = iou
                    best_k = k
            if best_k >= 0 and best_iou >= iou_threshold:
                # Extend existing track.
                tracks[track_idx].frames.append(t)
                tracks[track_idx].boxes.append(boxes_t[best_k])
                tracks[track_idx].mean_cam.append(scores_t[best_k])
                assigned[best_k] = True
                current_track_idx[best_k] = track_idx

        # Start new tracks for unassigned boxes.
        for k, b in enumerate(boxes_t):
            if assigned[k]:
                continue
            tracks.append(RegionTrack(frames=[t], boxes=[b], mean_cam=[scores_t[k]]))
            current_track_idx[k] = len(tracks) - 1

        prev_boxes = boxes_t
        prev_track_idx = current_track_idx

    return tracks


def summarize_region_anomalies(tracks: List[RegionTrack]) -> dict:
    """
    Summarize region tracks into a simple JSON-serializable structure.
    """
    summary = []
    for tr in tracks:
        if not tr.frames:
            continue
        duration = tr.frames[-1] - tr.frames[0] + 1
        summary.append(
            {
                "start_frame": int(tr.frames[0]),
                "end_frame": int(tr.frames[-1]),
                "duration_frames": int(duration),
                "mean_cam": float(np.mean(tr.mean_cam)),
                "max_cam": float(np.max(tr.mean_cam)),
                "boxes": tr.boxes,
            }
        )
    return {"tracks": summary}

