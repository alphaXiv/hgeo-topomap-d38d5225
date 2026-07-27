from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


CAMERAS = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
]
X_MIN, X_MAX = -51.2, 51.2
Y_MIN, Y_MAX = -25.6, 25.6


def resample_polyline(points: np.ndarray, n: int = 11) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, n, axis=0)
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distance)))
    if cumulative[-1] < 1e-6:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, cumulative[-1], n)
    return np.stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(points.shape[1])],
        axis=-1,
    ).astype(np.float32)


def normalize_xy(points: np.ndarray) -> np.ndarray:
    result = points[..., :2].copy()
    result[..., 0] = (result[..., 0] - X_MIN) / (X_MAX - X_MIN)
    result[..., 1] = (result[..., 1] - Y_MIN) / (Y_MAX - Y_MIN)
    return np.clip(result, 0.0, 1.0)


def metric_xy(points: np.ndarray) -> np.ndarray:
    result = points.copy()
    result[..., 0] = result[..., 0] * (X_MAX - X_MIN) + X_MIN
    result[..., 1] = result[..., 1] * (Y_MAX - Y_MIN) + Y_MIN
    return result


def geometry_group(points: torch.Tensor) -> int:
    """Paper-inspired groups: two dominant straight orientations and curved."""
    delta = points[1:] - points[:-1]
    angles = torch.atan2(delta[:, 1], delta[:, 0])
    if len(angles) > 1:
        bend = torch.atan2(
            torch.sin(angles[1:] - angles[:-1]), torch.cos(angles[1:] - angles[:-1])
        ).abs().mean()
    else:
        bend = torch.tensor(0.0, device=points.device)
    chord = torch.linalg.vector_norm(points[-1] - points[0])
    arc = torch.linalg.vector_norm(delta, dim=-1).sum().clamp_min(1e-6)
    if bend > 0.12 or chord / arc < 0.94:
        return 2
    mean_angle = torch.atan2(delta[:, 1].mean(), delta[:, 0].mean())
    return int(torch.abs(torch.sin(mean_angle)) > 0.707)


@dataclass
class FrameRecord:
    token: str
    segment: str
    timestamp: int
    json_path: Path
    image_paths: list[Path]
    calibrations: list[dict]
    lanes: np.ndarray


def discover_frames(data_root: Path, num_points: int = 11) -> list[FrameRecord]:
    frames: list[FrameRecord] = []
    for path in sorted(data_root.glob("train/*/info/*.json")):
        raw = json.loads(path.read_text())
        lanes = []
        for lane in raw["annotation"]["lane_centerline"]:
            pts = np.asarray(lane["points"], dtype=np.float32)
            inside = (
                (pts[:, 0] >= X_MIN)
                & (pts[:, 0] <= X_MAX)
                & (pts[:, 1] >= Y_MIN)
                & (pts[:, 1] <= Y_MAX)
            )
            if inside.sum() < 2:
                continue
            clipped = pts[inside]
            lanes.append(normalize_xy(resample_polyline(clipped, num_points)))
        if not lanes:
            continue
        sensors = raw["sensor"]
        image_paths = [data_root / sensors[name]["image_path"] for name in CAMERAS]
        calibrations = [
            {
                "K": sensors[name]["intrinsic"]["K"],
                "rotation": sensors[name]["extrinsic"]["rotation"],
                "translation": sensors[name]["extrinsic"]["translation"],
            }
            for name in CAMERAS
        ]
        frames.append(
            FrameRecord(
                token=f'{raw["segment_id"]}_{raw["timestamp"]}',
                segment=raw["segment_id"],
                timestamp=int(raw["timestamp"]),
                json_path=path,
                image_paths=image_paths,
                calibrations=calibrations,
                lanes=np.stack(lanes),
            )
        )
    return frames


def split_frames(frames: list[FrameRecord]) -> tuple[list[int], list[int]]:
    """A deterministic 3:1 temporal-stratified split within both public sample scenes."""
    train, val = [], []
    for segment in sorted({frame.segment for frame in frames}):
        indices = [i for i, frame in enumerate(frames) if frame.segment == segment]
        indices.sort(key=lambda i: frames[i].timestamp)
        for rank, index in enumerate(indices):
            (val if rank % 4 == 0 else train).append(index)
    return train, val


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

