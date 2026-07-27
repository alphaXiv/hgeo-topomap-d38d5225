from __future__ import annotations

import numpy as np


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    cache = np.full((len(a), len(b)), -1.0, dtype=np.float64)

    def recurse(i: int, j: int) -> float:
        if cache[i, j] >= 0:
            return float(cache[i, j])
        distance = float(np.linalg.norm(a[i] - b[j]))
        if i == 0 and j == 0:
            value = distance
        elif i > 0 and j == 0:
            value = max(recurse(i - 1, 0), distance)
        elif i == 0 and j > 0:
            value = max(recurse(0, j - 1), distance)
        else:
            value = max(
                min(recurse(i - 1, j), recurse(i - 1, j - 1), recurse(i, j - 1)),
                distance,
            )
        cache[i, j] = value
        return value

    return recurse(len(a) - 1, len(b) - 1)


def eleven_point_ap(records: list[tuple[float, int]], num_gt: int) -> float:
    if num_gt == 0:
        return 1.0
    records.sort(key=lambda item: -item[0])
    tp = np.cumsum([item[1] for item in records])
    fp = np.cumsum([1 - item[1] for item in records])
    recall = tp / max(num_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    return float(
        np.mean([precision[recall >= threshold].max() if np.any(recall >= threshold) else 0.0
                 for threshold in np.linspace(0, 1, 11)])
    )


def centerline_det_l(
    ground_truth: list[list[np.ndarray]], predictions: list[list[tuple[float, np.ndarray]]]
) -> dict[str, float]:
    aps = {}
    for threshold in (1.0, 2.0, 3.0):
        records: list[tuple[float, int]] = []
        num_gt = sum(len(frame) for frame in ground_truth)
        for gt_frame, pred_frame in zip(ground_truth, predictions):
            covered: set[int] = set()
            for confidence, pred in sorted(pred_frame, key=lambda item: -item[0]):
                distances = [discrete_frechet(gt, pred) for gt in gt_frame]
                if distances:
                    index = int(np.argmin(distances))
                    matched = distances[index] < threshold and index not in covered
                else:
                    matched = False
                    index = -1
                if matched:
                    covered.add(index)
                records.append((confidence, int(matched)))
        aps[f"AP@{threshold:.0f}m"] = eleven_point_ap(records, num_gt)
    aps["DET_l"] = float(np.mean(list(aps.values())))
    return aps

