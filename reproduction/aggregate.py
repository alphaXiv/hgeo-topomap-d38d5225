from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(x) for x in array],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(path.read_text()) for path in sorted(args.results.glob("seed_*.json"))]
    if not results:
        raise RuntimeError("No completed seed results")
    aggregate = {
        "variant": results[0]["variant"],
        "benchmark": results[0]["benchmark"],
        "seeds": [result["seed"] for result in results],
        "train_frames": results[0]["train_frames"],
        "val_frames": results[0]["val_frames"],
        "normal_DET_l": summary([result["normal"]["DET_l"] for result in results]),
        "normal_AP": {
            key: summary([result["normal"][key] for result in results])
            for key in ("AP@1m", "AP@2m", "AP@3m")
        },
        "missing_view_DET_l": {
            camera: summary(
                [result["missing_view"][camera]["DET_l"] for result in results]
            )
            for camera in results[0]["missing_view"]
        },
        "seed_elapsed_seconds": summary([result["elapsed_seconds"] for result in results]),
    }
    print("FINAL_METRICS " + json.dumps(aggregate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

