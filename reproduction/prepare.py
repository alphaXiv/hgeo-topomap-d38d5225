from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .data import CAMERAS, X_MAX, X_MIN, Y_MAX, Y_MIN, discover_frames, load_image, split_frames


@torch.inference_mode()
def road_bev_per_camera(
    images,
    calibrations,
    processor,
    segmenter,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    logits = segmenter(pixel_values=inputs.pixel_values.to(device)).logits
    # Cityscapes label 0 is road. Keep probabilities per camera before IPM fusion.
    road = logits.softmax(1)[:, 0:1]
    xs = torch.linspace(X_MIN, X_MAX, width, device=device)
    ys = torch.linspace(Y_MIN, Y_MAX, height, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xyz = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).reshape(-1, 3)
    outputs = []
    for camera_index, calibration in enumerate(calibrations):
        rotation = torch.tensor(calibration["rotation"], dtype=torch.float32, device=device)
        translation = torch.tensor(calibration["translation"], dtype=torch.float32, device=device)
        intrinsic = torch.tensor(calibration["K"], dtype=torch.float32, device=device)
        camera_xyz = (xyz - translation) @ rotation
        projected = camera_xyz @ intrinsic.T
        depth = projected[:, 2]
        uv = projected[:, :2] / depth[:, None].clamp_min(1e-5)
        image_width, image_height = images[camera_index].size
        grid = torch.stack(
            [
                2.0 * uv[:, 0] / max(image_width - 1, 1) - 1.0,
                2.0 * uv[:, 1] / max(image_height - 1, 1) - 1.0,
            ],
            dim=-1,
        ).reshape(1, height, width, 2)
        sampled = F.grid_sample(
            road[camera_index : camera_index + 1],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]
        valid = (depth > 0).reshape(height, width)
        outputs.append(sampled * valid)
    return torch.stack(outputs).cpu().half()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("reproduction/config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    frames = discover_frames(args.data_root, config["num_points"])
    train_indices, val_indices = split_frames(frames)

    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    feature_net = torch.nn.Sequential(*list(backbone.children())[:-2]).to(device).eval()
    preprocess = weights.transforms()

    processor = segmenter = None
    if config["use_gal"]:
        model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        processor = SegformerImageProcessor.from_pretrained(model_name)
        segmenter = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()

    feature_frames, prior_frames = [], []
    for frame_index, frame in enumerate(frames):
        images = [load_image(path) for path in frame.image_paths]
        batch = torch.stack([preprocess(image) for image in images]).to(device)
        feature = feature_net(batch).cpu().half()
        feature_frames.append(feature)
        if config["use_gal"]:
            prior_frames.append(
                road_bev_per_camera(
                    images,
                    frame.calibrations,
                    processor,
                    segmenter,
                    config["bev_height"],
                    config["bev_width"],
                    device,
                )
            )
        print(f"PREP frame={frame_index + 1}/{len(frames)} token={frame.token}", flush=True)

    payload = {
        "features": torch.stack(feature_frames),
        "priors": torch.stack(prior_frames) if prior_frames else torch.empty(0),
        "lanes": [torch.from_numpy(frame.lanes) for frame in frames],
        "tokens": [frame.token for frame in frames],
        "segments": [frame.segment for frame in frames],
        "train_indices": train_indices,
        "val_indices": val_indices,
        "cameras": CAMERAS,
    }
    torch.save(payload, args.cache / "features.pt")
    print(
        "PREP_SUMMARY "
        + json.dumps(
            {
                "frames": len(frames),
                "train_frames": len(train_indices),
                "val_frames": len(val_indices),
                "cameras": len(CAMERAS),
                "uses_public_segmentation": bool(config["use_gal"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

