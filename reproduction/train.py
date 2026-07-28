from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .data import CAMERAS, geometry_group, metric_xy
from .metrics import centerline_det_l
from .model import LaneDETR, supervised_contrastive


def match_predictions(logits: torch.Tensor, points: torch.Tensor, targets: torch.Tensor):
    with torch.no_grad():
        point_cost = torch.cdist(points.flatten(1), targets.flatten(1), p=1) / points[0].numel()
        class_cost = -logits.sigmoid()[:, None].expand_as(point_cost)
        pred_indices, target_indices = linear_sum_assignment((point_cost - 0.25 * class_cost).cpu())
    return (
        torch.as_tensor(pred_indices, device=points.device),
        torch.as_tensor(target_indices, device=points.device),
    )


def compute_loss(outputs: dict, targets: list[torch.Tensor], config: dict) -> tuple[torch.Tensor, dict]:
    classification = outputs["logits"].new_zeros(())
    regression = outputs["logits"].new_zeros(())
    contrastive_embeddings, contrastive_groups = [], []
    for batch_index, target in enumerate(targets):
        pred_index, target_index = match_predictions(
            outputs["logits"][batch_index], outputs["points"][batch_index], target
        )
        labels = torch.zeros_like(outputs["logits"][batch_index])
        labels[pred_index] = 1
        classification = classification + F.binary_cross_entropy_with_logits(
            outputs["logits"][batch_index], labels
        )
        regression = regression + F.l1_loss(
            outputs["points"][batch_index, pred_index], target[target_index]
        )
        if config["use_gcl"]:
            for p_idx, t_idx in zip(pred_index.tolist(), target_index.tolist()):
                contrastive_embeddings.append(outputs["embeddings"][batch_index, p_idx])
                # Negative control: preserve the contrastive loss and group
                # cardinality while severing labels from geometric orientation.
                contrastive_groups.append((p_idx * 7 + t_idx) % 3)
    classification = classification / len(targets)
    regression = regression / len(targets)
    gcl = (
        supervised_contrastive(contrastive_embeddings, contrastive_groups)
        if config["use_gcl"]
        else classification * 0
    )
    total = classification + 5.0 * regression + config["gcl_weight"] * gcl
    return total, {
        "loss_cls": float(classification.detach()),
        "loss_points": float(regression.detach()),
        "loss_gcl": float(gcl.detach()),
    }


@torch.inference_mode()
def evaluate(model, cache, indices, device, missing_camera=None):
    model.eval()
    ground_truth, predictions = [], []
    for index in indices:
        features = cache["features"][index : index + 1].to(device).float()
        priors = (
            cache["priors"][index : index + 1].to(device).float()
            if cache["priors"].numel()
            else None
        )
        outputs = model(features, priors, missing_camera=missing_camera)
        confidence = outputs["logits"][0].sigmoid().cpu().numpy()
        points = metric_xy(outputs["points"][0].cpu().numpy())
        keep = np.argsort(-confidence)[:32]
        predictions.append([(float(confidence[i]), points[i]) for i in keep])
        ground_truth.append([metric_xy(x.numpy()) for x in cache["lanes"][index]])
    return centerline_det_l(ground_truth, predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("reproduction/config.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    cache = torch.load(args.cache / "features.pt", map_location="cpu", weights_only=False)
    _, _, channels, height, width = cache["features"].shape
    model = LaneDETR(config, channels, (height, width)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    generator = np.random.default_rng(args.seed)
    train_indices = np.asarray(cache["train_indices"])
    last_losses = {}
    model.train()
    for step in range(1, config["steps"] + 1):
        batch_indices = generator.choice(
            train_indices, size=config["batch_size"], replace=len(train_indices) < config["batch_size"]
        )
        features = cache["features"][batch_indices].to(device).float()
        priors = (
            cache["priors"][batch_indices].to(device).float() if cache["priors"].numel() else None
        )
        targets = [cache["lanes"][int(index)].to(device).float() for index in batch_indices]
        outputs = model(features, priors)
        loss, last_losses = compute_loss(outputs, targets, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 200 == 0:
            print(
                f"TRAIN variant={config['variant']} seed={args.seed} step={step}/{config['steps']} "
                f"loss={float(loss):.5f} cls={last_losses['loss_cls']:.5f} "
                f"points={last_losses['loss_points']:.5f} gcl={last_losses['loss_gcl']:.5f}",
                flush=True,
            )

    normal = evaluate(model, cache, cache["val_indices"], device)
    missing = {}
    for camera_index, camera_name in enumerate(CAMERAS):
        missing[camera_name] = evaluate(
            model, cache, cache["val_indices"], device, missing_camera=camera_index
        )
    result = {
        "variant": config["variant"],
        "seed": args.seed,
        "benchmark": "OpenLane-V2 public sample, temporal-stratified held-out frames",
        "train_frames": len(cache["train_indices"]),
        "val_frames": len(cache["val_indices"]),
        "normal": normal,
        "missing_view": missing,
        "elapsed_seconds": time.time() - start,
        "final_losses": last_losses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("SEED_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
