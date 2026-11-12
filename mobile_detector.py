"""Train and run a lightweight MobileNetV3 SSDLite pest detector."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import box_iou
from torchvision.transforms.functional import pil_to_tensor

from train import CLASSES, IMAGE_SUFFIXES, audit


def build_model():
    return ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(CLASSES) + 1,
        score_thresh=0.02, detections_per_img=100, topk_candidates=200,
    )


class PestPhotos(Dataset):
    def __init__(self, root, split):
        self.images = sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.labels = root / "labels" / split
        self.augment = split == "train"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as source:
            image = pil_to_tensor(source.convert("RGB")).float() / 255
        height, width = image.shape[-2:]
        boxes, labels = [], []
        for row in (self.labels / (path.stem + ".txt")).read_text().splitlines():
            cls, x, y, w, h = map(float, row.split())
            boxes.append([(x - w / 2) * width, (y - h / 2) * height,
                          (x + w / 2) * width, (y + h / 2) * height])
            labels.append(int(cls) + 1)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment and random.random() < .5:
            image = image.flip(-1)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return image, {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}


def batch_items(batch):
    return tuple(zip(*batch))


def evaluate(model, loader, device):
    model.eval()
    matches = {cls: [] for cls in range(1, len(CLASSES) + 1)}
    totals = dict.fromkeys(matches, 0)
    with torch.inference_mode():
        for images, targets in loader:
            predictions = model([image.to(device) for image in images])
            for predicted, truth in zip(predictions, targets):
                for cls, rows in matches.items():
                    actual = truth["boxes"][truth["labels"] == cls]
                    totals[cls] += len(actual)
                    mask = predicted["labels"] == cls
                    overlap = box_iou(predicted["boxes"][mask].cpu(), actual)
                    used = torch.zeros(len(actual), dtype=torch.bool)
                    for confidence, ious in zip(predicted["scores"][mask].tolist(), overlap):
                        ious[used] = 0
                        found = bool(len(ious) and ious.max() >= .5)
                        if found:
                            used[ious.argmax()] = True
                        rows.append((confidence, found))
    results = {}
    for cls, rows in matches.items():
        precision, recall = [], []
        hits = selected = selected_hits = 0
        for rank, (score, found) in enumerate(sorted(rows, reverse=True), start=1):
            hits += found
            precision.append(hits / rank)
            recall.append(hits / totals[cls])
            if score >= .35:
                selected += 1
                selected_hits += found
        ap = sum(max((p for p, r in zip(precision, recall) if r >= k / 100), default=0.0)
                 for k in range(101)) / 101
        results[CLASSES[cls - 1]] = {
            "AP50": ap,
            "precision_at_0.35": selected_hits / max(1, selected),
            "recall_at_0.35": selected_hits / totals[cls],
            "test_boxes": totals[cls],
        }
    return {"mAP50": sum(row["AP50"] for row in results.values()) / len(results), "per_class": results}


def train(args):
    dataset = args.dataset.resolve()
    statistics = audit(dataset)
    print(json.dumps(statistics, indent=2), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = build_model()
    pretrained = torch.load(args.base, map_location="cpu", weights_only=True)
    # Keep the pretrained feature extractor and box regression. The class head
    # learns the three pest labels and the background class from the dataset.
    transferable = {key: tensor for key, tensor in pretrained.items()
                    if not key.startswith("head.classification_head.")}
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    if unexpected or not missing or any(not key.startswith("head.classification_head.") for key in missing):
        raise ValueError("Starting weights do not match SSDLite320 MobileNetV3.")
    model.backbone.requires_grad_(False)
    model.to(device)
    loaders = {
        split: DataLoader(PestPhotos(dataset, split), batch_size=args.batch,
                          shuffle=split == "train", collate_fn=batch_items)
        for split in ("train", "val", "test")
    }
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_score = -1.0
    history = []
    best_path = args.output / "best.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()
        # The smallest SSD feature map is 1x1. Frozen BN statistics also
        # allow the final batch to contain only one training image.
        for module in model.head.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        loss_sum = 0.0
        for step, (images, targets) in enumerate(loaders["train"], start=1):
            losses = model([image.to(device) for image in images],
                           [{key: value.to(device) for key, value in target.items()} for target in targets])
            loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError("Training produced a non-finite loss.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if step % 15 == 0:
                print(f"Epoch {epoch}/{args.epochs} batch {step}/{len(loaders['train'])}, loss {loss_sum / step:.3f}", flush=True)
        scheduler.step()
        validation = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "loss": loss_sum / len(loaders["train"]), "val_mAP50": validation["mAP50"]})
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if validation["mAP50"] > best_score:
            best_score = validation["mAP50"]
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch}: validation mAP50 {validation['mAP50']:.3f}", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    results = {"dataset": statistics, "best_validation_mAP50": best_score,
               "test": evaluate(model, loaders["test"], device)}
    (args.output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


class MobileDetector:
    def __init__(self, weights):
        import threading

        self.model = build_model()
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.eval()
        self.lock = threading.Lock()

    def detect(self, image):
        width, height = image.size
        inputs = pil_to_tensor(image).float() / 255
        with self.lock, torch.inference_mode():
            results = self.model([inputs])[0]
        detections = []
        for score, label, (x1, y1, x2, y2) in zip(
            results["scores"].tolist(), results["labels"].tolist(), results["boxes"].tolist()
        ):
            if score >= .35 and x2 > x1 and y2 > y1:
                detections.append({"pest": CLASSES[label - 1], "confidence": score,
                                   "box": [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]})
        return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    train(parser.parse_args())
