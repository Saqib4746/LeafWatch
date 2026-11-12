"""Train a Leafwatch object detector and evaluate its best checkpoint."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

CLASSES = ["aphid", "whitefly", "caterpillar"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def audit(dataset):
    """Require every class in each split and reject exact cross-split copies."""
    seen = {}
    statistics = {}
    for split in ("train", "val", "test"):
        images = dataset / "images" / split
        labels = dataset / "labels" / split
        counts = Counter()
        total = negatives = 0
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels / (image.stem + ".txt")
            if not label.is_file():
                raise ValueError(f"Missing annotation for {image}; use an empty file for negatives.")
            digest = hashlib.sha256(image.read_bytes()).digest()
            if seen.setdefault(digest, split) != split:
                raise ValueError(f"Identical image in multiple splits: {image}")
            rows = label.read_text(encoding="utf-8-sig").split()
            if len(rows) % 5:
                raise ValueError(f"Invalid detection row in {label}")
            for offset in range(0, len(rows), 5):
                cls, x, y, width, height = rows[offset:offset + 5]
                cls = int(cls)
                x, y, width, height = map(float, (x, y, width, height))
                if (not 0 <= cls < len(CLASSES) or not 0 < width <= 1 or not 0 < height <= 1
                        or not width / 2 - 1e-5 <= x <= 1 - width / 2 + 1e-5
                        or not height / 2 - 1e-5 <= y <= 1 - height / 2 + 1e-5):
                    raise ValueError(f"Invalid bounding box in {label}")
                counts[CLASSES[cls]] += 1
            total += 1
            negatives += not rows
        if set(counts) != set(CLASSES):
            raise ValueError(f"{split} needs labeled instances for all classes; found {dict(counts)}.")
        statistics[split] = {"images": total, "negative_images": negatives, "boxes": dict(counts)}
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base", default="yolov8n.pt")
    parser.add_argument("--name", default="leafwatch")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    stats = audit(dataset)
    print(json.dumps(stats, indent=2), flush=True)
    if args.audit_only:
        return

    from ultralytics import YOLO

    config = dataset / "leafwatch.yaml"
    config.write_text(json.dumps({
        "path": dataset.as_posix(), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": CLASSES,
    }, indent=2), encoding="utf-8")
    model = YOLO(args.base)
    model.train(
        data=str(config), epochs=args.epochs, imgsz=args.size, batch=args.batch,
        device=args.device, workers=0, seed=42, deterministic=True, patience=12,
        project=str(Path("training-runs").resolve()), name=args.name,
    )
    run = Path(model.trainer.save_dir)
    best = run / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(
        data=str(config), split="test", imgsz=args.size, device=args.device,
        batch=args.batch, workers=0, project=str(run), name="held-out-test",
    )
    report = {
        "dataset": stats,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "per_class": {
            CLASSES[int(cls)]: {
                "precision": float(metrics.box.p[index]),
                "recall": float(metrics.box.r[index]),
                "AP50": float(metrics.box.ap50[index]),
                "AP50-95": float(metrics.box.ap[index]),
            }
            for index, cls in enumerate(metrics.box.ap_class_index)
        },
    }
    (run / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Checkpoint: {best}\nHeld-out evaluation: {run / 'evaluation.json'}", flush=True)


if __name__ == "__main__":
    main()
