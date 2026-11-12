"""Convert the academic IP102 detection archives into Leafwatch's three classes."""
import argparse
import hashlib
import random
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from train import CLASSES, audit

APHIDS = {
    "aphid", "aphids", "english grain aphid", "green bug", "bird cherry-oataphid",
    "therioaphis maculata buckton", "toxoptera citricidus",
    "toxoptera aurantii", "aphis citricola vander goot",
}
WHITEFLIES = {"whitefly", "whiteflies", "trialeurodes vaporariorum", "aleurocanthus spiniferus"}
CATERPILLARS = {
    "caterpillar", "rice leaf caterpillar", "rice leaf roller", "black cutworm",
    "large cutworm", "yellow cutworm", "army worm", "cabbage army worm",
    "beet army worm", "flax budworm", "prodenia litura",
}


def target(name):
    name = " ".join(name.casefold().split())
    return next((i for i, names in enumerate((APHIDS, WHITEFLIES, CATERPILLARS)) if name in names), None)


def read_classes(path):
    entries = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            index, name = line.split(maxsplit=1)
            entries[str(int(index) - 1)] = name.strip()
    return entries


def read_annotations(archive, classes):
    labels = {}
    vocabulary = Counter()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".xml"):
                continue
            if member.size > 2_000_000:
                raise ValueError(f"Unexpected annotation size: {member.name}")
            try:
                root = ET.fromstring(tar.extractfile(member).read())
            except ET.ParseError:
                print(f"Skipping malformed annotation: {member.name}")
                continue
            width = float(root.findtext("size/width"))
            height = float(root.findtext("size/height"))
            boxes = []
            for obj in root.findall("object"):
                label = obj.findtext("name", "")
                name = classes.get(label, label)
                vocabulary[name] += 1
                cls = target(name)
                if cls is None:
                    continue
                x1, y1, x2, y2 = (
                    float(obj.findtext("bndbox/" + key))
                    for key in ("xmin", "ymin", "xmax", "ymax")
                )
                x1, x2 = max(0, x1 - 1), min(width, x2)
                y1, y2 = max(0, y1 - 1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                coords = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                          (x2 - x1) / width, (y2 - y1) / height)
                boxes.append(str(cls) + " " + " ".join(f"{value:.8f}" for value in coords))
            labels[Path(member.name).stem] = boxes
    print("Mapped annotation counts:", {name: count for name, count in vocabulary.items() if target(name) is not None})
    if {int(row.split()[0]) for rows in labels.values() for row in rows} != set(range(len(CLASSES))):
        raise ValueError(f"The archive lacks target class annotations. Label vocabulary: {dict(vocabulary)}")
    return labels


def read_ids(path):
    ids = {line.split()[0] for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    if not ids or not all(re.fullmatch(r"(?:IP)?[0-9]+", value) for value in ids):
        raise ValueError(f"Expected an official IP102 image ID list in {path}")
    return ids


def image_members(tar, archive_size, allow_prefix):
    try:
        for member in tar:
            if allow_prefix and member.offset_data + member.size > archive_size:
                return
            if member.isfile() and Path(member.name).suffix.casefold() in {".jpg", ".jpeg"}:
                yield member
    except tarfile.ReadError as exc:
        if not allow_prefix or "unexpected end of data" not in str(exc):
            raise


def prepare(args):
    labels = read_annotations(args.annotations, read_classes(args.classes))
    trainval, test = read_ids(args.trainval), read_ids(args.test)
    archive_size = args.images.stat().st_size
    if args.prefix:
        with tarfile.open(args.images, "r:") as tar:
            available = {Path(member.name).stem for member in image_members(tar, archive_size, True)}
        print(f"Using {len(available)} complete images from the archive prefix.")
        trainval &= available
        test &= available
    if trainval & test:
        raise ValueError("Official split lists contain overlapping IDs.")
    grouped = defaultdict(list)
    for image_id in sorted(trainval & labels.keys()):
        group = tuple(sorted({int(row.split()[0]) for row in labels[image_id]}))
        grouped[group].append(image_id)
    rng = random.Random(42)
    assignment = {image_id: "test" for image_id in sorted(test & labels.keys())}
    for ids in grouped.values():
        rng.shuffle(ids)
        cutoff = max(1, round(len(ids) * .15)) if len(ids) > 1 else 0
        assignment.update({image_id: "val" if i < cutoff else "train" for i, image_id in enumerate(ids)})
    # Keep a bounded sample of non-target annotated photos as hard negatives.
    for split in ("train", "val", "test"):
        positives = [i for i, s in assignment.items() if s == split and labels[i]]
        negatives = [i for i, s in assignment.items() if s == split and not labels[i]]
        rng.shuffle(negatives)
        for image_id in negatives[max(1, len(positives) // 3):]:
            del assignment[image_id]

    args.output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (args.output / kind / split).mkdir(parents=True)
    digests = {}
    with tarfile.open(args.images, "r:*") as tar:
        for member in image_members(tar, archive_size, args.prefix):
            image_id = Path(member.name).stem
            if image_id not in assignment:
                continue
            split = assignment.pop(image_id)
            payload = tar.extractfile(member).read()
            digest = hashlib.sha256(payload).digest()
            previous = digests.get(digest)
            if previous:
                priority = {"test": 0, "val": 1, "train": 2}
                if priority[previous[0]] <= priority[split]:
                    continue
                previous_split, previous_id = previous
                (args.output / "images" / previous_split / (previous_id + ".jpg")).unlink()
                (args.output / "labels" / previous_split / (previous_id + ".txt")).unlink()
            digests[digest] = split, image_id
            (args.output / "images" / split / (image_id + ".jpg")).write_bytes(payload)
            (args.output / "labels" / split / (image_id + ".txt")).write_text(
                "\n".join(labels[image_id]), encoding="utf-8"
            )
    if assignment:
        raise ValueError(f"{len(assignment)} selected images missing from the archive.")
    print(audit(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "images", "trainval", "test", "classes", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--prefix", action="store_true", help="Use only complete records in a partial uncompressed image archive.")
    prepare(parser.parse_args())


