"""Prepare a balanced index from local paintings; never download data."""

import csv
import hashlib
import json
import random
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

import config


def image_digest(path):
    """Fully decode an image and hash its size and RGB pixels."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            digest = hashlib.sha256(str(image.size).encode())
            digest.update(image.tobytes())
            return digest.hexdigest()


def collect_images(raw_directory):
    """Find readable, globally unique images and count rejected files."""
    raw_directory = Path(raw_directory)
    missing = [name for name in config.STYLE_FOLDERS.values()
               if not (raw_directory / name).is_dir()]
    if missing:
        raise ValueError(f"Missing style folders: {', '.join(missing)}")
    seen = set()
    images = {}
    audit = {}
    for style, folder in config.STYLE_FOLDERS.items():
        images[style] = []
        audit[style] = {"candidates": 0, "corrupt": 0, "duplicates": 0}
        for path in sorted((raw_directory / folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue
            audit[style]["candidates"] += 1
            try:
                digest = image_digest(path)
            except (OSError, ValueError, UnidentifiedImageError,
                    Image.DecompressionBombError, Image.DecompressionBombWarning):
                audit[style]["corrupt"] += 1
                continue
            # Deduplicate before selection, including identical images with different encodings.
            if digest in seen:
                audit[style]["duplicates"] += 1
                continue
            seen.add(digest)
            images[style].append({"path": path.relative_to(raw_directory).as_posix(),
                                  "style": style, "sha256": digest})
        audit[style]["valid_unique"] = len(images[style])
    return images, audit


def split_images(images, count=config.IMAGES_PER_CLASS, seed=config.RANDOM_SEED):
    """Select and shuffle each class deterministically into 70/15/15 splits."""
    shortages = {style: len(rows) for style, rows in images.items() if len(rows) < count}
    if shortages:
        raise ValueError(f"Need {count} valid unique images per class; available: {shortages}")
    if count < 7:
        raise ValueError("At least 7 images per class are required for three nonempty splits")
    rng = random.Random(seed)
    result = []
    train_end = int(count * config.TRAIN_FRACTION)
    validation_end = train_end + int(count * config.VALIDATION_FRACTION)
    for style in config.STYLE_FOLDERS:
        selected = rng.sample(images[style], count)
        for index, row in enumerate(selected):
            split = "train" if index < train_end else (
                "validation" if index < validation_end else "test")
            result.append({**row, "split": split})
    return result


def verify_records(records, raw_directory):
    """Recheck indexed images and reject duplicate content or invalid split labels."""
    seen = set()
    for row in records:
        if row["style"] not in config.STYLE_FOLDERS:
            raise ValueError(f"Unknown style: {row['style']}")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {row['split']}")
        digest = image_digest(Path(raw_directory) / row["path"])
        if digest != row["sha256"]:
            raise ValueError(f"Image changed: {row['path']}")
        if digest in seen:
            raise ValueError(f"Duplicate image: {row['path']}")
        seen.add(digest)


def prepare_dataset(raw_directory=config.RAW_DATA_DIR,
                    output_directory=config.PREPARED_DATA_DIR,
                    count=config.IMAGES_PER_CLASS, seed=config.RANDOM_SEED):
    """Validate local images and save the split index and an audit report."""
    images, audit = collect_images(raw_directory)
    for style, counts in audit.items():
        print(f"{style}: {counts}")
    records = split_images(images, count, seed)
    verify_records(records, raw_directory)
    distribution = {style: {split: sum(row["style"] == style and row["split"] == split
                                      for row in records)
                            for split in ("train", "validation", "test")}
                    for style in config.STYLE_FOLDERS}
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "index.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "style", "sha256", "split"])
        writer.writeheader()
        writer.writerows(records)
    report = {"seed": seed, "images_per_class": count, "audit": audit,
              "distribution": distribution,
              "selected_image_bytes": sum((Path(raw_directory) / row["path"]).stat().st_size
                                          for row in records)}
    (output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nStyle | train | validation | test")
    for style, counts in distribution.items():
        print(f"{style} | {counts['train']} | {counts['validation']} | {counts['test']}")
    print(f"\nIndex and report saved to {output_directory}")
    return records, report


def get_transform(training=False):
    """Build ResNet18 preprocessing, with random flipping only for training."""
    from torchvision import transforms

    steps = [transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))]
    if training:
        steps.append(transforms.RandomHorizontalFlip(config.HORIZONTAL_FLIP_PROBABILITY))
    steps.extend([transforms.ToTensor(),
                  transforms.Normalize(config.NORMALIZATION_MEAN, config.NORMALIZATION_STD)])
    return transforms.Compose(steps)


def load_image(path, training=False):
    """Load an RGB image and return its normalized 3 x 224 x 224 tensor."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return get_transform(training)(image)


if __name__ == "__main__":
    prepare_dataset()


class PaintingDataset:
    """Minimal map-style dataset compatible with PyTorch DataLoader."""

    def __init__(self, records, split):
        """Keep only the requested existing split and reuse configured label order."""
        self.records = [row for row in records if row["split"] == split]
        self.training = split == "train"
        self.classes = list(config.STYLE_FOLDERS)
        if not self.records:
            raise ValueError(f"Empty dataset split: {split}")

    def __len__(self):
        """Return the number of indexed paintings."""
        return len(self.records)

    def __getitem__(self, index):
        """Load and transform one painting and return its integer class label."""
        row = self.records[index]
        return (load_image(config.RAW_DATA_DIR / row["path"], self.training),
                self.classes.index(row["style"]))
