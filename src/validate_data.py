from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow;
      5. no split or class is empty.

    The report has no timestamp because it is a DVC output.
    """
    # check each split one by one
    splits = {}
    total = 0

    for split, expected_count in EXPECTED_SPLITS.items():
        split_dir = raw_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        classes = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if classes != EXPECTED_CLASSES:
            raise RuntimeError(
                f"{split}: expected classes {sorted(EXPECTED_CLASSES)}, found {sorted(classes)}"
            )

        class_counts = {}
        for class_name in sorted(classes):
            images = sorted(
                p for p in (split_dir / class_name).rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise RuntimeError(f"{split}/{class_name} has no image")

            for path in images:
                # load() reads the pixels, open() only reads the header
                with Image.open(path) as img:
                    img.load()

            class_counts[class_name] = len(images)

        count = sum(class_counts.values())
        if count != expected_count:
            raise RuntimeError(f"{split}: expected {expected_count} images, found {count}")

        splits[split] = {"images": count, "class_counts": class_counts}
        total += count

    return {
        "status": "PASS",
        "classes": sorted(EXPECTED_CLASSES),
        "splits": splits,
        "total_images": total,
    }


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
