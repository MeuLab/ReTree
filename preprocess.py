import argparse
from pathlib import Path

REQUIRED = [
    "entities.txt", "relations.txt", "train.tsv", "valid.tsv", "test.tsv",
    "entity_text.json", "entity_image.npy", "image_index.json",
]


def main():
    parser = argparse.ArgumentParser(description="Check dataset files for ReTree")
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    root = Path(args.data_dir)
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")
    print(f"Dataset folder is ready: {root}")


if __name__ == "__main__":
    main()
