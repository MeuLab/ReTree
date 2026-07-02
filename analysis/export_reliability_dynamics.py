import argparse
import csv
import torch


def main():
    parser = argparse.ArgumentParser(description="Export clean-noisy reliability gaps from a saved log tensor")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = torch.load(args.input, map_location="cpu")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "delta_text", "delta_vis", "delta_pair"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    main()
