import argparse
from retree.runners.train import run_train


def main():
    parser = argparse.ArgumentParser(description="Train ReTree for inductive MKGC")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()
    run_train(args.config)


if __name__ == "__main__":
    main()
