import argparse
from retree.runners.build_memory import run_build_memory


def main():
    parser = argparse.ArgumentParser(description="Build hierarchical tree memory for ReTree")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint path")
    args = parser.parse_args()
    run_build_memory(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
