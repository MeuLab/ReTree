import argparse
from retree.runners.evaluate import run_evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReTree with hierarchical retrieval")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint path")
    parser.add_argument("--memory", required=True, help="Tree memory path")
    args = parser.parse_args()
    run_evaluate(args.config, args.checkpoint, args.memory)


if __name__ == "__main__":
    main()
