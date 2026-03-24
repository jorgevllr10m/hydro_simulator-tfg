import argparse
from pathlib import Path

from simulator.config.loader import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydro-sim",
        description="Run a minimal synthetic basin simulation from a YAML configuration file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()  # Read arguments

    config = load_config(args.config)

    print("Synthetic Basin Simulator")
    print(f"Using configuration path: {args.config}")
    print("Configuration loaded successfully.")
    print(f"Run name: {config['run']['name']}")
    print(f"Domain preset: {config['domain']['preset']}")
    print(f"Scenario name: {config['scenario']['name']}")
    print("Minimal run completed successfully.")


if __name__ == "__main__":
    main()
