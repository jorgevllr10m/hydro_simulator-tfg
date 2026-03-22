from pathlib import Path


def main() -> None:
    config_path = Path("configs/config.yaml")

    print("Synthetic Basin Simulator")
    print(f"Using configuration path: {config_path}")

    if config_path.exists():
        print("Configuration file found.")
        print("Minimal run completed successfully.")
    else:
        print("Configuration file not found.")
        print("Minimal run completed without configuration.")

if __name__ == "__main__":
    main()