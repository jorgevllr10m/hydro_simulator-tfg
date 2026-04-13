from pathlib import Path

from simulator.config.loader import load_config


def test_load_config_returns_dictionary() -> None:
    config = load_config("configs/config.yaml")

    assert isinstance(config, dict)


def test_load_config_contains_required_top_level_keys() -> None:
    config = load_config("configs/config.yaml")

    assert "run" in config
    assert "simulation" in config
    assert "domain" in config
    assert "scenario" in config


def test_load_config_reads_expected_values() -> None:
    config = load_config(Path("configs/config.yaml"))

    assert config["run"]["name"] == "base_run"
    assert config["domain"]["preset"] == "small"
    assert config["scenario"]["name"] == "baseline"
