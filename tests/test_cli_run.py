from pathlib import Path

from simulator.cli.run import main


def test_config_file_exists() -> None:
    config_path = Path("configs/config.yaml")

    assert config_path.exists()
    assert config_path.is_file()


def test_main_runs_and_prints_expected_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["hydro-sim"])

    main()

    captured = capsys.readouterr()

    assert "Synthetic Basin Simulator" in captured.out
    assert (
        "Using configuration path: configs\\config.yaml" in captured.out
        or "Using configuration path: configs/config.yaml" in captured.out
    )
    assert "Configuration loaded successfully." in captured.out
    assert "Run name: base_run" in captured.out
    assert "Domain preset: small" in captured.out
    assert "Scenario name: baseline" in captured.out
    assert "Minimal run completed successfully." in captured.out
