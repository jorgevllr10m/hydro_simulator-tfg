from simulator.cli.run import main

def test_config_file_exists() -> None:
    with open("configs/config.yaml", encoding="utf-8") as file:
        assert file.read() is not None


def test_main_runs_and_prints_expected_output(capsys) -> None:
    main()

    captured = capsys.readouterr() # capsys allows you to capture what the program prints to the console while the test is running.

    assert "Synthetic Basin Simulator" in captured.out
    assert "Using configuration path: configs\\config.yaml" in captured.out or "Using configuration path: configs/config.yaml" in captured.out
    assert "Configuration file found." in captured.out
    assert "Minimal run completed successfully." in captured.out