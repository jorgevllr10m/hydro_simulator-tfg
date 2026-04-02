from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

from simulator.config.loader import load_config
from simulator.core.contracts import MeteoInput, MeteoOutput
from simulator.core.dataset import create_empty_dataset
from simulator.meteo.precipitation_model import StormPrecipitationModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydro-sim",
        description="Run the current simulator pipeline from a YAML configuration file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the master YAML configuration file.",
    )
    return parser


def _write_meteo_output_to_dataset(
    *,
    ds,
    meteo_output: MeteoOutput,
    step: int,
) -> None:
    """Write only meteorological outputs into the historical dataset.

    Current phase-3 runner executes only the meteorology module, so we write
    only the variables that are truly produced by that module and leave the
    rest of the dataset untouched (still NaN from initialization).
    """
    ds["precipitation"][step, :, :] = meteo_output.precipitation
    ds["air_temperature"][step, :, :] = meteo_output.air_temperature

    if meteo_output.background_precipitation is not None:
        ds["background_precipitation"][step, :, :] = meteo_output.background_precipitation

    if meteo_output.storm_mask is not None:
        ds["storm_mask"][step, :, :] = meteo_output.storm_mask


def _write_summary_csv(
    *,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write step diagnostics into a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _save_dataset(ds, *, output_dir: Path) -> Path:
    """Persist the dataset to disk.

    Preferred format is NetCDF. If the required backend is not available,
    the runner falls back to a pickle file so the run still completes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    netcdf_path = output_dir / "simulation_dataset.nc"
    try:
        ds.to_netcdf(netcdf_path)
        return netcdf_path
    except Exception:
        pickle_path = output_dir / "simulation_dataset.pkl"
        with pickle_path.open("wb") as file:
            pickle.dump(ds, file)
        return pickle_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    loaded = load_config(args.config)

    domain = loaded.build_simulation_domain()
    meteo_config = loaded.build_storm_precipitation_config()
    meteo_model = StormPrecipitationModel(meteo_config)

    run_output_dir = loaded.run_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    ds = create_empty_dataset(domain)
    ds.attrs["run_name"] = loaded.run_name
    ds.attrs["domain_preset"] = loaded.domain_preset_name
    ds.attrs["scenario_name"] = loaded.scenario_name

    summary_rows: list[dict[str, object]] = []

    print("Synthetic Basin Simulator")
    print(f"Configuration path: {loaded.config_path}")
    print(f"Run name: {loaded.run_name}")
    print(f"Domain preset: {loaded.domain_preset_name}")
    print(f"Scenario name: {loaded.scenario_name}")
    print(f"Output directory: {run_output_dir}")
    print(f"Spatial shape: {domain.shape}")
    print(f"Time steps: {domain.n_steps}")
    print(f"dt_seconds: {domain.time.dt_seconds}")

    for step in range(domain.n_steps):
        timestamp = domain.time.timestamps[step]

        meteo_input = MeteoInput(
            domain=domain,
            step=step,
            timestamp=timestamp,
            previous_state=None,
        )

        meteo_output = meteo_model.step(meteo_input)
        _write_meteo_output_to_dataset(
            ds=ds,
            meteo_output=meteo_output,
            step=step,
        )
        # TODO(phase4+): replace the meteorology-only dataset writing path with the
        # full module pipeline once energy/hydrology outputs are available.

        diagnostics = meteo_model.latest_diagnostics
        assert diagnostics is not None

        summary_rows.append(
            {
                "step": step,
                "timestamp": timestamp.isoformat(),
                "regime": diagnostics.latent_state.regime.value,
                "new_storms": diagnostics.n_new_storms,
                "tracked_storms": diagnostics.n_active_storms,
                "precipitation_sum_mm_dt": float(meteo_output.precipitation.sum()),
                "precipitation_max_mm_dt": float(meteo_output.precipitation.max()),
                "storm_mask_active_cells": int(0 if meteo_output.storm_mask is None else (meteo_output.storm_mask > 0.0).sum()),
                "air_temperature_mean_c": float(meteo_output.air_temperature.mean()),
            }
        )

        print(
            f"[step={step:03d}] "
            f"regime={diagnostics.latent_state.regime.value} | "
            f"new={diagnostics.n_new_storms} | "
            f"tracked={diagnostics.n_active_storms} | "
            f"precip_max={meteo_output.precipitation.max():.3f} mm/dt"
        )

    dataset_path = _save_dataset(ds, output_dir=run_output_dir)

    summary_csv_path = run_output_dir / "meteo_summary.csv"
    _write_summary_csv(
        rows=summary_rows,
        output_path=summary_csv_path,
    )

    print("Run completed successfully.")
    print(f"Dataset written to: {dataset_path}")
    print(f"Step summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
