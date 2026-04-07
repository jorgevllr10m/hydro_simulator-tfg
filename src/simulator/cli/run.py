from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulator.config.loader import load_config
from simulator.core.contracts import EnergyInput, EnergyOutput, MeteoInput, MeteoOutput
from simulator.core.dataset import create_empty_dataset
from simulator.energy.model import EnergyBalanceModel
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


def _write_meteo_energy_output_to_dataset(
    *,
    ds,
    meteo_output: MeteoOutput,
    energy_output: EnergyOutput,
    step: int,
) -> None:
    """Write meteorological and energy-balance outputs into the historical dataset."""
    ds["precipitation"][step, :, :] = meteo_output.precipitation
    ds["air_temperature"][step, :, :] = meteo_output.air_temperature
    ds["pet"][step, :, :] = energy_output.pet
    ds["aet"][step, :, :] = energy_output.aet
    ds["shortwave_radiation"][step, :, :] = energy_output.shortwave_radiation
    ds["net_radiation"][step, :, :] = energy_output.net_radiation
    ds["antecedent_storage"][step, :, :] = energy_output.antecedent_storage
    ds["antecedent_relative"][step, :, :] = energy_output.antecedent_relative
    ds["antecedent_overflow"][step, :, :] = energy_output.antecedent_overflow

    if meteo_output.background_precipitation is not None:
        ds["background_precipitation"][step, :, :] = meteo_output.background_precipitation

    if meteo_output.storm_mask is not None:
        ds["storm_mask"][step, :, :] = meteo_output.storm_mask


def _write_summary_csv(
    *,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write step diagnostics into a CSV file.

    CSV is formatted for spreadsheet tools using:
    - ';' as field separator
    - ',' as decimal separator for float values
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    def _format_csv_value(value: object) -> object:
        if isinstance(value, float):
            return f"{value:.6f}".replace(".", ",")
        return value

    formatted_rows = [{key: _format_csv_value(value) for key, value in row.items()} for row in rows]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(formatted_rows)


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


def _save_field_plot(
    field: np.ndarray,
    *,
    title: str,
    output_path: Path,
    colorbar_label: str,
) -> None:
    """Save one 2D field as a PNG image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    image = plt.imshow(field, origin="lower")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(image, label=colorbar_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save one simple line plot as a PNG image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_quicklook_plots(
    *,
    ds,
    rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Generate quick-look diagnostic plots for the meteorology run."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    precipitation = np.asarray(ds["precipitation"].values, dtype=float)
    background_precipitation = np.asarray(ds["background_precipitation"].values, dtype=float)
    storm_mask = np.asarray(ds["storm_mask"].values, dtype=float)

    accumulated_precipitation = np.nansum(precipitation, axis=0)
    accumulated_background_precipitation = np.nansum(background_precipitation, axis=0)

    _save_field_plot(
        accumulated_precipitation,
        title="Accumulated precipitation",
        output_path=plots_dir / "accumulated_precipitation.png",
        colorbar_label="mm",
    )

    _save_field_plot(
        accumulated_background_precipitation,
        title="Accumulated background precipitation",
        output_path=plots_dir / "accumulated_background_precipitation.png",
        colorbar_label="mm",
    )

    if precipitation.shape[0] > 0:
        peak_step = int(np.nanargmax(np.nanmax(precipitation, axis=(1, 2))))

        _save_field_plot(
            precipitation[peak_step],
            title=f"Precipitation at peak step {peak_step}",
            output_path=plots_dir / "step_peak_precipitation.png",
            colorbar_label="mm/dt",
        )

        _save_field_plot(
            storm_mask[peak_step],
            title=f"Storm mask at peak step {peak_step}",
            output_path=plots_dir / "step_peak_storm_mask.png",
            colorbar_label="1",
        )

    if rows and "band_reorganization_applied" in rows[0]:
        band_steps = [int(row["step"]) for row in rows if int(row["band_reorganization_applied"]) == 1]

        if band_steps:
            first_band_step = band_steps[0]

            _save_field_plot(
                precipitation[first_band_step],
                title=f"Precipitation at first band step {first_band_step}",
                output_path=plots_dir / "step_first_band_precipitation.png",
                colorbar_label="mm/dt",
            )

            _save_field_plot(
                storm_mask[first_band_step],
                title=f"Storm mask at first band step {first_band_step}",
                output_path=plots_dir / "step_first_band_storm_mask.png",
                colorbar_label="1",
            )

    if rows:
        step_index = np.asarray([int(row["step"]) for row in rows], dtype=int)

        precipitation_max = np.asarray(
            [float(row["precipitation_max_mm_dt"]) for row in rows],
            dtype=float,
        )
        _save_line_plot(
            step_index,
            precipitation_max,
            title="Maximum precipitation per step",
            xlabel="step",
            ylabel="mm/dt",
            output_path=plots_dir / "precipitation_max_timeseries.png",
        )

        if "background_fraction_of_total" in rows[0]:
            background_fraction = np.asarray(
                [float(row["background_fraction_of_total"]) for row in rows],
                dtype=float,
            )
            _save_line_plot(
                step_index,
                background_fraction,
                title="Background fraction of total precipitation",
                xlabel="step",
                ylabel="fraction",
                output_path=plots_dir / "background_fraction_timeseries.png",
            )

        if "band_reorganization_applied" in rows[0]:
            band_flag = np.asarray(
                [float(row["band_reorganization_applied"]) for row in rows],
                dtype=float,
            )
            _save_line_plot(
                step_index,
                band_flag,
                title="Band reorganization applied",
                xlabel="step",
                ylabel="0/1",
                output_path=plots_dir / "band_reorganization_timeseries.png",
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    loaded = load_config(args.config)

    domain = loaded.build_simulation_domain()
    meteo_config = loaded.build_storm_precipitation_config()
    meteo_model = StormPrecipitationModel(meteo_config)
    energy_config = loaded.build_energy_balance_config()
    energy_model = EnergyBalanceModel(
        energy_config,
        shape=domain.shape,
    )

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

        energy_input = EnergyInput(
            domain=domain,
            step=step,
            timestamp=timestamp,
            precipitation=meteo_output.precipitation,
            air_temperature=meteo_output.air_temperature,
        )
        energy_output = energy_model.step(energy_input)

        _write_meteo_energy_output_to_dataset(
            ds=ds,
            meteo_output=meteo_output,
            energy_output=energy_output,
            step=step,
        )

        diagnostics = meteo_model.latest_diagnostics
        assert diagnostics is not None

        total_precipitation_sum = float(meteo_output.precipitation.sum())
        total_precipitation_max = float(meteo_output.precipitation.max())

        if meteo_output.background_precipitation is None:
            background_precipitation_sum = 0.0
            background_precipitation_max = 0.0
        else:
            background_precipitation_sum = float(meteo_output.background_precipitation.sum())
            background_precipitation_max = float(meteo_output.background_precipitation.max())

        background_fraction_of_total = (
            0.0 if total_precipitation_sum <= 0.0 else background_precipitation_sum / total_precipitation_sum
        )

        summary_rows.append(
            {
                "step": step,
                "timestamp": timestamp.strftime("%d/%m/%y %H:%M:%S"),
                "regime": diagnostics.latent_state.regime.value,
                "new_storms": diagnostics.n_new_storms,
                "tracked_storms": diagnostics.n_active_storms,
                "precipitation_sum_mm_dt": total_precipitation_sum,
                "precipitation_max_mm_dt": total_precipitation_max,
                "background_precipitation_sum_mm_dt": background_precipitation_sum,
                "background_precipitation_max_mm_dt": background_precipitation_max,
                "background_fraction_of_total": background_fraction_of_total,
                "background_activity_factor": diagnostics.background_activity_factor,
                "precipitation_spell_index": diagnostics.precipitation_spell_index,
                "band_reorganization_applied": int(diagnostics.band_reorganization_applied),
                "band_births_count": diagnostics.band_births_count,
                "band_probability": diagnostics.band_probability,
                "storm_mask_active_cells": int(0 if meteo_output.storm_mask is None else (meteo_output.storm_mask > 0.0).sum()),
                "air_temperature_mean_c": float(meteo_output.air_temperature.mean()),
                "pet_mean_mm_dt": float(energy_output.pet.mean()),
                "aet_mean_mm_dt": float(energy_output.aet.mean()),
                "shortwave_radiation_mean_w_m2": float(energy_output.shortwave_radiation.mean()),
                "net_radiation_mean_mj_m2_dt": float(energy_output.net_radiation.mean()),
                "antecedent_storage_mean_mm": float(energy_output.antecedent_storage.mean()),
                "antecedent_relative_mean": float(energy_output.antecedent_relative.mean()),
                "antecedent_overflow_sum_mm_dt": float(energy_output.antecedent_overflow.sum()),
            }
        )

        print(
            f"[step={step:03d}] "
            f"regime={diagnostics.latent_state.regime.value} | "
            f"new={diagnostics.n_new_storms} | "
            f"tracked={diagnostics.n_active_storms} | "
            f"band={int(diagnostics.band_reorganization_applied)} | "
            f"band_births={diagnostics.band_births_count} | "
            f"precip_max={meteo_output.precipitation.max():.3f} mm/dt | "
            f"pet_mean={energy_output.pet.mean():.3f} mm/dt | "
            f"aet_mean={energy_output.aet.mean():.3f} mm/dt | "
            f"ant_rel_mean={energy_output.antecedent_relative.mean():.3f}"
        )

    dataset_path = _save_dataset(ds, output_dir=run_output_dir)

    summary_csv_path = run_output_dir / "meteo_energy_summary.csv"
    _write_summary_csv(
        rows=summary_rows,
        output_path=summary_csv_path,
    )

    _save_quicklook_plots(
        ds=ds,
        rows=summary_rows,
        output_dir=run_output_dir,
    )

    print("Run completed successfully.")
    print(f"Dataset written to: {dataset_path}")
    print(f"Plots written to: {run_output_dir / 'plots'}")
    print(f"Step summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
