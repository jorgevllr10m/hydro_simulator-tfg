from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulator.config.loader import load_config
from simulator.core.contracts import (
    EnergyInput,
    EnergyOutput,
    HydroInput,
    HydroOutput,
    MeteoInput,
    MeteoOutput,
)
from simulator.core.dataset import create_empty_dataset
from simulator.energy.model import EnergyBalanceModel
from simulator.hydro.model import HydroModel
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


def _write_step_outputs_to_dataset(
    *,
    ds,
    meteo_output: MeteoOutput,
    energy_output: EnergyOutput,
    hydro_output: HydroOutput,
    step: int,
) -> None:
    """Write meteorological, energy and hydrological outputs into the historical dataset."""
    ds["precipitation"][step, :, :] = meteo_output.precipitation
    ds["air_temperature"][step, :, :] = meteo_output.air_temperature

    ds["pet"][step, :, :] = energy_output.pet
    ds["shortwave_radiation"][step, :, :] = energy_output.shortwave_radiation
    ds["net_radiation"][step, :, :] = energy_output.net_radiation

    ds["aet"][step, :, :] = hydro_output.aet
    ds["soil_moisture"][step, :, :] = hydro_output.soil_moisture
    ds["infiltration"][step, :, :] = hydro_output.infiltration
    ds["surface_runoff"][step, :, :] = hydro_output.surface_runoff
    ds["channel_flow"][step, :, :] = hydro_output.channel_flow

    if hydro_output.subsurface_runoff is not None:
        ds["subsurface_runoff"][step, :, :] = hydro_output.subsurface_runoff

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
    """Generate quick-look diagnostic plots for the full simulation run."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    precipitation = np.asarray(ds["precipitation"].values, dtype=float)
    background_precipitation = np.asarray(ds["background_precipitation"].values, dtype=float)
    storm_mask = np.asarray(ds["storm_mask"].values, dtype=float)

    pet = np.asarray(ds["pet"].values, dtype=float)
    aet = np.asarray(ds["aet"].values, dtype=float)
    shortwave_radiation = np.asarray(ds["shortwave_radiation"].values, dtype=float)
    net_radiation = np.asarray(ds["net_radiation"].values, dtype=float)

    soil_moisture = np.asarray(ds["soil_moisture"].values, dtype=float)
    infiltration = np.asarray(ds["infiltration"].values, dtype=float)
    surface_runoff = np.asarray(ds["surface_runoff"].values, dtype=float)
    subsurface_runoff = np.asarray(ds["subsurface_runoff"].values, dtype=float)

    # ----- Accumulated / mean / final fields -----
    accumulated_precipitation = np.nansum(precipitation, axis=0)
    accumulated_background_precipitation = np.nansum(background_precipitation, axis=0)
    accumulated_pet = np.nansum(pet, axis=0)
    accumulated_aet = np.nansum(aet, axis=0)
    accumulated_infiltration = np.nansum(infiltration, axis=0)
    accumulated_surface_runoff = np.nansum(surface_runoff, axis=0)
    accumulated_subsurface_runoff = np.nansum(subsurface_runoff, axis=0)

    mean_shortwave_radiation = np.nanmean(shortwave_radiation, axis=0)
    mean_net_radiation = np.nanmean(net_radiation, axis=0)

    mean_soil_moisture = np.nanmean(soil_moisture, axis=0)
    final_soil_moisture = soil_moisture[-1] if soil_moisture.shape[0] > 0 else np.zeros(ds["soil_moisture"].shape[1:], dtype=float)

    # ----- Core field plots -----
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

    _save_field_plot(
        accumulated_pet,
        title="Accumulated PET",
        output_path=plots_dir / "accumulated_pet.png",
        colorbar_label="mm",
    )

    _save_field_plot(
        accumulated_aet,
        title="Accumulated AET",
        output_path=plots_dir / "accumulated_aet.png",
        colorbar_label="mm",
    )

    _save_field_plot(
        accumulated_infiltration,
        title="Accumulated infiltration",
        output_path=plots_dir / "accumulated_infiltration.png",
        colorbar_label="mm",
    )

    _save_field_plot(
        accumulated_surface_runoff,
        title="Accumulated surface runoff",
        output_path=plots_dir / "accumulated_surface_runoff.png",
        colorbar_label="mm",
    )

    _save_field_plot(
        accumulated_subsurface_runoff,
        title="Accumulated subsurface runoff",
        output_path=plots_dir / "accumulated_subsurface_runoff.png",
        colorbar_label="mm",
    )

    _save_field_plot(
        mean_soil_moisture,
        title="Mean soil moisture",
        output_path=plots_dir / "mean_soil_moisture.png",
        colorbar_label="mm",
    )

    _save_field_plot(
        final_soil_moisture,
        title="Final soil moisture",
        output_path=plots_dir / "final_soil_moisture.png",
        colorbar_label="mm",
    )

    _save_field_plot(
        mean_shortwave_radiation,
        title="Mean shortwave radiation",
        output_path=plots_dir / "mean_shortwave_radiation.png",
        colorbar_label="W/m2",
    )

    _save_field_plot(
        mean_net_radiation,
        title="Mean net radiation",
        output_path=plots_dir / "mean_net_radiation.png",
        colorbar_label="MJ/m2/dt",
    )

    # ----- Peak step plots -----
    if precipitation.shape[0] > 0 and np.isfinite(precipitation).any():
        peak_precip_step = int(np.nanargmax(np.nanmax(precipitation, axis=(1, 2))))

        _save_field_plot(
            precipitation[peak_precip_step],
            title=f"Precipitation at peak step {peak_precip_step}",
            output_path=plots_dir / "step_peak_precipitation.png",
            colorbar_label="mm/dt",
        )

        _save_field_plot(
            storm_mask[peak_precip_step],
            title=f"Storm mask at peak precipitation step {peak_precip_step}",
            output_path=plots_dir / "step_peak_storm_mask.png",
            colorbar_label="1",
        )

    if surface_runoff.shape[0] > 0 and np.isfinite(surface_runoff).any():
        peak_runoff_step = int(np.nanargmax(np.nanmax(surface_runoff, axis=(1, 2))))

        _save_field_plot(
            surface_runoff[peak_runoff_step],
            title=f"Surface runoff at peak step {peak_runoff_step}",
            output_path=plots_dir / "step_peak_surface_runoff.png",
            colorbar_label="mm/dt",
        )

        _save_field_plot(
            infiltration[peak_runoff_step],
            title=f"Infiltration at peak runoff step {peak_runoff_step}",
            output_path=plots_dir / "step_peak_infiltration.png",
            colorbar_label="mm/dt",
        )

        _save_field_plot(
            soil_moisture[peak_runoff_step],
            title=f"Soil moisture at peak runoff step {peak_runoff_step}",
            output_path=plots_dir / "step_peak_runoff_soil_moisture.png",
            colorbar_label="mm",
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

    # ----- Time series from summary rows -----
    if rows:
        step_index = np.asarray([int(row["step"]) for row in rows], dtype=int)

        def _plot_row_series(
            key: str,
            *,
            title: str,
            ylabel: str,
            filename: str,
        ) -> None:
            if key not in rows[0]:
                return

            values = np.asarray([float(row[key]) for row in rows], dtype=float)
            _save_line_plot(
                step_index,
                values,
                title=title,
                xlabel="step",
                ylabel=ylabel,
                output_path=plots_dir / filename,
            )

        _plot_row_series(
            "precipitation_max_mm_dt",
            title="Maximum precipitation per step",
            ylabel="mm/dt",
            filename="precipitation_max_timeseries.png",
        )

        _plot_row_series(
            "background_fraction_of_total",
            title="Background fraction of total precipitation",
            ylabel="fraction",
            filename="background_fraction_timeseries.png",
        )

        _plot_row_series(
            "band_reorganization_applied",
            title="Band reorganization applied",
            ylabel="0/1",
            filename="band_reorganization_timeseries.png",
        )

        _plot_row_series(
            "pet_mean_mm_dt",
            title="Mean PET per step",
            ylabel="mm/dt",
            filename="pet_mean_timeseries.png",
        )

        _plot_row_series(
            "aet_mean_mm_dt",
            title="Mean AET per step",
            ylabel="mm/dt",
            filename="aet_mean_timeseries.png",
        )

        _plot_row_series(
            "soil_moisture_mean_mm",
            title="Mean soil moisture per step",
            ylabel="mm",
            filename="soil_moisture_mean_timeseries.png",
        )

        _plot_row_series(
            "infiltration_sum_mm_dt",
            title="Total infiltration per step",
            ylabel="mm/dt",
            filename="infiltration_sum_timeseries.png",
        )

        _plot_row_series(
            "surface_runoff_sum_mm_dt",
            title="Total surface runoff per step",
            ylabel="mm/dt",
            filename="surface_runoff_sum_timeseries.png",
        )

        _plot_row_series(
            "subsurface_runoff_sum_mm_dt",
            title="Total subsurface runoff per step",
            ylabel="mm/dt",
            filename="subsurface_runoff_sum_timeseries.png",
        )

        _plot_row_series(
            "shortwave_radiation_mean_w_m2",
            title="Mean shortwave radiation per step",
            ylabel="W/m2",
            filename="shortwave_radiation_mean_timeseries.png",
        )

        _plot_row_series(
            "net_radiation_mean_mj_m2_dt",
            title="Mean net radiation per step",
            ylabel="MJ/m2/dt",
            filename="net_radiation_mean_timeseries.png",
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
    hydro_config = loaded.build_hydro_config()
    hydro_model = HydroModel(
        hydro_config,
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

        hydro_input = HydroInput(
            domain=domain,
            step=step,
            timestamp=timestamp,
            precipitation=meteo_output.precipitation,
            pet=energy_output.pet,
            soil_moisture_prev=hydro_model.latest_state.soil_moisture_mm,
        )
        hydro_output = hydro_model.step(hydro_input)

        _write_step_outputs_to_dataset(
            ds=ds,
            meteo_output=meteo_output,
            energy_output=energy_output,
            hydro_output=hydro_output,
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
                "shortwave_radiation_mean_w_m2": float(energy_output.shortwave_radiation.mean()),
                "net_radiation_mean_mj_m2_dt": float(energy_output.net_radiation.mean()),
                "aet_mean_mm_dt": float(hydro_output.aet.mean()),
                "soil_moisture_mean_mm": float(hydro_output.soil_moisture.mean()),
                "infiltration_sum_mm_dt": float(hydro_output.infiltration.sum()),
                "surface_runoff_sum_mm_dt": float(hydro_output.surface_runoff.sum()),
                "subsurface_runoff_sum_mm_dt": float(
                    0.0 if hydro_output.subsurface_runoff is None else hydro_output.subsurface_runoff.sum()
                ),
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
            f"aet_mean={hydro_output.aet.mean():.3f} mm/dt | "
            f"soil_mean={hydro_output.soil_moisture.mean():.3f} mm | "
            f"runoff_mean={hydro_output.surface_runoff.mean():.3f} mm/dt"
        )

    dataset_path = _save_dataset(ds, output_dir=run_output_dir)

    summary_csv_path = run_output_dir / "simulation_summary.csv"
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
