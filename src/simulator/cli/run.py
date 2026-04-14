from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

from simulator.config.loader import load_config
from simulator.core.contracts import (
    EnergyInput,
    HydroInput,
    MeteoInput,
    MeteoOutput,
    ObservationInput,
    ObservationOutput,
    RegulatedRoutingInput,
    RegulatedRoutingOutput,
)
from simulator.core.dataset import (
    create_empty_observation_dataset,
    create_empty_truth_dataset,
    write_observation_to_dataset,
    write_state_to_dataset,
)
from simulator.core.engine import merge_module_outputs
from simulator.energy.model import EnergyBalanceModel
from simulator.hydro.model import HydroModel
from simulator.meteo.precipitation_model import StormPrecipitationModel
from simulator.obs.model import (
    DISCHARGE_SENSOR_TYPE,
    PRECIPITATION_SENSOR_TYPE,
    RESERVOIR_STORAGE_SENSOR_TYPE,
    ObservationModel,
    ObservationQualityFlag,
)
from simulator.routing.model import RegulatedRoutingModel
from simulator.routing.network import build_simplified_drainage_network


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


def _quality_flag_label(flag_value: int) -> str:
    """Return a human-readable label for one observation quality flag."""
    try:
        return ObservationQualityFlag(int(flag_value)).name.lower()
    except ValueError:
        return "unknown"


def _extract_sensor_truth_value(
    *,
    domain,
    sensor,
    meteo_output: MeteoOutput,
    routing_output: RegulatedRoutingOutput,
) -> float:
    """Return the truth value associated with one sensor at the current step."""
    sensor_type = sensor.sensor_type

    if sensor_type == PRECIPITATION_SENSOR_TYPE:
        return float(meteo_output.precipitation[sensor.cell_y, sensor.cell_x])

    if sensor_type == DISCHARGE_SENSOR_TYPE:
        return float(routing_output.channel_flow_m3s[sensor.cell_y, sensor.cell_x])

    if sensor_type == RESERVOIR_STORAGE_SENSOR_TYPE:
        for reservoir_id, reservoir in enumerate(domain.reservoirs):
            if reservoir.cell_y == sensor.cell_y and reservoir.cell_x == sensor.cell_x:
                return float(routing_output.reservoir_storage_m3[reservoir_id])

        raise ValueError(
            f"Sensor '{sensor.name}' of type '{sensor_type}' does not match any reservoir cell at {(sensor.cell_y, sensor.cell_x)}."
        )

    raise ValueError(f"Unsupported sensor_type {sensor_type!r} for sensor '{sensor.name}'.")


def _extract_sensor_observed_value(
    *,
    sensor_type: str,
    sensor_index: int,
    observation_output: ObservationOutput,
) -> float:
    """Return the observed value stored in the matching observation array."""
    if sensor_type == PRECIPITATION_SENSOR_TYPE:
        assert observation_output.obs_precipitation is not None
        return float(observation_output.obs_precipitation[sensor_index])

    if sensor_type == DISCHARGE_SENSOR_TYPE:
        assert observation_output.obs_discharge is not None
        return float(observation_output.obs_discharge[sensor_index])

    if sensor_type == RESERVOIR_STORAGE_SENSOR_TYPE:
        assert observation_output.obs_storage is not None
        return float(observation_output.obs_storage[sensor_index])

    raise ValueError(f"Unsupported sensor_type {sensor_type!r}.")


def _build_observation_rows(
    *,
    domain,
    step: int,
    timestamp,
    meteo_output: MeteoOutput,
    routing_output: RegulatedRoutingOutput,
    observation_output: ObservationOutput,
) -> list[dict[str, object]]:
    """Flatten one observation step into per-sensor CSV rows."""
    rows: list[dict[str, object]] = []

    assert observation_output.obs_mask is not None
    assert observation_output.obs_quality_flag is not None

    for sensor_index, sensor in enumerate(domain.sensors):
        truth_value = _extract_sensor_truth_value(
            domain=domain,
            sensor=sensor,
            meteo_output=meteo_output,
            routing_output=routing_output,
        )
        observed_value = _extract_sensor_observed_value(
            sensor_type=sensor.sensor_type,
            sensor_index=sensor_index,
            observation_output=observation_output,
        )

        quality_flag = int(observation_output.obs_quality_flag[sensor_index])

        rows.append(
            {
                "step": step,
                "timestamp": timestamp.strftime("%d/%m/%y %H:%M:%S"),
                "sensor_index": sensor_index,
                "sensor_name": sensor.name,
                "sensor_type": sensor.sensor_type,
                "cell_y": sensor.cell_y,
                "cell_x": sensor.cell_x,
                "truth_value": truth_value,
                "observed_value": observed_value,
                "observation_available": int(bool(observation_output.obs_mask[sensor_index])),
                "quality_flag": quality_flag,
                "quality_label": _quality_flag_label(quality_flag),
            }
        )

    return rows


def _save_dataset(
    ds,
    *,
    output_dir: Path,
    file_stem: str,
) -> Path:
    """Persist the dataset to disk.

    Preferred format is NetCDF. If the required backend is not available,
    the runner falls back to a pickle file so the run still completes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    netcdf_path = output_dir / f"{file_stem}.nc"
    try:
        ds.to_netcdf(netcdf_path)
        return netcdf_path
    except Exception:
        pickle_path = output_dir / f"{file_stem}.pkl"
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
    routing_network = build_simplified_drainage_network(domain)
    routing_config = loaded.build_regulated_routing_config()
    routing_model = RegulatedRoutingModel(
        routing_config,
        domain=domain,
        network=routing_network,
    )
    observation_config = loaded.build_observation_config()
    observation_model = ObservationModel(observation_config)

    run_output_dir = loaded.run_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    truth_ds = create_empty_truth_dataset(domain)
    observation_ds = create_empty_observation_dataset(domain)
    truth_ds.attrs["run_name"] = loaded.run_name
    truth_ds.attrs["domain_preset"] = loaded.domain_preset_name
    truth_ds.attrs["scenario_name"] = loaded.scenario_name

    observation_ds.attrs["run_name"] = loaded.run_name
    observation_ds.attrs["domain_preset"] = loaded.domain_preset_name
    observation_ds.attrs["scenario_name"] = loaded.scenario_name

    summary_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []

    print("Synthetic Basin Simulator")
    print(f"Configuration path: {loaded.config_path}")
    print(f"Run name: {loaded.run_name}")
    print(f"Domain preset: {loaded.domain_preset_name}")
    print(f"Scenario name: {loaded.scenario_name}")
    print(f"Output directory: {run_output_dir}")
    print(f"Spatial shape: {domain.shape}")
    print(f"Time steps: {domain.n_steps}")
    print(f"dt_seconds: {domain.time.dt_seconds}")
    print(f"Reservoirs in domain: {len(domain.reservoirs)}")
    print(f"Reservoir regulation enabled: {routing_config.enable_reservoirs}")
    print(f"Sensors in domain: {len(domain.sensors)}")

    for step in range(domain.n_steps):
        timestamp = domain.time.timestamps[step]

        # * Execute Meteo module
        meteo_input = MeteoInput(
            domain=domain,
            step=step,
            timestamp=timestamp,
            previous_state=None,
        )

        meteo_output = meteo_model.step(meteo_input)

        # * Execute Energy module
        energy_input = EnergyInput(
            domain=domain,
            step=step,
            timestamp=timestamp,
            precipitation=meteo_output.precipitation,
            air_temperature=meteo_output.air_temperature,
        )
        energy_output = energy_model.step(energy_input)

        # * Execute Hydro module
        hydro_input = HydroInput(
            domain=domain,
            step=step,
            timestamp=timestamp,
            precipitation=meteo_output.precipitation,
            pet=energy_output.pet,
        )
        hydro_output = hydro_model.step(hydro_input)

        # * Execute Routing module
        routing_input = RegulatedRoutingInput(
            domain=domain,
            step=step,
            timestamp=timestamp,
            surface_runoff=hydro_output.surface_runoff,
            subsurface_runoff=hydro_output.subsurface_runoff,
            pet=energy_output.pet,
        )
        routing_output = routing_model.step(routing_input=routing_input)

        # * Execute Observation module
        observation_input = ObservationInput(
            domain=domain,
            step=step,
            timestamp=timestamp,
            precipitation=meteo_output.precipitation,
            channel_flow=routing_output.channel_flow_m3s,
            reservoir_storage=routing_output.reservoir_storage_m3,
        )
        observation_output = observation_model.step(observation_input)

        observation_rows.extend(
            _build_observation_rows(
                domain=domain,
                step=step,
                timestamp=timestamp,
                meteo_output=meteo_output,
                routing_output=routing_output,
                observation_output=observation_output,
            )
        )

        diagnostics = meteo_model.latest_diagnostics
        assert diagnostics is not None

        observation_diagnostics = observation_model.latest_diagnostics
        assert observation_diagnostics is not None

        state = merge_module_outputs(
            step=step,
            timestamp=timestamp,
            meteo_output=meteo_output,
            energy_output=energy_output,
            hydro_output=hydro_output,
            routing_output=routing_output,
        )

        truth_ds = write_state_to_dataset(
            truth_ds,
            state,
        )

        observation_ds = write_observation_to_dataset(
            observation_ds,
            observation_output,
            step=step,
        )

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
                "channel_flow_mean_m3s": float(routing_output.channel_flow_m3s.mean()),
                "channel_flow_max_m3s": float(routing_output.channel_flow_m3s.max()),
                "outlet_discharge_m3s": float(routing_output.outlet_discharge_m3s),
                "reservoir_inflow_sum_m3s": float(routing_output.reservoir_inflow_m3s.sum()),
                "reservoir_release_sum_m3s": float(routing_output.reservoir_release_m3s.sum()),
                "reservoir_spill_sum_m3s": float(routing_output.reservoir_spill_m3s.sum()),
                "reservoir_storage_sum_m3": float(routing_output.reservoir_storage_m3.sum()),
                "obs_available_count": int(observation_diagnostics.n_available),
                "obs_missing_count": int(observation_diagnostics.n_missing),
                "obs_censored_count": int(observation_diagnostics.n_censored),
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
            f"runoff_mean={hydro_output.surface_runoff.mean():.3f} mm/dt | "
            f"channel_mean={routing_output.channel_flow_m3s.mean():.3f} m3/s | "
            f"outlet={routing_output.outlet_discharge_m3s:.3f} m3/s | "
            f"obs_avail={observation_diagnostics.n_available} | "
            f"obs_missing={observation_diagnostics.n_missing} | "
            f"obs_censored={observation_diagnostics.n_censored}"
        )

    truth_dataset_path = _save_dataset(
        truth_ds,
        output_dir=run_output_dir,
        file_stem="simulation_truth",
    )
    observation_dataset_path = _save_dataset(
        observation_ds,
        output_dir=run_output_dir,
        file_stem="simulation_observations",
    )

    summary_csv_path = run_output_dir / "simulation_summary.csv"
    _write_summary_csv(
        rows=summary_rows,
        output_path=summary_csv_path,
    )

    observation_csv_path = run_output_dir / "simulation_observations.csv"
    _write_summary_csv(
        rows=observation_rows,
        output_path=observation_csv_path,
    )

    print("Run completed successfully.")
    print(f"Truth dataset written to: {truth_dataset_path}")
    print(f"Observation dataset written to: {observation_dataset_path}")
    print(f"Step summary CSV: {summary_csv_path}")
    print(f"Observation CSV: {observation_csv_path}")


if __name__ == "__main__":
    main()
