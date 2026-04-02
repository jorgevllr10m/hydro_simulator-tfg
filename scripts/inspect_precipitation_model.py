from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulator.core.contracts import MeteoInput
from simulator.core.time import TimeDefinition
from simulator.core.types import (
    BasinDefinition,
    GridDefinition,
    SimulationDomain,
    SpatialDomain,
)
from simulator.meteo.latent_state import (
    LatentEnvironmentConfig,
    MoistureScenario,
    ThermalScenario,
)
from simulator.meteo.precipitation_model import (
    StormPrecipitationConfig,
    StormPrecipitationModel,
)
from simulator.meteo.regimes import MeteorologicalRegime
from simulator.meteo.storm_birth import StormBirthConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect-precipitation-model",
        description="Run and visualize the phase-3 storm precipitation model.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="precipitation_model_preview",
        help="Name used for output files and folders.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-09-01T00:00:00",
        help="Simulation start datetime in ISO format.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=12,
        help="Number of simulation steps to run.",
    )
    parser.add_argument(
        "--dt-seconds",
        type=int,
        default=3600,
        help="Time-step duration in seconds.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=40,
        help="Number of grid columns.",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=30,
        help="Number of grid rows.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=1000.0,
        help="Grid spacing in x [m].",
    )
    parser.add_argument(
        "--dy",
        type=float,
        default=1000.0,
        help="Grid spacing in y [m].",
    )
    parser.add_argument(
        "--expected-births",
        type=float,
        default=1.5,
        help="Baseline expected number of new storms per step.",
    )
    parser.add_argument(
        "--max-new-storms",
        type=int,
        default=8,
        help="Maximum number of new storms per step.",
    )
    parser.add_argument(
        "--latent-seed",
        type=int,
        default=1234,
        help="Random seed for the latent meteorological environment.",
    )
    parser.add_argument(
        "--storm-seed",
        type=int,
        default=4321,
        help="Random seed for storm spawning and storm properties.",
    )
    return parser


def build_demo_domain(args: argparse.Namespace) -> SimulationDomain:
    grid = GridDefinition(
        nx=args.nx,
        ny=args.ny,
        dx=args.dx,
        dy=args.dy,
        x0=0.0,
        y0=0.0,
    )

    basin = BasinDefinition(mask=np.ones(grid.shape, dtype=bool))
    spatial = SpatialDomain(grid=grid, basin=basin)

    time = TimeDefinition(
        start=datetime.fromisoformat(args.start),
        dt_seconds=args.dt_seconds,
        n_steps=args.n_steps,
    )

    return SimulationDomain(spatial=spatial, time=time)


def build_demo_model_config(args: argparse.Namespace) -> StormPrecipitationConfig:
    latent_environment = LatentEnvironmentConfig(
        random_seed=args.latent_seed,
        initial_regime=MeteorologicalRegime.CONVECTIVE,
        thermal_scenario=ThermalScenario.WARM,
        moisture_scenario=MoistureScenario.WET,
    )

    birth = StormBirthConfig(
        expected_births_per_step=args.expected_births,
        max_new_storms_per_step=args.max_new_storms,
    )

    return StormPrecipitationConfig(
        latent_environment=latent_environment,
        birth=birth,
        random_seed=args.storm_seed,
    )


def _compute_plot_vmax(field: np.ndarray, fallback: float = 1.0) -> float:
    """Return a positive vmax for plotting non-negative fields."""
    field_max = float(np.nanmax(field))
    if field_max <= 0.0:
        return fallback
    return field_max


def save_step_figure(
    *,
    step: int,
    timestamp: datetime,
    precipitation_mm_dt: np.ndarray,
    accumulated_precipitation_mm: np.ndarray,
    storm_mask: np.ndarray,
    regime_name: str,
    new_storms: int,
    tracked_storms: int,
    wet_cells_step: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 4.8),
        constrained_layout=True,
    )

    step_vmax = _compute_plot_vmax(precipitation_mm_dt, fallback=1.0)
    accumulated_vmax = _compute_plot_vmax(accumulated_precipitation_mm, fallback=1.0)

    im0 = axes[0].imshow(
        precipitation_mm_dt,
        origin="lower",
        vmin=0.0,
        vmax=step_vmax,
        interpolation="nearest",
    )
    axes[0].set_title("Step precipitation [mm/dt]")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(
        accumulated_precipitation_mm,
        origin="lower",
        vmin=0.0,
        vmax=accumulated_vmax,
        interpolation="nearest",
    )
    axes[1].set_title("Accumulated precipitation [mm]")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(
        storm_mask,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[2].set_title("Storm mask [0/1]")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    fig.suptitle(
        f"Step {step:03d} | {timestamp.isoformat()} | "
        f"regime={regime_name} | new={new_storms} | "
        f"tracked={tracked_storms} | wet_cells={wet_cells_step}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_final_accumulation_figure(
    *,
    accumulated_precipitation_mm: np.ndarray,
    output_path: Path,
    run_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)

    accumulated_vmax = _compute_plot_vmax(accumulated_precipitation_mm, fallback=1.0)

    im = ax.imshow(
        accumulated_precipitation_mm,
        origin="lower",
        vmin=0.0,
        vmax=accumulated_vmax,
        interpolation="nearest",
    )
    ax.set_title(f"Final accumulated precipitation | {run_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.9, label="mm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_diagnostics_csv(
    *,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    domain = build_demo_domain(args)
    config = build_demo_model_config(args)
    model = StormPrecipitationModel(config)

    figures_dir = ROOT / "outputs" / "figures" / args.run_name
    diagnostics_dir = ROOT / "outputs" / "diagnostics"

    accumulated_precipitation_mm = np.zeros(domain.shape, dtype=float)
    diagnostics_rows: list[dict[str, object]] = []

    print("Running precipitation-model inspection...")
    print(f"Run name: {args.run_name}")
    print(f"Spatial shape: {domain.shape}")
    print(f"Time steps: {domain.n_steps}")
    print(f"dt_seconds: {domain.time.dt_seconds}")

    for step in range(domain.n_steps):
        meteo_input = MeteoInput(
            domain=domain,
            step=step,
            timestamp=domain.time.timestamps[step],
        )

        meteo_output = model.step(meteo_input)
        diagnostics = model.latest_diagnostics
        assert diagnostics is not None

        storm_mask = meteo_output.storm_mask if meteo_output.storm_mask is not None else np.zeros(domain.shape, dtype=float)

        accumulated_precipitation_mm += meteo_output.precipitation

        wet_cells_step = int(np.count_nonzero(meteo_output.precipitation > 0.0))
        tracked_storms = diagnostics.n_active_storms

        figure_path = figures_dir / f"step_{step:03d}.png"
        save_step_figure(
            step=step,
            timestamp=meteo_input.timestamp,
            precipitation_mm_dt=meteo_output.precipitation,
            accumulated_precipitation_mm=accumulated_precipitation_mm,
            storm_mask=storm_mask,
            regime_name=diagnostics.latent_state.regime.value,
            new_storms=diagnostics.n_new_storms,
            tracked_storms=tracked_storms,
            wet_cells_step=wet_cells_step,
            output_path=figure_path,
        )

        diagnostics_rows.append(
            {
                "step": step,
                "timestamp": meteo_input.timestamp.isoformat(),
                "regime": diagnostics.latent_state.regime.value,
                "background_temperature_c": diagnostics.latent_state.background_temperature_c,
                "storm_trigger_factor": diagnostics.storm_environment.storm_trigger_factor,
                "storm_organization_factor": diagnostics.storm_environment.storm_organization_factor,
                "moisture_availability": diagnostics.storm_environment.moisture_availability,
                "new_storms": diagnostics.n_new_storms,
                "tracked_storms": tracked_storms,
                "wet_cells_step": wet_cells_step,
                "precipitation_sum_mm_dt": float(meteo_output.precipitation.sum()),
                "precipitation_max_mm_dt": float(meteo_output.precipitation.max()),
                "storm_mask_active_cells": int(np.count_nonzero(storm_mask)),
                "air_temperature_mean_c": float(meteo_output.air_temperature.mean()),
            }
        )

        print(
            f"[step={step:03d}] "
            f"regime={diagnostics.latent_state.regime.value} | "
            f"new={diagnostics.n_new_storms} | "
            f"tracked={tracked_storms} | "
            f"wet_cells={wet_cells_step} | "
            f"precip_max={meteo_output.precipitation.max():.3f} mm/dt"
        )

    final_figure_path = figures_dir / "accumulated_precipitation.png"
    save_final_accumulation_figure(
        accumulated_precipitation_mm=accumulated_precipitation_mm,
        output_path=final_figure_path,
        run_name=args.run_name,
    )

    diagnostics_csv_path = diagnostics_dir / f"{args.run_name}_summary.csv"
    write_diagnostics_csv(
        rows=diagnostics_rows,
        output_path=diagnostics_csv_path,
    )

    print("Inspection completed successfully.")
    print(f"Step figures: {figures_dir}")
    print(f"Final accumulation figure: {final_figure_path}")
    print(f"Diagnostics CSV: {diagnostics_csv_path}")


if __name__ == "__main__":
    main()
