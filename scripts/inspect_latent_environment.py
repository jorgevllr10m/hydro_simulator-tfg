from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

from simulator.meteo.latent_state import (
    LatentEnvironmentConfig,
    LatentEnvironmentModel,
    MoistureScenario,
    ThermalScenario,
)

OUTPUT_PATH = Path("outputs/diagnostics/latent_environment_preview.csv")


def main() -> None:
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = LatentEnvironmentConfig(
        random_seed=1234,
        thermal_scenario=ThermalScenario.NORMAL,
        moisture_scenario=MoistureScenario.NORMAL,
    )
    model = LatentEnvironmentModel(config)

    start = datetime(2026, 1, 1, 0, 0, 0)
    n_steps = 60
    dt = timedelta(days=1)

    previous_state = None
    rows: list[dict[str, object]] = []

    for step in range(n_steps):
        timestamp = start + step * dt

        state = model.next_state(
            step=step,
            timestamp=timestamp,
            previous_state=previous_state,
        )
        storm_input = model.build_storm_environment_input(state)

        rows.append(
            {
                "step": state.step,
                "timestamp": state.timestamp.isoformat(),
                "regime": state.regime.value,
                "background_temperature_c": round(state.background_temperature_c, 3),
                "advection_speed_mps": round(state.advection_speed_mps, 3),
                "advection_direction_deg": round(state.advection_direction_deg, 3),
                "advection_u_mps": round(state.advection_u_mps, 3),
                "advection_v_mps": round(state.advection_v_mps, 3),
                "antecedent_wetness_index": round(state.antecedent_wetness_index, 3),
                "cloudiness_index": round(state.cloudiness_index, 3),
                "convective_potential_index": round(state.convective_potential_index, 3),
                "seasonality_factor": round(state.seasonality_factor, 3),
                "scenario_moisture_factor": round(state.scenario_moisture_factor, 3),
                "storm_trigger_factor": round(storm_input.storm_trigger_factor, 3),
                "storm_organization_factor": round(storm_input.storm_organization_factor, 3),
            }
        )

        previous_state = state

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Latent environment diagnostic written to: {output_path}")
    print()
    print("Preview:")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
