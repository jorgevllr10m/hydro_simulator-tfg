from datetime import datetime

from simulator.core.time import TimeDefinition


def test_time_domain():
    # Define el tiempo de inicio y paso temporal
    start = datetime(2026, 1, 1, 0, 0, 0)
    dt_seconds = 3600 * 24 * 20
    n_steps = 6

    # Crea el dominio temporal
    time_domain = TimeDefinition(start=start, dt_seconds=dt_seconds, n_steps=n_steps)

    # Imprime los resultados
    print(time_domain.timestamps)  # Timestamps generados
    print(time_domain.step_index)  # Índice de pasos
    print(time_domain.months)  # Ver los meses
    print(time_domain.seasons)  # Ver las estaciones
