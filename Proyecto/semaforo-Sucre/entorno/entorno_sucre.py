"""Entorno SUMO-RL para las calles de Sucre, Bolivia."""

import os
import sumo_rl
from configuracion.parametros_suma import (
    RUTA_NET, RUTA_RUTAS, NOMBRE_CSV_SALIDA,
    USAR_GUI, NUM_SEGUNDOS, TIEMPO_PASO,
    TIEMPO_AMARILLO, VERDE_MIN, VERDE_MAX,
    SEMILLA, FUNCION_RECOMPENSA, DELAY_MS
)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(ruta):
    return os.path.join(RAIZ, ruta) if not os.path.isabs(ruta) else ruta


def obtener_entorno(renderizar=None, funcion_recompensa=None, semilla=None):
    usar_gui = renderizar if renderizar is not None else USAR_GUI

    cmd = f"--delay {DELAY_MS}" if DELAY_MS > 0 else None

    return sumo_rl.SumoEnvironment(
        net_file=_abs(RUTA_NET),
        route_file=_abs(RUTA_RUTAS),
        out_csv_name=_abs(NOMBRE_CSV_SALIDA),
        use_gui=usar_gui,
        num_seconds=NUM_SEGUNDOS,
        delta_time=TIEMPO_PASO,
        yellow_time=TIEMPO_AMARILLO,
        min_green=VERDE_MIN,
        max_green=VERDE_MAX,
        reward_fn=funcion_recompensa or FUNCION_RECOMPENSA,
        single_agent=True,
        sumo_seed=semilla or SEMILLA,
        sumo_warnings=False,
        additional_sumo_cmd=cmd,
    )
