"""Prueba de SUMO-RL con una interseccion simple de 4 direcciones.

Usa la red de prueba incluida en sumo-rl (single-intersection).
Equivalente a la red 1x1 del proyecto original TrafficLight-RL-master.
"""

import os
import sys
import torch
import numpy as np
import sumo_rl


RUTA_RED = os.path.join(os.path.dirname(sumo_rl.__file__),
                        "nets", "single-intersection", "single-intersection.net.xml")
RUTA_RUTAS = os.path.join(os.path.dirname(sumo_rl.__file__),
                          "nets", "single-intersection", "single-intersection.rou.xml")

print("=" * 60)
print("PRUEBA DE SUMO-RL CON INTERSECCION SIMPLE")
print("=" * 60)
print(f"\nRed: {RUTA_RED}")
print(f"Rutas: {RUTA_RUTAS}")

entorno = sumo_rl.SumoEnvironment(
    net_file=RUTA_RED,
    route_file=RUTA_RUTAS,
    use_gui=True,
    num_seconds=200,
    delta_time=5,
    yellow_time=2,
    min_green=5,
    max_green=50,
    single_agent=True,
    reward_fn="diff-waiting-time",
    sumo_warnings=False,
)

dim_estado = entorno.observation_space.shape[0]
dim_accion = entorno.action_space.n
print(f"\nDimension del estado: {dim_estado}")
print(f"Acciones (fases de semaforo): {dim_accion}")
print("El estado contiene: [fase_actual, min_verde, densidad_carril1, ..., cola_carril1, ...]")

from agentes.dqn import DQNDoble, guardar_modelo
from agentes.entrenamiento_q import q_learning

modelo = DQNDoble(dim_estado, dim_accion, dim_oculta=64, lr=0.001)

print("\nEntrenando por 20 episodios...")
print("(Se abrira una ventana de SUMO - puedes ver la simulacion)\n")
recompensas = q_learning(
    entorno, modelo, episodios=20,
    gamma=0.95, epsilon=0.3, decaimiento_eps=0.99,
    usar_replay=True, tamano_replay=64,
    doble=True, n_actualizar=10,
)

print(f"\nResultados:")
print(f"  Recompensa promedio: {np.mean(recompensas):.2f}")
print(f"  Mejor recompensa: {max(recompensas):.2f}")
print(f"  Peor recompensa: {min(recompensas):.2f}")
print(f"  Todas las recompensas: {[f'{r:.2f}' for r in recompensas]}")

os.makedirs("modelos_entrenados", exist_ok=True)
guardar_modelo(modelo, "modelos_entrenados/dqn_prueba.pth")
print("\nModelo guardado en: modelos_entrenados/dqn_prueba.pth")

entorno.close()
print("\nPrueba completada!")
print("\nLa red 'single-intersection' es equivalente a la red 1x1")
print("que usaba el proyecto original con FLOW.")
print("\nPara usar tus calles de Sucre:")
print("  1. Pon tus archivos .net.xml y .rou.xml en redes/sucre/")
print("  2. Ejecuta: python -m entrenamiento.entrenar_dqn")
