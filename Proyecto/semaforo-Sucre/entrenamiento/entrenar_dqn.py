"""Entrena un agente DQN para controlar semaforos de Sucre."""

import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entorno.entorno_sucre import obtener_entorno
from agentes.dqn import DQN, DQNReplay, DQNDoble, guardar_modelo
from agentes.entrenamiento_q import q_learning
from agentes.visualizacion import graficar_todo
from configuracion.parametros_suma import (
    EPISODIOS, OCULTOS, TASA_APRENDIZAJE, GAMMA, EPSILON_INICIAL,
    DECAIMIENTO_EPS, TAMANO_REPLAY, USAR_REPLAY, DQL_DOBLE,
    N_ACTUALIZACION_OBJETIVO, RUTA_MODELOS, USAR_GUI
)


def main():
    print("Iniciando entorno SUMO-RL para Sucre...")

    entorno = obtener_entorno(renderizar=True)

    dim_estado = entorno.observation_space.shape[0]
    dim_accion = entorno.action_space.n

    print(f"Dimension del estado: {dim_estado}")
    print(f"Acciones disponibles: {dim_accion}")

    if DQL_DOBLE:
        print("Usando Double DQN")
        modelo = DQNDoble(dim_estado, dim_accion, OCULTOS, TASA_APRENDIZAJE)
    elif USAR_REPLAY:
        print("Usando DQN con Replay Buffer")
        modelo = DQNReplay(dim_estado, dim_accion, OCULTOS, TASA_APRENDIZAJE)
    else:
        print("Usando DQN Simple")
        modelo = DQN(dim_estado, dim_accion, OCULTOS, TASA_APRENDIZAJE)

    print(f"\nEntrenando por {EPISODIOS} episodios...")
    metricas = q_learning(
        entorno, modelo, EPISODIOS,
        gamma=GAMMA,
        epsilon=EPSILON_INICIAL,
        decaimiento_eps=DECAIMIENTO_EPS,
        usar_replay=USAR_REPLAY or DQL_DOBLE,
        tamano_replay=TAMANO_REPLAY,
        doble=DQL_DOBLE,
        n_actualizar=N_ACTUALIZACION_OBJETIVO,
    )

    os.makedirs(RUTA_MODELOS, exist_ok=True)
    ruta_modelo = os.path.join(RUTA_MODELOS, "dqn_sucre.pth")
    guardar_modelo(modelo, ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")

    recompensas = metricas["recompensas"]
    q_vals = metricas["q_por_episodio"]
    print(f"\n=== METRICAS DE ENTRENAMIENTO ===")
    print(f"  Recompensa promedio (total): {np.mean(recompensas):.4f}")
    print(f"  Mejor recompensa: {np.max(recompensas):.4f}")
    print(f"  Peor recompensa: {np.min(recompensas):.4f}")
    print(f"  Desviacion estandar: {np.std(recompensas):.4f}")
    print(f"  Mediana: {np.median(recompensas):.4f}")
    print(f"  Epsilon final: {metricas['epsilones'][-1]:.4f}")
    print()
    inicio = np.mean(recompensas[:10])
    final = np.mean(recompensas[-10:])
    mejora = ((final - inicio) / abs(inicio)) * 100 if inicio != 0 else 0
    print(f"  Recompensa primeros 10 ep: {inicio:.4f}")
    print(f"  Recompensa ultimos 10 ep:  {final:.4f}")
    if mejora > 0:
        print(f"  Mejoria: +{mejora:.1f}%  (el agente aprendio a reducir esperas)")
    else:
        print(f"  Mejoria: {mejora:.1f}%  (no hubo mejora significativa)")
    print()
    q0_inicio = np.mean([q[0] for q in q_vals[:10]])
    q1_inicio = np.mean([q[1] for q in q_vals[:10]])
    q0_final = np.mean([q[0] for q in q_vals[-10:]])
    q1_final = np.mean([q[1] for q in q_vals[-10:]])
    print(f"  Q(verde_H) inicio: {q0_inicio:.6f}  |  Q(verde_V) inicio: {q1_inicio:.6f}")
    print(f"  Q(verde_H) final:  {q0_final:.6f}  |  Q(verde_V) final:  {q1_final:.6f}")
    print(f"  Diferencia Q final: {abs(q0_final - q1_final):.6f}  (mayor = mejor discriminacion)")
    print()

    entorno.close()

    # Generar graficas
    graficar_todo(modelo, metricas, dim_estado, dim_accion)
    print("Graficas generadas en: resultados/presentacion/")


if __name__ == "__main__":
    main()
