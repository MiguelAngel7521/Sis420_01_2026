"""Algoritmo de Q-Learning profundo para control de semaforos."""

import random
import torch
import numpy as np
from collections import deque


def q_learning(entorno, modelo, episodios, gamma=0.95, #factor de descuento 
               epsilon=0.3, decaimiento_eps=0.995,
               usar_replay=True, tamano_replay=64,
               doble=False, n_actualizar=10, verbose=True,
               buffer_max=20000):
    recompensas_por_episodio = []
    epsilones_por_episodio = []
    perdidas_totales = []
    q_por_episodio = []
    espera_acumulada_por_episodio = []
    memoria = deque(maxlen=buffer_max)
    mejor_recompensa = -float("inf")

    for ep in range(episodios):
        obs, info = entorno.reset()
        done = False
        truncado = False
        total_recompensa = 0
        espera_acumulada = 0.0
        paso = 0
        q_acum_0 = 0.0
        q_acum_1 = 0.0
        pasos_q = 0

        while not (done or truncado):
            if random.random() < epsilon:
                accion = entorno.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = modelo.predecir(obs)
                accion = torch.argmax(q_vals).item()

            sig_obs, recompensa, done, truncado, info = entorno.step(accion)
            total_recompensa += recompensa
            espera_acumulada += info.get("system_total_waiting_time", 0)

            with torch.no_grad():
                q_actual = modelo.predecir(obs).numpy()
            q_acum_0 += q_actual[0]
            q_acum_1 += q_actual[1]
            pasos_q += 1

            memoria.append((obs, accion, sig_obs, recompensa, done or truncado))

            if usar_replay:
                perdida = modelo.replay(memoria, tamano_replay, gamma)
                if perdida is not None:
                    perdidas_totales.append(perdida)
            else:
                q_vals = modelo.predecir(obs).tolist()
                q_sig = modelo.predecir(sig_obs)
                q_vals[accion] = recompensa + gamma * (0 if (done or truncado) else torch.max(q_sig).item())
                modelo.actualizar([obs], [q_vals])

            obs = sig_obs
            paso += 1

        if doble and ep % n_actualizar == 0 and ep > 0:
            modelo.actualizar_objetivo()
        #aplicamos E-greidy
        epsilon = max(epsilon * decaimiento_eps, 0.01)
        recompensas_por_episodio.append(total_recompensa)
        epsilones_por_episodio.append(epsilon)
        q_por_episodio.append([q_acum_0 / max(pasos_q, 1), q_acum_1 / max(pasos_q, 1)])
        espera_acumulada_por_episodio.append(espera_acumulada)

        if verbose:
            print(f"Episodio {ep+1}/{episodios} | Recompensa: {total_recompensa:.6f} | Espera acum: {espera_acumulada:.1f}s | Epsilon: {epsilon:.4f} | Buffer: {len(memoria)}")

        if total_recompensa > mejor_recompensa:
            mejor_recompensa = total_recompensa

    print(f"Entrenamiento completado. Mejor recompensa: {mejor_recompensa:.6f}")

    metricas = {
        "recompensas": recompensas_por_episodio,
        "epsilones": epsilones_por_episodio,
        "perdidas": perdidas_totales,
        "q_por_episodio": q_por_episodio,
        "espera_acumulada": espera_acumulada_por_episodio,
    }
    return metricas
