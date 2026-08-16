"""
ENTRENAMIENTO DEL AGENTE Q-LEARNING PARA SEMAFORO INTELIGENTE
===============================================================

Entrena al agente durante miles de episodios.
Al final guarda la tabla Q en modelos/q_table.pickle.

Estructura del entrenamiento (como en 02_bandits.ipynb):
    for episodio in range(total_episodios):
        for paso in range(pasos_por_episodio):
            accion = agente.elegir_accion(estado, explorar=True)
            nuevo_estado, recompensa = entorno.step(accion)
            agente.actualizar(estado, accion, recompensa, nuevo_estado)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from entorno import TrafficEnv
from agente import QLearningAgent


def entrenar(episodios=5000, pasos_por_episodio=300, alpha=0.1, gamma=0.95, epsilon=0.3):
    """
    Entrena al agente en el entorno de tráfico.

    Args:
        episodios: numero de episodios de entrenamiento
        pasos_por_episodio: segundos simulados por episodio (5 minutos)
        alpha: tasa de aprendizaje
        gamma: factor de descuento
        epsilon: probabilidad de exploracion inicial

    Returns:
        agente entrenado, historial de recompensas
    """
    env = TrafficEnv()
    agente = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)

    historial_recompensas = []
    historial_colas = []

    print("=== ENTRENANDO SEMAFORO INTELIGENTE ===")
    print(f"Episodios: {episodios}")
    print(f"Pasos por episodio: {pasos_por_episodio}")
    print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}")
    print(f"Tasas de llegada: N={env.tasas['norte']}, S={env.tasas['sur']}, "
          f"E={env.tasas['este']}, O={env.tasas['oeste']}")
    print("-" * 50)

    for ep in tqdm(range(episodios)):
        estado, _ = env.reset()
        recompensa_total = 0
        colas_totales = 0

        for paso in range(pasos_por_episodio):
            accion = agente.elegir_accion(estado, explorar=True)
            nuevo_estado, recompensa, done, truncado, info = env.step(accion)
            agente.actualizar(estado, accion, recompensa, nuevo_estado)

            recompensa_total += recompensa
            colas_totales += sum(env.colas.values()) / 4  # promedio de colas
            estado = nuevo_estado

        historial_recompensas.append(recompensa_total)
        historial_colas.append(colas_totales / pasos_por_episodio)  # cola promedio

        # Reducir epsilon gradualmente para pasar de explorar a explotar
        if epsilon > 0.01:
            epsilon = max(0.01, epsilon * 0.9995)
            agente.epsilon = epsilon

    print(f"\nEntrenamiento completado!")
    print(f"Estados aprendidos: {len(agente.Q)}")
    print(f"Recompensa promedio (ultimos 100 ep.): {np.mean(historial_recompensas[-100:]):.1f}")
    print(f"Cola promedio (ultimos 100 ep.): {np.mean(historial_colas[-100:]):.2f} autos")

    return agente, historial_recompensas, historial_colas


def graficar_resultados(historial_recompensas, historial_colas):
    """Grafica la evolución del aprendizaje."""
    plt.figure(figsize=(12, 4))

    # Grafica 1: Recompensa total por episodio
    plt.subplot(1, 2, 1)
    plt.plot(historial_recompensas, alpha=0.3, label='Cruda')
    ventana = 100
    suave = np.convolve(historial_recompensas,
                        np.ones(ventana) / ventana, mode='valid')
    plt.plot(suave, label=f'Media móvil ({ventana})', color='red', linewidth=2)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa total')
    plt.title('Recompensa por episodio')
    plt.legend()
    plt.grid(True)

    # Grafica 2: Cola promedio por episodio
    plt.subplot(1, 2, 2)
    plt.plot(historial_colas, alpha=0.3, label='Cruda')
    suave = np.convolve(historial_colas,
                        np.ones(ventana) / ventana, mode='valid')
    plt.plot(suave, label=f'Media móvil ({ventana})', color='red', linewidth=2)
    plt.xlabel('Episodio')
    plt.ylabel('Cola promedio (autos)')
    plt.title('Cola promedio por episodio')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('modelos/curva_aprendizaje.png', dpi=150)
    plt.show()


def comparar_con_semaforo_fijo(agente, pasos=3600):
    """
    Compara el agente entrenado contra un semáforo de tiempo fijo
    (30 segundos cada fase, sin importar el tráfico).

    Args:
        agente: agente entrenado
        pasos: segundos a simular (1 hora)

    Returns:
        cola_promedio_agente, cola_promedio_fijo
    """
    env = TrafficEnv()

    # --- Semáforo con agente inteligente ---
    estado, _ = env.reset()
    colas_agente = []

    for _ in range(pasos):
        accion = agente.elegir_accion(estado, explorar=False)
        estado, recompensa, done, truncado, info = env.step(accion)
        colas_agente.append(sum(env.colas.values()) / 4)

    cola_prom_agente = np.mean(colas_agente)

    # --- Semáforo de tiempo fijo (30s cada fase) ---
    env2 = TrafficEnv()
    estado, _ = env2.reset()
    colas_fijo = []
    tiempo_fase = 0

    for _ in range(pasos):
        if tiempo_fase >= 30:
            # Cambiar de fase cada 30 segundos
            env2.step(1)
            tiempo_fase = 0
        else:
            env2.step(0)
        tiempo_fase += 1
        colas_fijo.append(sum(env2.colas.values()) / 4)

    cola_prom_fijo = np.mean(colas_fijo)

    return cola_prom_agente, cola_prom_fijo


# ========== EJECUTAR ENTRENAMIENTO ==========
if __name__ == "__main__":
    # Entrenar agente
    agente, recompensas, colas = entrenar(
        episodios=5000,
        pasos_por_episodio=300,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.3
    )

    # Graficar resultados
    graficar_resultados(recompensas, colas)

    # Guardar agente
    agente.guardar('modelos/q_table.pickle')
    print("\nAgente guardado en modelos/q_table.pickle")

    # Comparar con semáforo fijo
    print("\n=== COMPARACION CON SEMAFORO FIJO ===")
    cola_agente, cola_fijo = comparar_con_semaforo_fijo(agente, pasos=3600)
    mejora = ((cola_fijo - cola_agente) / cola_fijo) * 100
    print(f"Cola promedio (agente inteligente): {cola_agente:.2f} autos")
    print(f"Cola promedio (semáforo fijo 30s): {cola_fijo:.2f} autos")
    print(f"Mejora: {mejora:.1f}%")
