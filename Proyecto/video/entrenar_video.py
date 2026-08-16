"""
Entrenamiento del pipeline completo de video:

  1) Genera datos sinteticos de trafico (usa TrafficEnv existente).
  2) Etiqueta una pequena muestra por umbrales (simula el etiquetado manual).
  3) Entrena K-Means semi-supervisado.
  4) Entrena el agente Q-Learning con el entorno de video.
  5) Guarda todo en modelos_video/.

Uso:
    python -m video.entrenar_video
"""
import os
import sys
import numpy as np

# Permitir import del paquete padre
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video.semi_supervisado import ClasificadorCongestion
from video.entorno_video import SemaforoVideoEnv
from video.agente_video import QLearningVideoAgent
from entorno import TrafficEnv


def generar_datos_sinteticos(n_muestras: int = 2000, semilla: int = 42):
    """
    Usa el simulador para generar muestras de conteo de coches.

    Crea tres tipos de entorno con tasas de llegada distintas para
    obtener ejemplos de los 4 niveles de congestion:
      - Valle (tasas bajas)
      - Normal (tasas del simulador original)
      - Pico (tasas altas, trafico denso)
    """
    np.random.seed(semilla)
    datos = []

    configs = [
        # (tasa_norte, tasa_sur, tasa_este, tasa_oeste, proporcion)
        (0.10, 0.10, 0.05, 0.05, 0.20),  # Valle
        (0.30, 0.30, 0.15, 0.15, 0.50),  # Normal
        (0.70, 0.70, 0.50, 0.50, 0.30),  # Pico
    ]

    for tn, ts, te, to, prop in configs:
        n = int(n_muestras * prop)
        env = TrafficEnv(tasa_norte=tn, tasa_sur=ts, tasa_este=te, tasa_oeste=to)
        for _ in range(n):
            env.reset()
            pasos = np.random.randint(30, 150)
            for _ in range(pasos):
                env.step(np.random.randint(0, 2))
            datos.append(dict(env.colas))

    np.random.shuffle(datos)
    return datos


def etiquetar_por_umbrales(datos):
    """
    Simula el etiquetado manual: usa los mismos umbrales del K-means
    (0, 1-3, 4-7, 8+). En la practica, esto seria una muestra pequena
    etiquetada a mano.
    """
    def nivel(n):
        if n == 0:
            return 0
        elif n <= 3:
            return 1
        elif n <= 7:
            return 2
        return 3

    etiquetas = {d: [] for d in ['norte', 'sur', 'este', 'oeste']}
    for d in datos:
        for direccion in etiquetas:
            etiquetas[direccion].append(nivel(d.get(direccion, 0)))
    return etiquetas


def entrenar_clasificador(datos, etiquetas, k=4):
    clasificador = ClasificadorCongestion(k=k)
    clasificador.entrenar(datos, etiquetas)
    return clasificador


def entrenar_agente(clasificador, episodios=2000, pasos=300,
                    alpha=0.1, gamma=0.95, epsilon_inicial=0.3):
    agente = QLearningVideoAgent(alpha=alpha, gamma=gamma, epsilon=epsilon_inicial)
    env = SemaforoVideoEnv(clasificador)
    recompensas = []
    epsilon = epsilon_inicial

    print(f"  Entrenando Q-Learning: {episodios} episodios x {pasos} pasos")
    for ep in range(episodios):
        estado = env.reset()
        recompensa_total = 0
        for _ in range(pasos):
            accion = agente.elegir_accion(
                estado, explorar=True, fase_completa=env.fase_completa
            )
            # Simular pequenas variaciones en el trafico
            nuevo = {k: max(0, v + np.random.randint(-1, 3))
                     for k, v in env.ultimo_conteo.items()}
            env.actualizar_conteo(nuevo)
            nuevo_estado, r, _, _ = env.step(accion)
            agente.actualizar(estado, accion, r, nuevo_estado)
            recompensa_total += r
            estado = nuevo_estado
        recompensas.append(recompensa_total)

        epsilon = max(0.01, epsilon * 0.9995)
        agente.epsilon = epsilon
        if (ep + 1) % 500 == 0:
            media = np.mean(recompensas[-100:])
            print(f"    Episodio {ep+1}: "
                  f"recompensa media (ult 100) = {media:.1f}, "
                  f"estados aprendidos = {len(agente.Q)}")
    return agente, recompensas


def main():
    print("=" * 65)
    print("ENTRENAMIENTO PIPELINE VIDEO")
    print("=" * 65)

    print("\n[1/4] Generando datos sinteticos con TrafficEnv...")
    datos = generar_datos_sinteticos(2000)
    print(f"      {len(datos)} muestras generadas")

    print("\n[2/4] Etiquetando muestra (semi-supervisado)...")
    etiquetas = etiquetar_por_umbrales(datos)
    total_labels = sum(len(v) for v in etiquetas.values())
    print(f"      {total_labels} etiquetas (4 dirs x {len(datos)} muestras)")

    print("\n[3/4] Entrenando K-Means semi-supervisado...")
    clasificador = entrenar_clasificador(datos, etiquetas, k=4)
    ruta_clasif = "modelos_video/clasificador.pickle"
    clasificador.guardar(ruta_clasif)
    print(f"      Guardado en {ruta_clasif}")

    print("\n[4/4] Entrenando agente Q-Learning...")
    agente, recompensas = entrenar_agente(clasificador, episodios=2000)
    ruta_agente = "modelos_video/agente.pickle"
    agente.guardar(ruta_agente)
    print(f"      Guardado en {ruta_agente}")

    print("\n" + "=" * 65)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"  - Estados aprendidos: {len(agente.Q)}")
    print(f"  - Recompensa media (ult 100): {np.mean(recompensas[-100:]):.1f}")
    print(f"  - Silhouette K-Means: {clasificador.silhouette:.3f}")
    print("=" * 65)
    print("\nSiguiente paso: ejecuta `python -m video.procesar_video`")
    print("Si no tienes video, ejecuta primero: `python -m video.demo_sintetico`")


if __name__ == "__main__":
    main()
