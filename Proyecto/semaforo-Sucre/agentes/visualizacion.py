"""Visualizaciones del entrenamiento DQN: curvas, tabla Q, metricas.

Uso:
    from agentes.visualizacion import graficar_todo
    graficar_todo(modelo, metricas, dim_estado, dim_accion)
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

DIR_SALIDA = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "resultados", "presentacion", "Graficas")
os.makedirs(DIR_SALIDA, exist_ok=True)


def graficar_todo(modelo, metricas, dim_estado, dim_accion):
    """Genera todas las graficas del entrenamiento.

    Args:
        modelo: agente DQN entrenado (con .modelo y .predecir())
        metricas: dict con keys 'recompensas', 'epsilones', 'perdidas', 'q_por_episodio'
        dim_estado: dimension del estado (15 para Sucre)
        dim_accion: numero de acciones (2 para Sucre)
    """
    recompensas = np.array(metricas.get("recompensas", []))
    epsilones = np.array(metricas.get("epsilones", []))
    perdidas = np.array(metricas.get("perdidas", []))
    q_por_ep = metricas.get("q_por_episodio", [])

    print("\nGenerando graficas de entrenamiento...")

    # 1. Curva de aprendizaje
    _curva_aprendizaje(recompensas)

    # 2. Epsilon por episodio
    _epsilon_decay(epsilones)

    # 3. Perdida (loss)
    if len(perdidas) > 0:
        _curva_perdida(perdidas)

    # 4. Tabla Q conceptual
    _tabla_q_conceptual(modelo, dim_estado, dim_accion)

    # 5. Evolucion de Q-valores
    if len(q_por_ep) > 0:
        _evolucion_q(q_por_ep)

    # 6. Distribucion de recompensas
    _histograma_recompensas(recompensas)

    # 7. Dashboard completo
    _dashboard_completo(recompensas, epsilones, perdidas, q_por_ep)

    print(f"  Graficas guardadas en: {DIR_SALIDA}")
    plt.close("all")


# ============================================================
# 1. CURVA DE APRENDIZAJE
# ============================================================
def _curva_aprendizaje(recompensas):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    episodios = np.arange(1, len(recompensas) + 1)

    ax.plot(episodios, recompensas, "b-", linewidth=2, label="Recompensa por episodio", zorder=3)

    # Media movil
    if len(recompensas) >= 5:
        ventana = max(1, len(recompensas) // 10)
        media = np.convolve(recompensas, np.ones(ventana) / ventana, mode="valid")
        ax.plot(np.arange(ventana, len(recompensas) + 1), media,
                "r-", linewidth=2, label=f"Media movil ({ventana} ep)", zorder=4)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=np.mean(recompensas), color="green", linestyle="--",
               label=f"Promedio: {np.mean(recompensas):.2f}", alpha=0.7)

    ax.fill_between(episodios, recompensas, np.min(recompensas) - 1,
                    alpha=0.1, color="steelblue")

    ax.set_xlabel("Episodio", fontsize=12)
    ax.set_ylabel("Recompensa total (diff-waiting-time)", fontsize=12)
    ax.set_title("Curva de Aprendizaje - Double DQN", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Anotaciones
    mejor_ep = np.argmax(recompensas) + 1
    mejor_val = np.max(recompensas)
    ax.annotate(f"Mejor: ep {mejor_ep} = {mejor_val:.2f}",
                xy=(mejor_ep, mejor_val), xytext=(mejor_ep + 5, mejor_val + 2),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                fontsize=10, fontweight="bold", color="green",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(DIR_SALIDA, "curva_aprendizaje.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  1. curva_aprendizaje.png")


# ============================================================
# 2. EPSILON DECAY
# ============================================================
def _epsilon_decay(epsilones):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")

    episodios = np.arange(1, len(epsilones) + 1)
    ax.plot(episodios, epsilones, "orange", linewidth=2, marker=".", markersize=3)
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="Minimo epsilon (0.01)")

    ax.fill_between(episodios, epsilones, 0, alpha=0.2, color="orange")

    ax.set_xlabel("Episodio", fontsize=12)
    ax.set_ylabel("Epsilon (exploracion)", fontsize=12)
    ax.set_title("Decaimiento de Epsilon - Politica Epsilon-Greedy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(DIR_SALIDA, "epsilon_decay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  2. epsilon_decay.png")


# ============================================================
# 3. PERDIDA (LOSS)
# ============================================================
def _curva_perdida(perdidas):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")

    ax.plot(perdidas, "purple", linewidth=1.5, alpha=0.6, label="Perdida (MSE)")
    if len(perdidas) >= 10:
        ventana = max(1, len(perdidas) // 20)
        media = np.convolve(perdidas, np.ones(ventana) / ventana, mode="valid")
        ax.plot(np.arange(ventana, len(perdidas) + 1), media,
                "red", linewidth=2, label=f"Media movil ({ventana})")

    ax.set_xlabel("Paso de entrenamiento", fontsize=12)
    ax.set_ylabel("Perdida (MSE)", fontsize=12)
    ax.set_title("Funcion de Perdida durante el Entrenamiento", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(DIR_SALIDA, "curva_perdida.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  3. curva_perdida.png")


# ============================================================
# 4. TABLA Q CONCEPTUAL (HEATMAP)
# ============================================================
def _tabla_q_conceptual(modelo, dim_estado, dim_accion):
    """Genera un heatmap de Q-valores para estados de ejemplo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    # Estados de ejemplo
    estados_ejemplo = np.array([
        [1, 0, 0.8, 0.6, 0.1, 0.2, 14, 10, 2, 120, 80, 15, 8, 6, 12],   # Muchos autos horizontal
        [0, 1, 0.1, 0.2, 0.8, 0.6, 2, 3, 14, 10, 15, 120, 80, 6, 8],     # Muchos autos vertical
        [1, 0, 0.4, 0.3, 0.4, 0.3, 5, 4, 5, 50, 40, 50, 10, 8, 10],      # Trafico equilibrado
        [1, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 15, 15, 15],        # Sin autos
        [0, 1, 0.9, 0.8, 0.9, 0.8, 18, 15, 18, 200, 150, 200, 3, 2, 4],  # Hora pico ambas
    ])

    nombres = ["Muchos autos\nHorizontal", "Muchos autos\nVertical",
               "Trafico\nEquilibrado", "Sin autos", "Hora pico\nambas"]

    # Calcular Q-valores
    q_vals = []
    with torch.no_grad():
        for est in estados_ejemplo:
            q = modelo.predecir(est).numpy()
            q_vals.append(q)
    q_vals = np.array(q_vals)

    # Heatmap
    im = axes[0].imshow(q_vals, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
    axes[0].set_xticks(range(dim_accion))
    axes[0].set_xticklabels(["Verde\nHorizontal", "Verde\nVertical"], fontsize=10)
    axes[0].set_yticks(range(len(nombres)))
    axes[0].set_yticklabels(nombres, fontsize=9)
    axes[0].set_title("Q(s, a) para estados de ejemplo", fontsize=12, fontweight="bold")

    for i in range(len(nombres)):
        for j in range(dim_accion):
            val = q_vals[i, j]
            color = "white" if abs(val) > 2 else "black"
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    plt.colorbar(im, ax=axes[0], label="Q-valor")

    # Grafico de barras
    x = np.arange(len(nombres))
    ancho = 0.35
    axes[1].bar(x - ancho / 2, q_vals[:, 0], ancho, label="Q(s, verde Horizontal)",
                color="#4CAF50", alpha=0.8)
    axes[1].bar(x + ancho / 2, q_vals[:, 1], ancho, label="Q(s, verde Vertical)",
                color="#2196F3", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(nombres, fontsize=8)
    axes[1].set_ylabel("Q-valor", fontsize=12)
    axes[1].set_title("Comparacion de Q-valores por accion", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Tabla Q Conceptual - Valores Aprendidos por el DQN",
                 fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(DIR_SALIDA, "tabla_q_conceptual.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  4. tabla_q_conceptual.png")


# ============================================================
# 5. EVOLUCION DE Q-VALORES
# ============================================================
def _evolucion_q(q_por_ep):
    """Muestra como evolucionan los Q-valores a lo largo de los episodios."""
    q_por_ep = np.array(q_por_ep)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

    episodios = np.arange(1, len(q_por_ep) + 1)

    # q_por_ep tiene forma (episodios, 2) - Q-valores para accion 0 y 1
    ax.plot(episodios, q_por_ep[:, 0], "g-", linewidth=2, label="Q(s, verde Horizontal)", alpha=0.8)
    ax.plot(episodios, q_por_ep[:, 1], "b-", linewidth=2, label="Q(s, verde Vertical)", alpha=0.8)

    ax.fill_between(episodios, q_por_ep[:, 0], alpha=0.1, color="green")
    ax.fill_between(episodios, q_por_ep[:, 1], alpha=0.1, color="blue")

    ax.set_xlabel("Episodio", fontsize=12)
    ax.set_ylabel("Q-valor promedio", fontsize=12)
    ax.set_title("Evolucion de Q-valores durante el Entrenamiento", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(DIR_SALIDA, "evolucion_q.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  5. evolucion_q.png")


# ============================================================
# 6. HISTOGRAMA DE RECOMPENSAS
# ============================================================
def _histograma_recompensas(recompensas):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    # Histograma
    axes[0].hist(recompensas, bins=min(20, len(recompensas) // 2),
                 color="steelblue", edgecolor="white", alpha=0.7)
    axes[0].axvline(x=np.mean(recompensas), color="red", linestyle="--",
                    linewidth=2, label=f"Media: {np.mean(recompensas):.2f}")
    axes[0].axvline(x=np.median(recompensas), color="green", linestyle="--",
                    linewidth=2, label=f"Mediana: {np.median(recompensas):.2f}")
    axes[0].set_xlabel("Recompensa", fontsize=12)
    axes[0].set_ylabel("Frecuencia", fontsize=12)
    axes[0].set_title("Distribucion de Recompensas", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Boxplot
    axes[1].boxplot(recompensas, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.6),
                    medianprops=dict(color="red", linewidth=2))
    axes[1].set_xticklabels(["Recompensas"])
    axes[1].set_ylabel("Recompensa", fontsize=12)
    axes[1].set_title("Boxplot de Recompensas", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(DIR_SALIDA, "distribucion_recompensas.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  6. distribucion_recompensas.png")


# ============================================================
# 7. DASHBOARD COMPLETO
# ============================================================
def _dashboard_completo(recompensas, epsilones, perdidas, q_por_ep):
    """Panel unico con todas las metricas."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
    fig.suptitle("Dashboard de Entrenamiento - Double DQN", fontsize=16, fontweight="bold", y=0.98)

    episodios = np.arange(1, len(recompensas) + 1)

    # [0,0] Curva de aprendizaje
    ax = axes[0, 0]
    ax.plot(episodios, recompensas, "b-", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=np.mean(recompensas), color="green", linestyle="--",
               label=f"Prom: {np.mean(recompensas):.2f}", alpha=0.7)
    ax.set_title("Curva de Aprendizaje", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [0,1] Epsilon
    ax = axes[0, 1]
    ax.plot(np.arange(1, len(epsilones) + 1), epsilones, "orange", linewidth=1.5)
    ax.fill_between(np.arange(1, len(epsilones) + 1), epsilones, 0, alpha=0.2, color="orange")
    ax.set_title("Exploracion (Epsilon)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episodio"); ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)

    # [0,2] Perdida
    ax = axes[0, 2]
    if len(perdidas) > 0:
        ax.plot(perdidas, "purple", linewidth=1, alpha=0.5)
        ax.set_title("Perdida (Loss)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Paso"); ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)

    # [1,0] Tabla Q heatmap
    ax = axes[1, 0]
    ax.text(0.5, 0.5, "Ver Tabla Q Conceptual\nen tabla_q_conceptual.png",
            ha="center", va="center", fontsize=11, color="gray",
            transform=ax.transAxes, style="italic")
    ax.set_title("Tabla Q (ver archivo aparte)", fontsize=11, fontweight="bold")
    ax.axis("off")

    # [1,1] Histograma
    ax = axes[1, 1]
    ax.hist(recompensas, bins=min(15, len(recompensas) // 2),
            color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(x=np.mean(recompensas), color="red", linestyle="--", label=f"Media")
    ax.set_title("Distribucion Recompensas", fontsize=11, fontweight="bold")
    ax.set_xlabel("Recompensa"); ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [1,2] Estadisticas
    ax = axes[1, 2]
    ax.axis("off")
    stats_text = (
        f"Estadisticas del Entrenamiento\n"
        f"{'='*30}\n"
        f"Episodios: {len(recompensas)}\n"
        f"Recompensa promedio: {np.mean(recompensas):.2f}\n"
        f"Mejor recompensa: {np.max(recompensas):.2f}\n"
        f"Peor recompensa: {np.min(recompensas):.2f}\n"
        f"Desviacion estandar: {np.std(recompensas):.2f}\n"
        f"Mediana: {np.median(recompensas):.2f}\n"
        f"Epsilon final: {epsilones[-1]:.3f}\n"
        f"Epsilon inicial: {epsilones[0]:.3f}"
    )
    ax.text(0.5, 0.5, stats_text, ha="center", va="center",
            fontsize=10, family="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="gray"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(DIR_SALIDA, "dashboard_entrenamiento.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  7. dashboard_entrenamiento.png")
