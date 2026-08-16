"""
AGENTE Q-LEARNING PARA SEMAFORO INTELIGENTE
============================================

Basado en Q-Learning (como en gymnasium/2_01_Ejemplo_Algoritmos_Q_LEARNING_y_SARSA.ipynb).

La tabla Q guarda el valor de cada par (estado, accion):
    Q[(estado, accion)] = valor esperado de tomar esa accion en ese estado

Actualizacion (Ecuacion de Bellman):
    Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))

Donde:
    alpha = tasa de aprendizaje (que tan rapido aprende)
    gamma = factor de descuento (importancia de recompensas futuras)
    r     = recompensa inmediata
    s'    = siguiente estado
"""

import numpy as np
import pickle
import os


class QLearningAgent:
    """
    Agente que aprende a controlar el semáforo usando Q-Learning.

    Args:
        alpha: tasa de aprendizaje (0.1 = aprende lento, 0.9 = aprende rapido)
        gamma: factor de descuento (0.9 = valora recompensas futuras)
        epsilon: probabilidad de explorar (0.1 = 10% acciones al azar)
    """

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        # Tabla Q: diccionario donde la clave es (estado, accion) -> valor
        self.Q = {}

        # Hiperparámetros
        self.alpha = alpha      # tasa de aprendizaje
        self.gamma = gamma      # factor de descuento
        self.epsilon = epsilon  # probabilidad de exploracion

        # Contador de pasos (para estadisticas)
        self.pasos = 0

    def obtener_q(self, estado, accion):
        """
        Obtiene el valor Q para un par (estado, accion).
        Si no existe, devuelve 0.
        """
        return self.Q.get((estado, accion), 0.0)

    def elegir_accion(self, estado, explorar=True):
        """
        Elige una accion usando politica epsilon-greedy.

        Args:
            estado: tupla (n, s, e, o, fase)
            explorar: si es True, usa epsilon-greedy

        Returns:
            0 = mantener fase, 1 = cambiar fase
        """
        if explorar and np.random.random() < self.epsilon:
            # EXPLORACION: elegir accion al azar
            return np.random.randint(0, 2)

        # EXPLOTACION: elegir la accion con mayor valor Q
        q0 = self.obtener_q(estado, 0)
        q1 = self.obtener_q(estado, 1)

        if q0 >= q1:
            return 0  # mantener fase
        else:
            return 1  # cambiar fase

    def actualizar(self, estado, accion, recompensa, siguiente_estado):
        """
        Actualiza la tabla Q usando la ecuación de Bellman.

        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        """
        # Valor actual de Q(s,a)
        q_actual = self.obtener_q(estado, accion)

        # Mejor valor del siguiente estado: max(Q(s',a'))
        q_siguiente = max(
            self.obtener_q(siguiente_estado, 0),
            self.obtener_q(siguiente_estado, 1)
        )

        # Ecuación de Bellman: target = r + gamma * max(Q(s'))
        target = recompensa + self.gamma * q_siguiente

        # Actualización: Q(s,a) = Q(s,a) + alpha * (target - Q(s,a))
        nuevo_valor = q_actual + self.alpha * (target - q_actual)

        # Guardar en tabla
        self.Q[(estado, accion)] = nuevo_valor
        self.pasos += 1

    def guardar(self, ruta="modelos/q_table.pickle"):
        """
        Guarda la tabla Q en un archivo.
        """
        directorio = os.path.dirname(ruta)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio)

        with open(ruta, 'wb') as f:
            pickle.dump({
                'Q': self.Q,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'pasos': self.pasos
            }, f)

    def cargar(self, ruta="modelos/q_table.pickle"):
        """
        Carga una tabla Q guardada previamente.
        """
        with open(ruta, 'rb') as f:
            datos = pickle.load(f)
            self.Q = datos['Q']
            self.alpha = datos['alpha']
            self.gamma = datos['gamma']
            self.epsilon = datos['epsilon']
            self.pasos = datos['pasos']

    def obtener_metricas(self):
        """Devuelve estadísticas del agente."""
        return {
            'tamano_tabla': len(self.Q),
            'pasos_totales': self.pasos,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }


# ========== PRUEBA RAPIDA ==========
if __name__ == "__main__":
    from entorno import TrafficEnv

    # Crear entorno y agente
    env = TrafficEnv()
    agente = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.3)

    print("=== PRUEBA DEL AGENTE ===")
    print("Hiperparámetros:", agente.obtener_metricas())

    estado, _ = env.reset()
    recompensa_total = 0

    for i in range(100):
        accion = agente.elegir_accion(estado, explorar=True)
        siguiente_estado, recompensa, done, truncado, info = env.step(accion)
        agente.actualizar(estado, accion, recompensa, siguiente_estado)
        recompensa_total += recompensa
        estado = siguiente_estado

    print(f"Recompensa total en 100 pasos: {recompensa_total}")
    print(f"Estados aprendidos: {len(agente.Q)}")
