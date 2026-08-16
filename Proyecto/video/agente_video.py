"""
Agente Q-Learning adaptado al pipeline de video.

Accion: 0 = mantener fase, 1 = cambiar fase.
El agente respeta `fase_completa` del entorno para forzar el cambio
cuando ya expiro el tiempo objetivo.
"""
import numpy as np
import pickle
import os
from typing import Dict


class QLearningVideoAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.Q: Dict = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.pasos = 0

    def obtener_q(self, estado, accion) -> float:
        return self.Q.get((estado, accion), 0.0)

    def elegir_accion(self, estado, explorar: bool = True,
                      fase_completa: bool = False) -> int:
        if fase_completa:
            return 1
        if explorar and np.random.random() < self.epsilon:
            return int(np.random.randint(0, 2))
        q0 = self.obtener_q(estado, 0)
        q1 = self.obtener_q(estado, 1)
        return 0 if q0 >= q1 else 1

    def actualizar(self, estado, accion, recompensa, siguiente_estado):
        q_actual = self.obtener_q(estado, accion)
        q_siguiente = max(
            self.obtener_q(siguiente_estado, 0),
            self.obtener_q(siguiente_estado, 1)
        )
        target = recompensa + self.gamma * q_siguiente
        self.Q[(estado, accion)] = q_actual + self.alpha * (target - q_actual)
        self.pasos += 1

    def guardar(self, ruta: str):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, 'wb') as f:
            pickle.dump({
                'Q': self.Q,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'pasos': self.pasos,
            }, f)

    def cargar(self, ruta: str):
        with open(ruta, 'rb') as f:
            d = pickle.load(f)
            self.Q = d['Q']
            self.alpha = d['alpha']
            self.gamma = d['gamma']
            self.epsilon = d['epsilon']
            self.pasos = d['pasos']
        return self

    def metricas(self) -> dict:
        return {
            'tamano_tabla': len(self.Q),
            'pasos_totales': self.pasos,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
        }
