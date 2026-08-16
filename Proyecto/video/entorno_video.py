"""
Entorno RL adaptado a video real.

El estado se construye a partir del clasificador semi-supervisado
(K-Means) que recibe el conteo de autos por direccion.

Logica de tiempo de fase (lo que pediste):
  - 0 calles llenas  ->  5s
  - 1 calle llena    ->  7s
  - 2 calles llenas  -> 10s
  - 3 calles llenas  -> 20s
  - 4 calles llenas  -> 30s
Ademas escala segun la cantidad total de coches en esas calles.

Accion del agente Q-Learning:
  0 = mantener la fase actual
  1 = cambiar a la otra fase
El agente tambien puede forzar el cambio si la fase ya esta completa.
"""
import numpy as np
from typing import Dict
from .semi_supervisado import ClasificadorCongestion


class SemaforoVideoEnv:
    """Entorno de semaforo alimentado por deteccion de video."""

    def __init__(self, clasificador: ClasificadorCongestion, max_cola: int = 20,
                 tiempo_min_fase: int = 3):
        self.clasificador = clasificador
        self.max_cola = max_cola
        self.tiempo_min_fase = tiempo_min_fase
        self.fase = 0  # 0 = N-S verde, 1 = E-O verde
        self.tiempo_fase = 0
        self.tiempo_objetivo = 10
        self.paso = 0
        self.ultimo_conteo = {'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0}
        self.historial = []

    def reset(self):
        self.fase = 0
        self.tiempo_fase = 0
        self.tiempo_objetivo = 10
        self.paso = 0
        self.ultimo_conteo = {'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0}
        self.historial = []
        return self.obtener_estado()

    def obtener_estado(self) -> tuple:
        """Estado discreto: (n_norte, n_sur, n_este, n_oeste, fase)."""
        niveles = self.clasificador.predecir(self.ultimo_conteo)
        return (
            niveles.get('norte', 0),
            niveles.get('sur', 0),
            niveles.get('este', 0),
            niveles.get('oeste', 0),
            self.fase,
        )

    def calcular_tiempo_fase(self, conteos: Dict[str, int]) -> int:
        """
        Calcula cuantos segundos debe durar la fase actual.
        - 2+ calles llenas -> minimo 10s
        - 3+ calles llenas -> 20-30s
        """
        niveles = self.clasificador.predecir(conteos)
        n_llenas = sum(1 for v in niveles.values() if v >= 3)
        total_autos = sum(conteos.values())

        if n_llenas >= 4:
            base = 30
        elif n_llenas >= 3:
            base = 20
        elif n_llenas >= 2:
            base = 10
        elif n_llenas == 1:
            base = 7
        else:
            base = 5

        # Escalar por total de coches (mas coches -> un poco mas de tiempo)
        factor = 1.0 + min(total_autos / 40.0, 0.5)
        tiempo = int(np.clip(base * factor, 5, 40))
        return tiempo

    def step(self, accion: int):
        """
        Avanza un segundo en el entorno.

        Args:
            accion: 0 = mantener, 1 = cambiar
        Returns:
            (estado, recompensa, terminado, truncado)
        """
        if accion == 1 and self.tiempo_fase >= self.tiempo_min_fase:
            self.fase = 1 - self.fase
            self.tiempo_fase = 0
            self.tiempo_objetivo = self.calcular_tiempo_fase(self.ultimo_conteo)

        self.tiempo_fase += 1
        self.paso += 1

        niveles = self.clasificador.predecir(self.ultimo_conteo)
        recompensa = -float(sum(niveles.values()))
        # Bonus si cambia cuando hay calles congestionadas esperando
        if accion == 1 and self.tiempo_fase >= self.tiempo_min_fase:
            if sum(1 for v in niveles.values() if v >= 3) >= 2:
                recompensa += 1.0

        self.historial.append({
            'paso': self.paso,
            'fase': self.fase,
            'conteo': dict(self.ultimo_conteo),
            'niveles': niveles,
            'tiempo_fase': self.tiempo_fase,
            'tiempo_objetivo': self.tiempo_objetivo,
            'recompensa': recompensa,
        })
        return self.obtener_estado(), recompensa, False, False

    def actualizar_conteo(self, conteos: Dict[str, int]):
        """Recibe el conteo actual desde el detector de video."""
        self.ultimo_conteo = dict(conteos)
        if self.tiempo_fase == 0:
            self.tiempo_objetivo = self.calcular_tiempo_fase(self.ultimo_conteo)
        return self.obtener_estado()

    @property
    def fase_completa(self) -> bool:
        return self.tiempo_fase >= self.tiempo_objetivo

    def color_semaforo(self, direccion: str) -> tuple:
        """
        Devuelve (color_bgr, nombre) del semaforo para una direccion.
        Reglas:
          - Fase verde para esa direccion + tiempo_ok -> VERDE
          - Fase verde + ultimos 2 segundos -> AMARILLO
          - Resto -> ROJO
        """
        es_ns = direccion in ('norte', 'sur')
        en_fase_verde = (es_ns and self.fase == 0) or (not es_ns and self.fase == 1)
        if not en_fase_verde:
            return (0, 0, 255), 'ROJO'
        if self.tiempo_fase >= self.tiempo_objetivo - 2 and self.tiempo_fase >= 3:
            return (0, 255, 255), 'AMARILLO'
        return (0, 255, 0), 'VERDE'
