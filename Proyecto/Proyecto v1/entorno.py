"""
ENTORNO DE SIMULACION DE TRAFICO (TrafficEnv)
=============================================

Problema: Semáforo inteligente para una intersección de 4 direcciones.

ELEMENTOS DE APRENDIZAJE POR REFUERZO:
  - Estado:   (autos_norte, autos_sur, autos_este, autos_oeste, fase)
              donde cada cantidad está discretizada: 0, 1, 2, 3 (pocos, medios, muchos, lleno)
              y fase es 0 (N-S verde) o 1 (E-O verde)

  - Entorno:  Intersección de 4 vías. Los autos llegan aleatoriamente
              en cada dirección con tasas de llegada configurables.

  - Acción:   0 = mantener la fase actual
              1 = cambiar a la otra fase

  - Política: ε-greedy (el agente decide si mantener o cambiar el semáforo)

  - Recompensa: -total_autos_esperando (negativo = mientras menos autos esperen, mejor)
                Se busca MINIMIZAR las colas.

  - Función de valor: Q(s, a) = valor esperado de tomar acción a en estado s

ALGORITMO: Q-Learning (como en gymnasium/2_01)

USO:
    from entorno import TrafficEnv
    env = TrafficEnv()
    estado = env.reset()
    for _ in range(3600):  # 1 hora en segundos
        estado, recompensa, done, info = env.step(accion)
"""

import numpy as np
import random


class TrafficEnv:
    """
    Entorno de simulación de tráfico para una intersección de 4 vías.

    Parámetros:
        tasa_norte: probabilidad de que llegue un auto del norte por segundo
        tasa_sur:   probabilidad de que llegue un auto del sur por segundo
        tasa_este:  probabilidad de que llegue un auto del este por segundo
        tasa_oeste: probabilidad de que llegue un auto del oeste por segundo
        max_cola:   máximo de autos que pueden esperar en un carril
        autos_por_verde: cuántos autos pasan por segundo cuando hay luz verde
    """

    def __init__(self, tasa_norte=0.3, tasa_sur=0.3,
                 tasa_este=0.15, tasa_oeste=0.15,
                 max_cola=20, autos_por_verde=2):
        # Tasas de llegada de autos por segundo en cada dirección
        self.tasas = {
            'norte': tasa_norte,
            'sur': tasa_sur,
            'este': tasa_este,
            'oeste': tasa_oeste
        }
        self.max_cola = max_cola
        self.autos_por_verde = autos_por_verde

        # Colas de autos esperando en cada dirección
        self.colas = {'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0}

        # Fase actual del semáforo: 0 = N-S verde, 1 = E-O verde
        self.fase = 0

        # Contador de tiempo dentro de la fase actual
        self.tiempo_fase = 0

        # Tiempo mínimo que una fase debe durar (seguridad)
        self.tiempo_min_fase = 5

        # Tiempo máximo de una fase (para evitar期待os demasiado largos)
        self.tiempo_max_fase = 60

        # Total de pasos transcurridos
        self.pasos = 0

        # Para guardar historial
        self.historial = []

    def obtener_estado(self):
        """
        Devuelve el estado discretizado del entorno.

        Discretización de colas:
            0 = 0 autos (vacío)
            1 = 1-3 autos (poco tráfico)
            2 = 4-7 autos (tráfico moderado)
            3 = 8+ autos  (congestión)
        """
        def discretizar(autos):
            if autos == 0:
                return 0
            elif autos <= 3:
                return 1
            elif autos <= 7:
                return 2
            else:
                return 3

        n = discretizar(self.colas['norte'])
        s = discretizar(self.colas['sur'])
        e = discretizar(self.colas['este'])
        o = discretizar(self.colas['oeste'])

        return (n, s, e, o, self.fase)

    def _llegada_autos(self):
        """Genera llegada aleatoria de autos según las tasas de cada dirección."""
        for direccion in ['norte', 'sur', 'este', 'oeste']:
            if random.random() < self.tasas[direccion]:
                self.colas[direccion] = min(
                    self.colas[direccion] + 1,
                    self.max_cola
                )

    def _pasar_autos(self):
        """
        Los autos en las vías con luz verde avanzan.
        Cada segundo pasan 'autos_por_verde' autos por carril.
        """
        if self.fase == 0:
            # Fase N-S: norte y sur tienen luz verde
            self.colas['norte'] = max(0, self.colas['norte'] - self.autos_por_verde)
            self.colas['sur'] = max(0, self.colas['sur'] - self.autos_por_verde)
        else:
            # Fase E-O: este y oeste tienen luz verde
            self.colas['este'] = max(0, self.colas['este'] - self.autos_por_verde)
            self.colas['oeste'] = max(0, self.colas['oeste'] - self.autos_por_verde)

    def reset(self):
        """
        Reinicia el entorno a su estado inicial.
        Devuelve el estado inicial.
        """
        self.colas = {'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0}
        self.fase = 0
        self.tiempo_fase = 0
        self.pasos = 0
        self.historial = []
        return self.obtener_estado(), {}

    def step(self, accion):
        """
        Ejecuta una acción y avanza el entorno 1 segundo.

        Args:
            accion: 0 = mantener fase, 1 = cambiar fase

        Returns:
            estado, recompensa, terminado, truncado, info
        """
        # Guardar estado anterior para el historial
        estado_anterior = self.obtener_estado()

        # Aplicar acción: cambiar o mantener la fase
        if accion == 1:
            # Solo se puede cambiar si han pasado los segundos mínimos
            if self.tiempo_fase >= self.tiempo_min_fase:
                self.fase = 1 - self.fase  # alternar entre 0 y 1
                self.tiempo_fase = 0

        # Incrementar tiempo de fase
        self.tiempo_fase += 1

        # Llegan nuevos autos
        self._llegada_autos()

        # Pasan autos por los carriles con luz verde
        self._pasar_autos()

        # Calcular recompensa: negativa = menos colas es mejor
        total_autos = sum(self.colas.values())
        recompensa = -total_autos

        # Registrar en historial
        self.historial.append({
            'paso': self.pasos,
            'estado': estado_anterior,
            'colas': dict(self.colas),
            'fase': self.fase,
            'recompensa': recompensa
        })

        self.pasos += 1

        # El episodio termina después de muchos pasos (por conveniencia)
        terminado = False
        truncado = self.pasos >= 3600  # 1 hora simulada

        return self.obtener_estado(), recompensa, terminado, truncado, {}

    def render(self):
        """Muestra el estado actual del semáforo en texto."""
        print(f"\nPaso {self.pasos} - Fase: {'N-S VERDE' if self.fase == 0 else 'E-O VERDE'}")
        print(f"  Norte: {'#' * self.colas['norte']:20s} ({self.colas['norte']})")
        print(f"  Sur:   {'#' * self.colas['sur']:20s} ({self.colas['sur']})")
        print(f"  Este:  {'#' * self.colas['este']:20s} ({self.colas['este']})")
        print(f"  Oeste: {'#' * self.colas['oeste']:20s} ({self.colas['oeste']})")
        print(f"  Recompensa: {-(sum(self.colas.values()))}")


# ========== PRUEBA RÁPIDA ==========
if __name__ == "__main__":
    env = TrafficEnv()
    estado, _ = env.reset()
    print("=== PRUEBA DEL ENTORNO ===")
    print("Estado inicial:", estado)

    for _ in range(10):
        accion = random.randint(0, 1)
        estado, recompensa, done, truncado, info = env.step(accion)
        env.render()
        print(f"  Acción: {'CAMBIAR' if accion == 1 else 'MANTENER'}")
        print(f"  Nuevo estado: {estado}")
