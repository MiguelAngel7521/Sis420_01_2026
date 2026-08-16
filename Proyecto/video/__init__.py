"""
Modulo de Vision por Computadora para el Semaforo Inteligente.

Pipeline:
  Video -> YOLO/BackgroundSub -> Conteo por direccion
       -> K-Means semi-supervisado -> Nivel de congestion
       -> Q-Learning -> Decision de fase y tiempo
       -> Overlay en video con semaforo rojo/amarillo/verde
"""
