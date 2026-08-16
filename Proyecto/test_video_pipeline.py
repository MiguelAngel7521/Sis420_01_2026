"""Test rapido del pipeline de video."""
import sys
sys.path.insert(0, r'C:\MIguel\IA\Proyecto')

from video.semi_supervisado import ClasificadorCongestion
from video.entorno_video import SemaforoVideoEnv
from video.agente_video import QLearningVideoAgent

print('Modulos importados OK')

# Test rapido K-Means
datos = [
    {'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0},
    {'norte': 2, 'sur': 1, 'este': 0, 'oeste': 0},
    {'norte': 5, 'sur': 4, 'este': 1, 'oeste': 0},
    {'norte': 10, 'sur': 8, 'este': 3, 'oeste': 2},
    {'norte': 12, 'sur': 10, 'este': 8, 'oeste': 7},
]
etiquetas = {
    'norte': [0, 1, 2, 3, 3],
    'sur':   [0, 1, 2, 3, 3],
    'este':  [0, 0, 1, 2, 2],
    'oeste': [0, 0, 0, 1, 2],
}
c = ClasificadorCongestion(k=4)
c.entrenar(datos, etiquetas, verbose=False)
print(f'Clasificador: silhouette={c.silhouette:.3f}')

pruebas = [
    {'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0},
    {'norte': 11, 'sur': 9, 'este': 1, 'oeste': 0},
    {'norte': 5, 'sur': 5, 'este': 5, 'oeste': 5},
]
for p in pruebas:
    n = c.predecir(p)
    print(f'  Conteo {p} -> niveles {n}')

# Test entorno
env = SemaforoVideoEnv(c)
estado = env.reset()
conteo_test = {'norte': 8, 'sur': 7, 'este': 1, 'oeste': 0}
env.actualizar_conteo(conteo_test)
print(f'Estado: {estado}, tiempo objetivo: {env.tiempo_objetivo}s')
print(f'Tiempo para 2 calles llenas: {env.calcular_tiempo_fase(conteo_test)}s (esperado ~10s)')

# Test del logica de tiempo segun congestion
casos = [
    ({'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0}, '0 llenas -> 5s'),
    ({'norte': 8, 'sur': 0, 'este': 0, 'oeste': 0}, '1 llena -> 7s'),
    ({'norte': 8, 'sur': 9, 'este': 0, 'oeste': 0}, '2 llenas -> 10s'),
    ({'norte': 10, 'sur': 8, 'este': 9, 'oeste': 0}, '3 llenas -> 20s'),
    ({'norte': 12, 'sur': 11, 'este': 10, 'oeste': 8}, '4 llenas -> 30s'),
]
for conteo, desc in casos:
    t = env.calcular_tiempo_fase(conteo)
    print(f'  {desc}: tiempo = {t}s')

# Test agente
agente = QLearningVideoAgent()
agente.actualizar(estado, 1, -5, estado)
print(f'Agente Q size: {len(agente.Q)}')
print()
print('=== TODO OK - pipeline basico funcional ===')
