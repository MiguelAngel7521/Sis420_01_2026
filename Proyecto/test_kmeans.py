"""Test del K-Means con datos sinteticos diversos."""
import sys
sys.path.insert(0, r'C:\MIguel\IA\Proyecto')

from video.entrenar_video import (
    generar_datos_sinteticos, etiquetar_por_umbrales, entrenar_clasificador
)

datos = generar_datos_sinteticos(1000)
print(f'Muestras: {len(datos)}')

conteos_max = [max(d.values()) for d in datos]
vacia = sum(1 for c in conteos_max if c == 0)
poca = sum(1 for c in conteos_max if 1 <= c <= 3)
media = sum(1 for c in conteos_max if 4 <= c <= 7)
llena = sum(1 for c in conteos_max if c >= 8)
print(f'Distribucion: vacia={vacia}, poca={poca}, media={media}, llena={llena}')

labels = etiquetar_por_umbrales(datos)
c = entrenar_clasificador(datos, labels, k=4)
print('Mapeo cluster->nivel:', c.cluster_a_nivel)

print('\nPredicciones:')
casos = [
    {'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0},
    {'norte': 2, 'sur': 1, 'este': 0, 'oeste': 0},
    {'norte': 5, 'sur': 4, 'este': 1, 'oeste': 0},
    {'norte': 10, 'sur': 8, 'este': 3, 'oeste': 2},
    {'norte': 12, 'sur': 11, 'este': 10, 'oeste': 8},
]
for p in casos:
    n = c.predecir(p)
    print(f'  N={p["norte"]:2d} S={p["sur"]:2d} E={p["este"]:2d} O={p["oeste"]:2d} -> {n}')

# Test del tiempo
print('\nTiempos calculados:')
from video.entorno_video import SemaforoVideoEnv
env = SemaforoVideoEnv(c)
casos_t = [
    ({'norte': 0, 'sur': 0, 'este': 0, 'oeste': 0}, '0 llenas -> esperado 5s'),
    ({'norte': 8, 'sur': 0, 'este': 0, 'oeste': 0}, '1 llena -> esperado 7s'),
    ({'norte': 8, 'sur': 9, 'este': 0, 'oeste': 0}, '2 llenas -> esperado 10s'),
    ({'norte': 10, 'sur': 8, 'este': 9, 'oeste': 0}, '3 llenas -> esperado 20s'),
    ({'norte': 12, 'sur': 11, 'este': 10, 'oeste': 8}, '4 llenas -> esperado 30s'),
]
for conteo, desc in casos_t:
    t = env.calcular_tiempo_fase(conteo)
    print(f'  {desc}: tiempo = {t}s')
