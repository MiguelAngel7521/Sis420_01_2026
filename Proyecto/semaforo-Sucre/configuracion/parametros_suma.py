"""Parametros de configuracion para el proyecto de control de semaforos con RL."""

# RUTAS DE ARCHIVOS SUMO 
RUTA_REDES = "redes"
CIUDAD = "sucre"

RUTA_NET = f"{RUTA_REDES}/{CIUDAD}/Sucre.net.xml"
RUTA_RUTAS = f"{RUTA_REDES}/{CIUDAD}/Sucre.flujo_medio.rou.xml"

#  PARAMETROS DE SIMULACION SUMO-RL 
USAR_GUI = False  # False = entrenamiento rapido (sin ventana), True = con ventana
NUM_SEGUNDOS = 3600
TIEMPO_PASO = 5
TIEMPO_AMARILLO = 3
VERDE_MIN = 10
VERDE_MAX = 50
SEMILLA = "random"

# PARAMETROS DQN 
EPISODIOS = 300
OCULTOS = 64
TASA_APRENDIZAJE = 0.001
GAMMA = 0.99
EPSILON_INICIAL = 0.5
DECAIMIENTO_EPS = 0.992
TAMANO_REPLAY = 128
USAR_REPLAY = True
DQL_DOBLE = True
N_ACTUALIZACION_OBJETIVO = 20

# FUNCION DE RECOMPENSA 
# Opciones: "diff-waiting-time", "average-speed", "queue", "pressure"
FUNCION_RECOMPENSA = "diff-waiting-time"
#Diferencia en el Tiempo de Espera
#El modelo no calcula la recompensa por sí mismo; es 
#el simulador de tráfico (SUMO) el que mide la física de las calles de Sucre en cada paso de tiempo y le entrega la calificación al agente.

# VELOCIDAD DE SIMULACION (solo GUI)
# 0 = max velocidad, mayor valor = mas lento (ms por paso)
DELAY_MS =0

# RUTAS DE SALIDA 
RUTA_MODELOS = "modelos_entrenados"
RUTA_RESULTADOS = "resultados"
NOMBRE_CSV_SALIDA = f"{RUTA_RESULTADOS}/{CIUDAD}"
