# Control Inteligente de Semaforos con RL - Sucre, Bolivia

Proyecto de **Aprendizaje por Refuerzo (RL)** para optimizar el flujo vehicular en las calles de **Sucre, Bolivia** usando **SUMO-RL**.

## Requisitos

1. **SUMO 1.27+** - Descargar: https://sumo.dlr.de/releases/1.27.0/sumo-win64-1.27.0.msi
2. **Python 3.8+**
3. **Dependencias Python:**
   ```powershell
   pip install sumo-rl torch torchvision numpy matplotlib gymnasium
   ```
   Opcional (para algoritmos modernos):
   ```powershell
   pip install stable-baselines3[extra]
   ```

## Estructura

```
TrafficLight-RL-Espanol/
├── configuracion/parametros_suma.py   ← Un solo archivo de config
├── entorno/entorno_sucre.py           ← Crea el entorno SUMO-RL
├── agentes/
│   ├── dqn.py                         ← DQN, DQNReplay, DQNDoble
│   └── entrenamiento_q.py             ← Algoritmo q_learning
├── entrenamiento/entrenar_dqn.py      ← Script de entrenamiento
├── prediccion/predecir.py             ← Script de prediccion
├── redes/sucre/                       ← Tus archivos SUMO van AQUI
│   ├── sucre.net.xml                  ← Red vial de Sucre
│   ├── sucre.rou.xml                  ← Rutas de vehiculos
│   └── sucre.sumocfg                  ← Config de simulacion
├── modelos_entrenados/                ← Modelos .pth guardados aqui
└── resultados/                        ← CSV de resultados aqui
```

## Como usar con las calles de Sucre

### 1. Colocar tus archivos SUMO

Copia los archivos que generaste en SUMO dentro de la carpeta:

```
redes/sucre/
├── sucre.net.xml     ← Tu red vial de Sucre
├── sucre.rou.xml     ← Tus rutas de vehiculos
└── sucre.sumocfg     ← Tu config de simulacion
```

### 2. Entrenar el agente

```powershell
cd TrafficLight-RL-Espanol
python -m entrenamiento.entrenar_dqn
```

### 3. Probar el modelo entrenado

```powershell
python -m prediccion.predecir --modo modelo
```

### 4. Probar linea base (semaforo estatico)

```powershell
python -m prediccion.predecir --modo linea_base
```

## Ajustar configuracion

Todo se configura en `configuracion/parametros_suma.py`:

| Parametro | Que hace | Default |
|---|---|---|
| SEGUNDOS_SIMULACION | Duracion de cada episodio | 3600 |
| TIEMPO_PASO | Segundos entre acciones | 5 |
| TIEMPO_AMARILLO | Duracion del amarillo | 3 |
| VERDE_MIN | Minimo tiempo en verde | 10 |
| VERDE_MAX | Maximo tiempo en verde | 50 |
| EPISODIOS | Numero de episodios de entrenamiento | 200 |
| TASA_APRENDIZAJE | Learning rate de la red | 0.001 |
| GAMMA | Factor de descuento | 0.95 |
| FUNCION_RECOMPENSA | Funcion de recompensa | diff-waiting-time |

## Generar archivos SUMO para Sucre (osmWebWizard)

Si aun no tienes los archivos:

```powershell
python "%SUMO_HOME%/tools/osmWebWizard.py"
```

1. Selecciona **Sucre, Bolivia** en el mapa
2. Ajusta el area a la zona que te interesa
3. Haz clic en "Generate Scenario"
4. Los archivos se generan en una carpeta con fecha
5. Copia `*.net.xml`, `*.rou.xml` y `*.sumocfg` a `redes/sucre/`
6. Renombralos como `sucre.net.xml`, `sucre.rou.xml`, `sucre.sumocfg`
