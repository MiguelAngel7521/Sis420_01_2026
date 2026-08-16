# Modulo `video/` - Semaforo Inteligente con Video Real

Pipeline completo de vision por computadora + aprendizaje semi-supervisado
+ aprendizaje por refuerzo, aplicado a video real de una interseccion.

## Estructura

```
video/
├── __init__.py
├── README.md                  <- este archivo
├── roi_config.py              <- define las regiones de interes por calle
├── deteccion.py               <- YOLOv8 / MOG2 para detectar autos
├── semi_supervisado.py        <- K-Means semi-supervisado (vacia/poca/media/llena)
├── entorno_video.py           <- entorno RL alimentado por detecciones
├── agente_video.py            <- Q-Learning
├── entrenar_video.py          <- entrena K-Means + Q-Learning
├── procesar_video.py          <- pipeline en tiempo real con un video
└── demo_sintetico.py          <- genera video fake y lo procesa
```

## Por que NO Unreal Engine

UE es un motor de juegos 3D, no una herramienta de vision por computadora.
Para este proyecto usaremos el estandar de la industria:

| Componente | Herramienta | Comando |
|---|---|---|
| Deteccion de vehiculos | YOLOv8 (Ultralytics) | `pip install ultralytics` |
| Procesamiento de video | OpenCV | `pip install opencv-python` |
| K-Means semi-supervisado | scikit-learn | `pip install scikit-learn` |
| Visualizacion | OpenCV (cv2) | ya instalado |

YOLOv8 funciona en CPU a ~5-10 FPS (modelo `yolov8n`) y en GPU a 30+ FPS.

## Instalacion

```bash
pip install ultralytics opencv-python scikit-learn
```

YOLOv8n se descarga automaticamente la primera vez que se usa (~6 MB).

## Uso rapido (3 pasos)

### 1) Entrenar (datos sinteticos del simulador)

```bash
python -m video.entrenar_video
```

Esto:
- Genera 2000 muestras con el `TrafficEnv` existente
- Etiqueta una muestra (simula el "etiquetado manual")
- Entrena K-Means con propagacion de etiquetas
- Entrena Q-Learning (2000 episodios)
- Guarda `modelos_video/clasificador.pickle` y `agente.pickle`

### 2a) Probar con video sintetico (sin camara)

```bash
python -m video.demo_sintetico
```

Genera un video de una interseccion con trafico simulado y lo procesa
en pantalla. Util para verificar que el pipeline funciona.

### 2b) Probar con video real

Coloca tu video en `data/videos/mi_interseccion.mp4` y ejecuta:

```bash
python -m video.procesar_video --video data/videos/mi_interseccion.mp4
```

O usa tu webcam:

```bash
python -m video.procesar_video --webcam
```

### 3) Ajustar las ROIs a tu video

Edita `video/roi_config.py` y modifica el diccionario `rois`. Cada ROI
es un poligono con coordenadas normalizadas (0 a 1):

```python
rois: Dict[str, List[Tuple[float, float]]] = field(
    default_factory=lambda: {
        'norte': [(0.30, 0.00), (0.70, 0.00), (0.70, 0.30), (0.30, 0.30)],
        'sur':   [(0.30, 0.70), (0.70, 0.70), (0.70, 1.00), (0.30, 1.00)],
        'este':  [(0.70, 0.30), (1.00, 0.30), (1.00, 0.70), (0.70, 0.70)],
        'oeste': [(0.00, 0.30), (0.30, 0.30), (0.30, 0.70), (0.00, 0.70)],
    }
)
```

## Logica del semaforo (Q-Learning + tiempo)

El agente Q-Learning decide si mantener o cambiar la fase. El tiempo de
cada fase se calcula a partir de la congestion detectada:

| Calles "llenas" (nivel 3) | Tiempo de fase |
|---|---|
| 0 | 5 segundos |
| 1 | 7 segundos |
| 2 | 10 segundos |
| 3 | 20 segundos |
| 4 | 30 segundos |

Ademas, el tiempo escala segun la cantidad total de coches
(factor entre 1.0 y 1.5).

## Colores del semaforo en pantalla

- **VERDE**: direccion en fase verde
- **AMARILLO**: ultimos 2 segundos antes del cambio
- **ROJO**: direccion en fase roja

## Si no tienes GPU o quieres algo mas liviano

Usa la opcion `--no-yolo` (o no instales ultralytics). El sistema usara
sustraccion de fondo MOG2 de OpenCV. Es menos preciso pero no necesita
descargar nada.

## Teclas en el video en vivo

- `q` = salir
- `p` = pausa
- `ESPACIO` = avanzar un frame (cuando esta en pausa)
- `r` = reiniciar simulacion

## Combinacion con el proyecto original

El modulo `video/` es independiente del `Proyecto/` original (no modifica
`agente.py`, `entorno.py`, `entrenar.py`). Pero puedes compararlos:

1. El proyecto original usa discretizacion fija (umbrales hardcodeados).
2. El modulo `video/` aprende los umbrales desde los datos con K-Means.

Los archivos del proyecto original siguen funcionando tal cual.
