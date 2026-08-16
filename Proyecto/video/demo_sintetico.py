"""
Demo sintetico: genera un video de una interseccion 4-vias con trafico
y lo procesa con el pipeline completo, SIN necesidad de camara real.

Sirve para:
  - Verificar que el pipeline funciona end-to-end
  - Hacer pruebas antes de tener video real
  - Demostrar el proyecto sin grabacion

Uso:
    python -m video.demo_sintetico                       # genera y procesa
    python -m video.demo_sintetico --solo-video out.mp4  # solo guarda el video
    python -m video.demo_sintetico --no-yolo             # sin descargar YOLO
"""
import cv2
import numpy as np
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video.deteccion import DetectorFondo, YOLO_DISPONIBLE
from video.semi_supervisado import ClasificadorCongestion
from video.entorno_video import SemaforoVideoEnv
from video.agente_video import QLearningVideoAgent
from video.roi_config import ConfiguracionVideo, obtener_rois


class ClasificadorDummy:
    def predecir(self, conteos):
        def nivel(n):
            if n == 0: return 0
            elif n <= 3: return 1
            elif n <= 7: return 2
            return 3
        return {k: nivel(v) for k, v in conteos.items()}
    def cargar(self, r): pass


def generar_frame_sintetico(frame_id, width=1280, height=720):
    """
    Genera un frame sintetico de una interseccion 4-vias vista en planta.
    Los autos son circulos de colores que se mueven desde las 4 direcciones.
    """
    frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Fondo gris oscuro

    # Calles (cruz horizontal y vertical grises)
    cv2.rectangle(frame, (width // 2 - 80, 0), (width // 2 + 80, height), (90, 90, 90), -1)
    cv2.rectangle(frame, (0, height // 2 - 80), (width, height // 2 + 80), (90, 90, 90), -1)

    # Lineas de division blancas
    for y in range(0, height, 40):
        cv2.line(frame, (width // 2 - 5, y), (width // 2 - 5, y + 20), (255, 255, 255), 2)
        cv2.line(frame, (width // 2 + 5, y), (width // 2 + 5, y + 20), (255, 255, 255), 2)
    for x in range(0, width, 40):
        cv2.line(frame, (x, height // 2 - 5), (x + 20, height // 2 - 5), (255, 255, 255), 2)
        cv2.line(frame, (x, height // 2 + 5), (x + 20, height // 2 + 5), (255, 255, 255), 2)

    # Centro de la interseccion
    cv2.rectangle(frame,
                  (width // 2 - 80, height // 2 - 80),
                  (width // 2 + 80, height // 2 + 80),
                  (40, 40, 40), -1)

    # Autos moviendose desde cada direccion
    # Norte -> va hacia abajo
    for i in range(6):
        y = (frame_id * 4 + i * 50) % (height - 100)
        x = width // 2 - 30
        cv2.rectangle(frame, (x - 12, y), (x + 12, y + 24), (255, 100, 100), -1)
        cv2.putText(frame, "N", (x - 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Sur -> va hacia arriba
    for i in range(6):
        y = height - ((frame_id * 4 + i * 50) % (height - 100)) - 30
        x = width // 2 + 30
        cv2.rectangle(frame, (x - 12, y), (x + 12, y + 24), (100, 255, 255), -1)
        cv2.putText(frame, "S", (x - 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Este -> va hacia izquierda
    for i in range(6):
        x = width - ((frame_id * 4 + i * 50) % (width - 100)) - 30
        y = height // 2 + 30
        cv2.rectangle(frame, (x, y - 12), (x + 24, y + 12), (255, 100, 255), -1)
        cv2.putText(frame, "E", (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Oeste -> va hacia derecha
    for i in range(6):
        x = (frame_id * 4 + i * 50) % (width - 100)
        y = height // 2 - 30
        cv2.rectangle(frame, (x, y - 12), (x + 24, y + 12), (100, 255, 100), -1)
        cv2.putText(frame, "O", (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return frame


def generar_video_sintetico(ruta_salida, n_frames=600, fps=15,
                            width=1280, height=720):
    """Genera un video MP4 con trafico simulado."""
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))
    print(f"Generando {n_frames} frames en {ruta_salida}...")
    for i in range(n_frames):
        frame = generar_frame_sintetico(i, width, height)
        writer.write(frame)
    writer.release()
    print(f"Video guardado: {ruta_salida}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solo-video", type=str, default=None,
                   help="solo generar el video, sin procesarlo")
    p.add_argument("--no-yolo", action="store_true",
                   help="usar sustraccion de fondo en vez de YOLO")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    ruta_video = "data/videos/demo_sintetico.mp4"

    if args.solo_video:
        generar_video_sintetico(args.solo_video, args.frames, args.fps,
                                args.width, args.height)
        return

    # Generar el video si no existe
    if not os.path.exists(ruta_video):
        generar_video_sintetico(ruta_video, args.frames, args.fps,
                                args.width, args.height)
    else:
        print(f"Usando video existente: {ruta_video}")

    # Ahora procesarlo (importar dinamicamente para no romper todo si falta cv2)
    from video.procesar_video import procesar
    class Args:
        video = ruta_video
        webcam = False
        usar_yolo = (not args.no_yolo) and YOLO_DISPONIBLE
        modelo = "yolov8n.pt"
        device = "cpu"
        clasificador = "modelos_video/clasificador.pickle"
        agente = "modelos_video/agente.pickle"
        salida = None
        loop = True
    procesar(Args())


if __name__ == "__main__":
    main()
