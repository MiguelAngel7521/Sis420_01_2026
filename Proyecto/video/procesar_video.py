"""
Procesa video en tiempo real con el pipeline completo.

  Video  --->  YOLO/MOG2  --->  conteo por direccion
                                   |
                                   v
                          K-Means semi-supervisado
                                   |
                                   v
                              Q-Learning
                                   |
                                   v
                    Semaforo rojo/amarillo/verde en pantalla

Uso:
    python -m video.procesar_video --video data/videos/mi_video.mp4
    python -m video.procesar_video --webcam          # usa camara 0
    python -m video.procesar_video --no-yolo         # usa MOG2 (sin descargas)

Teclas en tiempo real:
    q = salir
    p = pausa
    ESPACIO = avanzar 1 frame
    r = reiniciar simulacion
"""
import cv2
import numpy as np
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video.deteccion import DetectorCarros, DetectorFondo, crear_detector, YOLO_DISPONIBLE
from video.semi_supervisado import ClasificadorCongestion
from video.entorno_video import SemaforoVideoEnv
from video.agente_video import QLearningVideoAgent
from video.roi_config import ConfiguracionVideo, obtener_rois


class ClasificadorDummy:
    """Fallback: clasifica por umbrales si no hay K-Means entrenado."""
    def predecir(self, conteos):
        def nivel(n):
            if n == 0: return 0
            elif n <= 3: return 1
            elif n <= 7: return 2
            return 3
        return {k: nivel(v) for k, v in conteos.items()}
    def cargar(self, r): pass


def dibujar_semaforo(frame, x, y, color, estado, tamano=22):
    """Dibuja un semaforo vertical de 3 luces en (x, y)."""
    w, h = tamano, 3 * tamano + 8
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    colores = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]
    for i, c in enumerate(colores):
        cy = y + 4 + i * (tamano + 2)
        if c == color:
            cv2.circle(frame, (x + w // 2, cy), tamano // 2 - 2, c, -1)
            cv2.circle(frame, (x + w // 2, cy), tamano // 2 - 2, (255, 255, 255), 1)
        else:
            cv2.circle(frame, (x + w // 2, cy), tamano // 2 - 2, (40, 40, 40), -1)
    # Etiqueta
    cv2.putText(frame, estado, (x + w + 5, y + h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def procesar(args):
    # Configuracion
    config = ConfiguracionVideo(ruta_video=args.video)
    cap = cv2.VideoCapture(0 if args.webcam else args.video)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir {'webcam' if args.webcam else args.video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    config.ancho_frame = width
    config.alto_frame = height
    print(f"Video: {width}x{height} @ {fps:.1f} FPS")

    # ROIs
    rois_obj = obtener_rois(config)
    rois_abs = {n: r.to_absolute(width, height) for n, r in rois_obj.items()}

    # Detector
    if args.usar_yolo and YOLO_DISPONIBLE:
        detector = DetectorCarros(modelo=args.modelo, device=args.device)
        print("Detector: YOLOv8")
    else:
        detector = DetectorFondo()
        print("Detector: MOG2 (sustraccion de fondo)")

    # Clasificador
    clasificador = None
    if args.clasificador and os.path.exists(args.clasificador):
        clasificador = ClasificadorCongestion(k=4)
        clasificador.cargar(args.clasificador)
        print(f"Clasificador K-Means cargado: {args.clasificador}")
    if clasificador is None or not getattr(clasificador, 'entrenado', False):
        clasificador = ClasificadorDummy()
        print("Clasificador: dummy (umbrales). Entrena primero con --entrenar")

    # Agente
    agente = QLearningVideoAgent()
    if args.agente and os.path.exists(args.agente):
        agente.cargar(args.agente)
        print(f"Agente Q-Learning cargado: {args.agente}")
    else:
        print("Agente: sin entrenar (politica fija)")

    # Entorno
    env = SemaforoVideoEnv(clasificador)
    estado = env.reset()

    # Ventana
    cv2.namedWindow("Semaforo Inteligente - Video Real", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Semaforo Inteligente - Video Real", width, height)

    print("\nControles: [q]salir  [p]pausa  [ESPACIO]frame  [r]reset")
    pausado = False
    frame_count = 0
    t_inicio = time.time()

    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            frame_count += 1

        # 1) Deteccion de vehiculos
        conteos = detector.contar_en_roi(frame, rois_abs)
        env.actualizar_conteo(conteos)

        # 2) Agente Q-Learning decide
        accion = agente.elegir_accion(
            estado, explorar=False, fase_completa=env.fase_completa
        )
        nuevo_estado, recompensa, _, _ = env.step(accion)
        estado = nuevo_estado

        # 3) Visualizar detecciones
        if hasattr(detector, 'dibujar_detecciones'):
            vis = detector.dibujar_detecciones(frame, rois_abs, conteos)
        else:
            vis = frame.copy()

        alto, ancho = vis.shape[:2]

        # 4) Dibujar semaforos
        pos = {
            'norte': (ancho // 2 - 30, 30),
            'sur':   (ancho // 2 + 5, alto - 100),
            'este':  (ancho - 80, alto // 2 - 40),
            'oeste': (10, alto // 2 - 40),
        }
        for direccion, (x, y) in pos.items():
            color, estado_s = env.color_semaforo(direccion)
            dibujar_semaforo(vis, x, y, color, estado_s)

        # 5) Panel de informacion
        cv2.rectangle(vis, (0, 0), (ancho, 90), (0, 0, 0), -1)
        cv2.putText(vis,
                    f"Fase: {'N-S' if env.fase == 0 else 'E-O'}  "
                    f"Tiempo: {env.tiempo_fase}/{env.tiempo_objetivo}s  "
                    f"Paso: {frame_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis,
                    f"Conteos: N={conteos['norte']} S={conteos['sur']} "
                    f"E={conteos['este']} O={conteos['oeste']}  "
                    f"Recompensa: {recompensa:+.1f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        niveles = clasificador.predecir(conteos)
        cv2.putText(vis,
                    f"Niveles: N={niveles.get('norte',0)} S={niveles.get('sur',0)} "
                    f"E={niveles.get('este',0)} O={niveles.get('oeste',0)} "
                    f"(0=vacio 1=poco 2=medio 3=lleno)",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)

        # Calcular FPS real
        t_ahora = time.time()
        fps_real = frame_count / max(1e-6, (t_ahora - t_inicio))
        cv2.putText(vis, f"FPS: {fps_real:.1f}", (ancho - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Salida
        cv2.imshow("Semaforo Inteligente - Video Real", vis)
        if args.salida:
            # Si se quiere grabar, inicializar fuera del bucle
            pass

        # Teclas
        delay = max(1, int(1000 / fps))
        key = cv2.waitKey(0 if pausado else delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            pausado = not pausado
        elif key == ord(' '):
            pausado = True
        elif key == ord('r'):
            estado = env.reset()
            frame_count = 0
            t_inicio = time.time()
            print("Simulacion reiniciada")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcesados {frame_count} frames en {time.time()-t_inicio:.1f}s "
          f"({fps_real:.1f} FPS)")


def main():
    p = argparse.ArgumentParser(description="Semaforo inteligente sobre video real")
    p.add_argument("--video", type=str, default="data/videos/interseccion.mp4")
    p.add_argument("--webcam", action="store_true", help="usar webcam (camara 0)")
    p.add_argument("--usar-yolo", dest="usar_yolo", action="store_true", default=True)
    p.add_argument("--no-yolo", dest="usar_yolo", action="store_false")
    p.add_argument("--modelo", type=str, default="yolov8n.pt")
    p.add_argument("--device", type=str, default="cpu", help="cpu o cuda")
    p.add_argument("--clasificador", type=str, default="modelos_video/clasificador.pickle")
    p.add_argument("--agente", type=str, default="modelos_video/agente.pickle")
    p.add_argument("--salida", type=str, default=None, help="video de salida .mp4")
    p.add_argument("--loop", action="store_true", help="reiniciar video al finalizar")
    args = p.parse_args()
    procesar(args)


if __name__ == "__main__":
    main()
