"""
Deteccion de vehiculos en video.

Soporta dos modos:
  1. YOLOv8 (Ultralytics): alta precision, requiere `pip install ultralytics`
  2. Sustraccion de fondo MOG2 (OpenCV): sin dependencias extra, menor precision

Uso:
    detector = DetectorCarros(modelo="yolov8n.pt")
    conteos = detector.contar_en_roi(frame, rois)  # {'norte': 3, 'sur': 1, ...}
    vis = detector.dibujar_detecciones(frame, rois, conteos)
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple
import os

try:
    from ultralytics import YOLO
    YOLO_DISPONIBLE = True
except ImportError:
    YOLO_DISPONIBLE = False


# Mapeo de IDs de clases COCO a nombres comunes
COCO_CLASES = {2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}


class DetectorCarros:
    """Detector de vehiculos basado en YOLOv8."""

    def __init__(self, modelo: str = "yolov8n.pt",
                 clases: List[str] = None,
                 confianza: float = 0.4,
                 device: str = "cpu"):
        if not YOLO_DISPONIBLE:
            raise ImportError(
                "ultralytics no esta instalado. Instala con: pip install ultralytics"
            )
        self.modelo = YOLO(modelo)
        if clases is None:
            clases = ["car", "truck", "bus", "motorcycle"]
        self.clases = clases
        self.confianza = confianza
        self.device = device
        # IDs de COCO para las clases pedidas
        self.clases_ids = [k for k, v in COCO_CLASES.items() if v in clases]

    def detectar(self, frame: np.ndarray):
        """Ejecuta YOLO sobre un frame y devuelve el primer resultado."""
        resultados = self.modelo(
            frame, conf=self.confianza,
            classes=self.clases_ids, verbose=False, device=self.device
        )
        return resultados[0]

    def contar_en_roi(self, frame: np.ndarray,
                      rois: Dict[str, List[Tuple[int, int]]]) -> Dict[str, int]:
        """Cuenta vehiculos cuyo centroide cae dentro de cada ROI."""
        res = self.detectar(frame)
        conteos = {nombre: 0 for nombre in rois}
        poligonos = {n: np.array(p, dtype=np.int32) for n, p in rois.items()}
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
            for nombre, poly in poligonos.items():
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    conteos[nombre] += 1
                    break
        return conteos

    def dibujar_detecciones(self, frame: np.ndarray,
                            rois: Dict[str, List[Tuple[int, int]]] = None,
                            conteos: Dict[str, int] = None) -> np.ndarray:
        """Devuelve una copia del frame con bounding boxes y ROIs dibujadas."""
        vis = frame.copy()
        res = self.detectar(frame)
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls = int(box.cls[0])
            nombre = COCO_CLASES.get(cls, "?")
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, nombre, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if rois:
            colores = {'norte': (255, 100, 100), 'sur': (100, 255, 255),
                       'este': (255, 100, 255), 'oeste': (100, 255, 100)}
            for nombre, poly in rois.items():
                color = colores.get(nombre, (255, 255, 255))
                cv2.polylines(vis, [np.array(poly)], True, color, 2)
                if conteos and nombre in conteos:
                    x, y = poly[0]
                    cv2.putText(vis, f"{nombre[:3].upper()}: {conteos[nombre]}",
                                (x, max(20, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis


class DetectorFondo:
    """
    Alternativa sin dependencias de ML: sustraccion de fondo MOG2.
    Detecta vehiculos como blobs en movimiento.
    """

    def __init__(self, history: int = 200, var_threshold: int = 50,
                 area_min: int = 800):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.area_min = area_min

    def detectar(self, frame: np.ndarray) -> np.ndarray:
        """Devuelve una mascara binaria con los blobs en movimiento."""
        mask = self.subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def contar_en_roi(self, frame: np.ndarray,
                      rois: Dict[str, List[Tuple[int, int]]]) -> Dict[str, int]:
        mask = self.detectar(frame)
        conteos = {}
        for nombre, poly in rois.items():
            mask_roi = np.zeros_like(mask)
            cv2.fillPoly(mask_roi, [np.array(poly)], 255)
            roi_mask = cv2.bitwise_and(mask, mask_roi)
            contornos, _ = cv2.findContours(
                roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            conteos[nombre] = sum(1 for c in contornos
                                  if cv2.contourArea(c) > self.area_min)
        return conteos

    def dibujar_detecciones(self, frame: np.ndarray,
                            rois: Dict[str, List[Tuple[int, int]]] = None,
                            conteos: Dict[str, int] = None) -> np.ndarray:
        vis = frame.copy()
        mask = self.detectar(frame)
        # Dibuja contornos en verde
        contornos, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contornos:
            if cv2.contourArea(c) > self.area_min:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if rois:
            colores = {'norte': (255, 100, 100), 'sur': (100, 255, 255),
                       'este': (255, 100, 255), 'oeste': (100, 255, 100)}
            for nombre, poly in rois.items():
                color = colores.get(nombre, (255, 255, 255))
                cv2.polylines(vis, [np.array(poly)], True, color, 2)
                if conteos and nombre in conteos:
                    x, y = poly[0]
                    cv2.putText(vis, f"{nombre[:3].upper()}: {conteos[nombre]}",
                                (x, max(20, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis


def crear_detector(usar_yolo: bool = True, **kwargs):
    """Factory: crea YOLO si esta disponible, si no MOG2."""
    if usar_yolo and YOLO_DISPONIBLE:
        return DetectorCarros(**kwargs)
    return DetectorFondo()
