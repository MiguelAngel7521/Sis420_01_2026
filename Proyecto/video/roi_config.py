"""
Configuracion de Regiones de Interes (ROIs) por direccion de la interseccion.

Para un video de una interseccion de 4 vias, se define un poligono
(normalizado 0-1 respecto al tamano del frame) por cada calle.

El usuario puede editar ROIs para sus videos: el ROI debe cubrir el area
donde se acumulan los autos de cada direccion (cerca del semaforo).
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class ROI:
    """Region de interes poligonal para una direccion de la calle."""
    nombre: str
    polygon: List[Tuple[float, float]]

    def to_absolute(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Convierte coords normalizadas (0-1) a pixeles absolutos."""
        return [(int(x * width), int(y * height)) for x, y in self.polygon]


@dataclass
class ConfiguracionVideo:
    """Configuracion completa del video y la interseccion."""
    ruta_video: str = "data/videos/interseccion.mp4"
    ancho_frame: int = 1280
    alto_frame: int = 720
    # Clases COCO de vehiculos
    clases_detectar: List[str] = field(
        default_factory=lambda: ["car", "truck", "bus", "motorcycle"]
    )

    # ROIs por direccion (coordenadas normalizadas 0-1).
    # Por defecto: interseccion vista cenital con Norte arriba.
    # Ajustar segun el angulo de la camara.
    rois: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=lambda: {
            'norte': [(0.30, 0.00), (0.70, 0.00), (0.70, 0.30), (0.30, 0.30)],
            'sur':   [(0.30, 0.70), (0.70, 0.70), (0.70, 1.00), (0.30, 1.00)],
            'este':  [(0.70, 0.30), (1.00, 0.30), (1.00, 0.70), (0.70, 0.70)],
            'oeste': [(0.00, 0.30), (0.30, 0.30), (0.30, 0.70), (0.00, 0.70)],
        }
    )


CONFIG_DEFAULT = ConfiguracionVideo()


def obtener_rois(config: ConfiguracionVideo) -> Dict[str, ROI]:
    """Devuelve las ROIs como objetos ROI."""
    return {nombre: ROI(nombre, poly) for nombre, poly in config.rois.items()}


def obtener_rois_absolutas(config: ConfiguracionVideo) -> Dict[str, List[Tuple[int, int]]]:
    """Devuelve las ROIs en coordenadas absolutas para el frame actual."""
    rois = obtener_rois(config)
    return {n: r.to_absolute(config.ancho_frame, config.alto_frame) for n, r in rois.items()}
