"""
Aprendizaje semi-supervisado con K-Means para clasificar congestion.

Replica el patron del notebook `lab7_kmeans_aprendizaje_no_supervisado.ipynb`:
  1. K-Means agrupa todos los puntos (no supervisado).
  2. Una pequena muestra esta etiquetada a mano (vacia / poca / media / llena).
  3. Las etiquetas se propagan al cluster mayoritario (semi-supervisado).

Niveles:
  0 = vacia   (0 autos)
  1 = poca    (1-3 autos)
  2 = media   (4-7 autos)
  3 = llena   (8+ autos)
"""
import numpy as np
import pickle
import os
from typing import Dict, List, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


NIVELES = {0: "vacia", 1: "poca", 2: "media", 3: "llena"}


class ClasificadorCongestion:
    """Clasificador semi-supervisado de congestion por direccion."""

    def __init__(self, k: int = 4):
        self.k = k
        self.kmeans: Optional[KMeans] = None
        self.cluster_a_nivel: Dict[int, int] = {}
        self.silhouette: float = -1.0
        self.entrenado: bool = False

    def _vector_a_features(self, conteos: Dict[str, int]) -> np.ndarray:
        """Convierte dict de conteos por direccion en vector de caracteristicas."""
        return np.array([
            conteos.get('norte', 0),
            conteos.get('sur', 0),
            conteos.get('este', 0),
            conteos.get('oeste', 0),
            sum(conteos.values()),
            max(conteos.values()) if conteos else 0,
        ], dtype=float)

    def _propagar_etiquetas(self, X: np.ndarray,
                            etiquetas: Dict[str, List[int]]):
        """
        Estrategia robusta de propagacion de etiquetas:

        1. Calcula la media del conteo total (feature 4) por cluster.
        2. Ordena los clusters por esa media.
        3. Asigna niveles 0,1,2,3 en orden creciente.

        Esto garantiza una progresion monotona vacia -> llena, sin
        importar como el K-Means haya nombrado los clusters.
        """
        clusters = self.kmeans.labels_
        # Media del total de coches en cada cluster
        totales_por_cluster = {}
        for c in range(self.k):
            mask = (clusters == c)
            totales = X[mask, 4]  # feature 4 = suma total
            totales_por_cluster[c] = float(np.mean(totales)) if len(totales) > 0 else 0.0

        # Ordenar clusters por total promedio (menor a mayor)
        clusters_ordenados = sorted(totales_por_cluster.keys(),
                                    key=lambda c: totales_por_cluster[c])

        # Mapear a niveles 0..k-1 (limitado a max 3)
        self.cluster_a_nivel = {}
        for i, c in enumerate(clusters_ordenados):
            nivel = min(i, 3)
            self.cluster_a_nivel[c] = nivel

        # Si hay etiquetas, ajustar segun ejemplos etiquetados
        if etiquetas:
            nivel_por_cluster_etiquetado = {c: [] for c in range(self.k)}
            for dir_nombre, labels in etiquetas.items():
                labels = np.array(labels, dtype=int)
                for c in range(self.k):
                    mask = (clusters == c)
                    validos = labels[mask]
                    validos = validos[(validos >= 0) & (validos <= 3)]
                    if len(validos) > 0:
                        nivel_por_cluster_etiquetado[c].append(int(np.round(np.mean(validos))))
            # Combinar con la asignacion por total
            for c in range(self.k):
                if nivel_por_cluster_etiquetado[c]:
                    nivel_medio = int(np.round(np.mean(nivel_por_cluster_etiquetado[c])))
                    self.cluster_a_nivel[c] = int(np.clip(nivel_medio, 0, 3))

    def entrenar(self, datos: List[Dict[str, int]],
                 etiquetas: Optional[Dict[str, List[int]]] = None,
                 verbose: bool = True):
        """
        Entrena el K-Means y propaga etiquetas.

        Args:
            datos: lista de dicts con conteos por direccion
                   (ej. [{'norte': 3, 'sur': 0, ...}, ...])
            etiquetas: dict con niveles manuales por direccion,
                       ej. {'norte': [0,1,2,1,3,...], 'sur': [...]}.
                       Si es None, asigna niveles segun el orden del cluster.
        """
        X = np.array([self._vector_a_features(d) for d in datos])
        self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        self.kmeans.fit(X)

        if len(X) > self.k:
            try:
                self.silhouette = float(silhouette_score(X, self.kmeans.labels_))
            except Exception:
                self.silhouette = 0.0

        if etiquetas is not None and len(etiquetas) > 0:
            self._propagar_etiquetas(X, etiquetas)
        else:
            # Modo no supervisado: el cluster 0 -> vacia, ..., cluster k-1 -> llena
            for c in range(self.k):
                self.cluster_a_nivel[c] = min(c, 3)

        self.entrenado = True

        if verbose:
            print(f"  K-Means entrenado con k={self.k}, silhouette={self.silhouette:.3f}")
            print(f"  Mapeo cluster -> nivel: {self.cluster_a_nivel}")
        return self

    def predecir(self, conteos: Dict[str, int]) -> Dict[str, int]:
        """
        Predice el nivel (0-3) de congestion por cada direccion.

        Estrategia hibrida:
          1. Identifica el cluster mas cercano (K-Means).
          2. Obtiene el nivel global del cluster.
          3. Para cada direccion, compara su conteo con el centroide
             del cluster y ajusta +/-1 segun corresponda.
        """
        if not self.entrenado:
            return {k: 0 for k in conteos}
        features = self._vector_a_features(conteos)
        cluster = int(self.kmeans.predict(features.reshape(1, -1))[0])
        nivel_global = self.cluster_a_nivel.get(cluster, 0)
        centroide = self.kmeans.cluster_centers_[cluster]

        niveles = {}
        for i, direccion in enumerate(['norte', 'sur', 'este', 'oeste']):
            count = conteos.get(direccion, 0)
            centroide_dir = centroide[i]
            # Si el conteo supera al centroide, subir nivel
            if count > centroide_dir * 1.5:
                nivel = min(3, nivel_global + 1)
            elif count < centroide_dir * 0.5:
                nivel = max(0, nivel_global - 1)
            else:
                nivel = nivel_global
            niveles[direccion] = nivel
        return niveles

    def predecir_distancia(self, conteos: Dict[str, int]) -> np.ndarray:
        """Devuelve distancias a cada centroide (util para debugging)."""
        if not self.entrenado:
            return np.zeros(self.k)
        features = self._vector_a_features(conteos)
        return self.kmeans.transform(features.reshape(1, -1))[0]

    def guardar(self, ruta: str):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'cluster_a_nivel': self.cluster_a_nivel,
                'k': self.k,
                'silhouette': self.silhouette,
            }, f)

    def cargar(self, ruta: str):
        with open(ruta, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.cluster_a_nivel = data['cluster_a_nivel']
            self.k = data['k']
            self.silhouette = data['silhouette']
            self.entrenado = True
        return self
