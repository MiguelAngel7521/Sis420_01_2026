"""
Herramienta para etiquetar manualmente una muestra pequena de frames.
Replica el rol de las etiquetas manuales que se usan en el notebook
lab7_kmeans_aprendizaje_no_supervisado.ipynb (bloodMNIST).

Uso:
    python -m video.etiquetar_manual --video data/videos/mi_video.mp4 --n 30

Pasos:
  1. El script toma 30 frames distribuidos a lo largo del video.
  2. Para cada frame, te muestra la imagen y te pide el nivel (0-3)
     de cada una de las 4 direcciones.
  3. Guarda el resultado en data/labels/labels.csv

Niveles: 0=vacia 1=poca 2=media 3=llena
"""
import cv2
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def etiquetar(video_path, n_muestras=30, salida="data/labels/labels.csv"):
    if not os.path.exists(video_path):
        print(f"ERROR: video no encontrado: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("ERROR: no se pudo leer el video")
        return
    n_muestras = min(n_muestras, total)
    indices = [int(i * total / n_muestras) for i in range(n_muestras)]

    os.makedirs(os.path.dirname(salida), exist_ok=True)
    archivo = open(salida, "w", newline="", encoding="utf-8")
    writer = csv.writer(archivo)
    writer.writerow(["frame_idx", "norte", "sur", "este", "oeste"])

    print(f"\nEtiquetando {n_muestras} frames del video {video_path}")
    print("Para cada frame, ingresa el nivel de congestion por direccion.")
    print("Niveles: 0=vacia  1=poca  2=media  3=llena\n")

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Mostrar el frame redimensionado
        h, w = frame.shape[:2]
        escala = min(800 / w, 600 / h, 1.0)
        vis = cv2.resize(frame, (int(w * escala), int(h * escala)))
        cv2.imshow("Frame a etiquetar", vis)
        cv2.waitKey(300)

        print(f"\n[Frame {i+1}/{n_muestras}  idx={idx}]")
        niveles = {}
        for d in ['norte', 'sur', 'este', 'oeste']:
            while True:
                v = input(f"  {d} (0-3): ").strip()
                if v in ['0', '1', '2', '3']:
                    niveles[d] = int(v)
                    break
                print("    ingresa 0, 1, 2 o 3")
        writer.writerow([idx, niveles['norte'], niveles['sur'],
                         niveles['este'], niveles['oeste']])
        print(f"  -> guardado")

    archivo.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nEtiquetas guardadas en {salida}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--n", type=int, default=30, help="numero de frames a etiquetar")
    p.add_argument("--salida", type=str, default="data/labels/labels.csv")
    args = p.parse_args()
    etiquetar(args.video, args.n, args.salida)


if __name__ == "__main__":
    main()
