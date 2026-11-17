"""Bronze -> Silver: pose de PadelTracker100 a una tabla limpia en Parquet.

Una fila = una persona anotada en un frame.

Decisiones de limpieza (documentadas en docs/entregas/02_datos_y_preparacion.md):
- El número de frame se toma del file_name, nunca del image_id 
(en el partido masculino faltan 65 frames y el image_id se desalinea).
- Los keypoints se normalizan por nombre a un orden canónico
(el orden izquierda/derecha cambia entre los dos partidos).
- Keypoints no anotados (visibilidad 0) -> NaN.
- Frames con más de 4 personas: se eliminan duplicados (IoU > 0.7).
- Frames con menos de 4 personas se conservan y se marcan.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
BRONZE = RAIZ / "data/bronze/padeltracker100/labels"
SILVER = RAIZ / "data/silver"
SAMPLE = RAIZ / "data/sample"
FPS = 30.0
PARTIDOS = ["2022_BCN_FinalF_1", "2022_BCN_FinalM_1"]

# Orden canónico COCO de los 17 keypoints
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def num_frame(nombre: str) -> int:
    """'frame_000123.PNG' -> 123"""
    return int(re.search(r"(\d+)", nombre).group(1))


def iou(a, b) -> float:
    """IoU entre dos cajas [x, y, w, h]."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def cargar_pose(partido: str) -> pd.DataFrame:
    with open(BRONZE / f"{partido}_pose.json", encoding="utf-8") as fh:
        d = json.load(fh)

    cat = next(c for c in d["categories"] if c["name"] == "person")
    nombres = cat["keypoints"]  # orden propio de ESTE fichero
    img2frame = {i["id"]: num_frame(i["file_name"]) for i in d["images"]}

    filas = []
    for a in d["annotations"]:
        if a["category_id"] != cat["id"]:
            continue
        k = a["keypoints"]
        fila = {
            "match_id": partido,
            "frame": img2frame[a["image_id"]],
            "ann_id": a["id"],
            "bbox_x": a["bbox"][0], "bbox_y": a["bbox"][1],
            "bbox_w": a["bbox"][2], "bbox_h": a["bbox"][3],
            "occluded": a.get("attributes", {}).get("occluded")
        }
        for j, nombre in enumerate(nombres):  # se guarda por NOMBRE
            fila[f"{nombre}_x"] = k[3 * j]
            fila[f"{nombre}_y"] = k[3 * j + 1]
            fila[f"{nombre}_v"] = int(k[3 * j + 2])
        filas.append(fila)

    df = pd.DataFrame(filas)
    base = ["match_id", "frame", "ann_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "occluded"]
    cols_kp = [f"{n}_{s}" for n in KEYPOINTS for s in ("x", "y", "v")]
    return df[base + cols_kp]  # columnas en orden canónico


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    informe = {"filas_iniciales": len(df)}

    # Keypoints no anotados -> NaN
    for n in KEYPOINTS:
        sin_anotar = df[f"{n}_v"] == 0
        df.loc[sin_anotar, [f"{n}_x", f"{n}_y"]] = np.nan

    # Duplicados en frames con más de 4 personas
    n_por_frame = df.groupby("frame")["ann_id"].transform("size")
    eliminar = set()
    for _, g in df[n_por_frame > 4].groupby("frame"):
        g = g.assign(area=g.bbox_w * g.bbox_h).sort_values("area", ascending=False)
        cajas = g[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy()
        ids = g["ann_id"].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if ids[j] not in eliminar and iou(cajas[i], cajas[j]) > 0.7:
                    eliminar.add(ids[j])
    df = df[~df["ann_id"].isin(eliminar)].copy()
    informe["duplicados_eliminados"] = len(eliminar)

    # Tiempo y nº de personas por frame
    df["t_s"] = df["frame"] / FPS
    df["n_personas_frame"] = df.groupby("frame")["ann_id"].transform("size")

    informe["filas_finales"] = len(df)
    informe["frames"] = df["frame"].nunique()
    informe["frames_con_menos_de_4"] = int((df.groupby("frame").size() < 4).sum())
    informe["frames_con_mas_de_4"] = int((df.groupby("frame").size() > 4).sum())
    informe["pct_tobillos_visibles"] = float(round(
        100 * ((df["left_ankle_v"] > 0) & (df["right_ankle_v"] > 0)).mean(), 2
    ))
    return df.sort_values(["frame", "bbox_y"]).reset_index(drop=True), informe


def main():
    SILVER.mkdir(parents=True, exist_ok=True)
    SAMPLE.mkdir(parents=True, exist_ok=True)
    for partido in PARTIDOS:
        df, informe = limpiar(cargar_pose(partido))
        df.to_parquet(SILVER / f"{partido}_pose.parquet", index=False)
        df.head(200).to_csv(SAMPLE / f"silver_{partido}_pose.csv", index=False)
        print(partido, informe)


if __name__ == "__main__":
    main()