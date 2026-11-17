"""Silver -> Gold: posiciones en metros, puestos en pista y zona (red/fondo)

La fuente 'gt' es el ground truth de PadelTracker100 (tobillos anotados) proyectado a la
pista. En la Entrega 3, los modelos añadirán filas con otras fuentes ('baseline', 'pose')
a la misma tabla, para compararlas en igualdad de condiciones.

Proyección a metros = homografía (precisa en el suelo) + corrección de paralaje del tobillo (modelo 3D de la cámara).

Gold se reconstruye desde cero en cada ejecución (pipeline determinista).
Si se vuelve a ejecutar, hay que relanzar después los modelos.
"""
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pandas as pd

from padel_vision.vision.camara import Camara, calibrar_camara, px_a_metros_altura
from padel_vision.vision.pista import (
    ANCHO, LARGO, RED_Y,
    calcular_homografia, cargar_calibracion, error_validacion,
    punto_suelo_px, px_a_metros,
)

RAIZ = Path(__file__).resolve().parents[3]
SILVER = RAIZ / "data/silver"
VIDEOS = RAIZ / "data/bronze/padeltracker100"
SAMPLE = RAIZ / "data/sample"
DB = RAIZ / "data/gold/padel.duckdb"
SCHEMA = RAIZ / "db/schema.sql"

FPS = 30.0
ALTURA_TOBILLO_M = 0.12  # altura del tobillo sobre el suelo, ajustada con analisis/simetria.py
UMBRAL_RED_M = 5.5       # revisa con analisis/simetria.py
MARGEN_PISTA_M = 1.0     # margen fuera de las líneas antes de descartar a alguien

PARTIDOS = {"2022_BCN_FinalF_1": "femenino", "2022_BCN_FinalM_1": "masculino"}
PUESTOS = [
    ("cerca_izq", "cerca", "izq", "Fondo cercano a la cámara, izquierda"),
    ("cerca_der", "cerca", "der", "Fondo cercano a la cámara, derecha"),
    ("lejos_izq", "lejos", "izq", "Fondo lejano, izquierda"),
    ("lejos_der", "lejos", "der", "Fondo lejano, derecha"),
]


def proyectar(df: pd.DataFrame, H: np.ndarray, cam: Camara,
              altura_m: float = ALTURA_TOBILLO_M) -> pd.DataFrame:
    """Añade el punto de apoyo en píxeles y su posición en metros.

    1. La homografía da la posición suponiendo que el tobillo está en el suelo.
    2. El modelo de cámara calcula cuánto se desplaza ese punto si en realidad está a
       `altura_m` metros, y se suma ese desplazamiento (corrección de paralaje).
    """
    df = df.copy()
    suelo = punto_suelo_px(df)
    df["u_px"], df["v_px"] = suelo[:, 0], suelo[:, 1]
    xy_suelo = px_a_metros(H, suelo)
    correccion = px_a_metros_altura(cam, suelo, altura_m) - px_a_metros_altura(cam, suelo, 0.0)
    xy = xy_suelo + correccion
    df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]
    return df.dropna(subset=["x_m", "y_m"])


def filtrar_pista(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina a quien esté fuera de la pista (recogepelotas, árbitro...)."""
    m = MARGEN_PISTA_M
    dentro = df["x_m"].between(-m, ANCHO + m) & df["y_m"].between(-m, LARGO + m)
    return df[dentro]


def asignar_puestos(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna a cada jugador su puesto (lado y posición) y descarta frames no válidos.

    Un frame es válido si hay exactamente 2 jugadores en cada lado de la red.
    Los cambios de lado (jugadores cruzando la red) quedan fuera; no son juego.
    """
    df = df.copy()
    df["lado"] = np.where(df["y_m"] < RED_Y, "cerca", "lejos")
    df = df[df.groupby(["frame", "lado"])["ann_id"].transform("size") == 2]
    df = df[df.groupby("frame")["ann_id"].transform("size") == 4].copy()
    orden_x = df.groupby(["frame", "lado"])["x_m"].rank(method="first")
    df["puesto_id"] = df["lado"] + "_" + np.where(orden_x == 1, "izq", "der")
    return df


def zona(df: pd.DataFrame, umbral_m: float = UMBRAL_RED_M) -> pd.DataFrame:
    df = df.copy()
    df["dist_red_m"] = (df["y_m"] - RED_Y).abs()
    df["zona"] = np.where(df["dist_red_m"] < umbral_m, "red", "fondo")
    return df


def info_video(partido: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(VIDEOS / f"{partido}.mp4"))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se encuentra el vídeo de {partido} en {VIDEOS}")
    datos = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return datos


def main():
    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)  # reconstrucción desde cero
    con = duckdb.connect(str(DB))
    con.execute(SCHEMA.read_text(encoding="utf-8"))

    con.executemany("INSERT INTO dim_puesto VALUES (?, ?, ?, ?)", PUESTOS)

    for partido, categoria in PARTIDOS.items():
        ancho, alto, n_frames = info_video(partido)
        con.execute(
            "INSERT INTO dim_partido VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [partido, categoria, "PadelTracker100 (WPT Barcelona 2022)", "CC BY 4.0",
             FPS, ancho, alto, n_frames],
        )

        pts_px, pts_m = cargar_calibracion(partido)
        H = calcular_homografia(pts_px, pts_m)
        loo = error_validacion(pts_px, pts_m)
        cam = calibrar_camara(pts_px, pts_m, ancho, alto)
        cx, cy, cz = cam.centro
        con.execute(
            "INSERT INTO dim_calibracion VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [partido, 0, len(pts_px), H.flatten().tolist(),
             loo["error_medio_m"], loo["error_max_m"],
             float(cx), float(cy), float(cz), float(cam.K[0, 0]), cam.rms_px,
             ALTURA_TOBILLO_M],
        )

        silver = pd.read_parquet(SILVER / f"{partido}_pose.parquet")
        proyectado = proyectar(silver, H, cam)
        en_pista = filtrar_pista(proyectado)
        con_puesto = asignar_puestos(en_pista)
        gold = zona(con_puesto)
        gold["fuente"] = "gt"
        gold["confianza"] = np.nan
        gold["match_id"] = partido

        con.register("gold_df", gold)
        con.execute("""
            INSERT INTO fact_posicion
            SELECT match_id, frame, t_s, puesto_id, fuente, u_px, v_px,
                   x_m, y_m, dist_red_m, zona, confianza
            FROM gold_df
        """)
        con.unregister("gold_df")

        print(f"\n{partido}")
        print(f"  cámara estimada (x, y, z) m:  ({cx:.2f}, {cy:.2f}, {cz:.2f}) | "
              f"focal {cam.K[0, 0]:.0f} px | error reproyección {cam.rms_px:.1f} px")
        print(f"  personas en Silver:          {len(silver)}")
        print(f"  descartadas fuera de pista:  {len(proyectado) - len(en_pista)}")
        print(f"  frames en Silver:            {silver['frame'].nunique()}")
        print(f"  frames válidos (2+2):        {gold['frame'].nunique()}")

    # Métrica táctica, calculada en SQL dentro de la base de datos
    con.execute("""
        INSERT INTO fact_metrica
        SELECT p.match_id, d.lado, p.fuente,
               round(100 * avg(CASE WHEN p.zona = 'red'   THEN 1 ELSE 0 END), 1) AS pct_red,
               round(100 * avg(CASE WHEN p.zona = 'fondo' THEN 1 ELSE 0 END), 1) AS pct_fondo,
               count(DISTINCT p.frame) AS n_frames
        FROM fact_posicion p
        JOIN dim_puesto d USING (puesto_id)
        GROUP BY p.match_id, d.lado, p.fuente
    """)
    print(f"\nMétrica (% del tiempo en red por lado, umbral {UMBRAL_RED_M} m):")
    print(con.sql("SELECT * FROM fact_metrica ORDER BY match_id, lado").df())

    SAMPLE.mkdir(parents=True, exist_ok=True)
    con.sql("SELECT * FROM fact_posicion ORDER BY match_id, frame, puesto_id LIMIT 200") \
       .df().to_csv(SAMPLE / "gold_fact_posicion.csv", index=False)
    con.sql("SELECT * FROM fact_metrica").df().to_csv(SAMPLE / "gold_fact_metrica.csv", index=False)
    con.close()
    print("\nBase de datos Gold en", DB)


if __name__ == "__main__":
    main()