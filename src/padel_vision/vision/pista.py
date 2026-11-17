"""
Modelo de la pista de pádel y homografía imagen -> pista (metros).

Sistema de coordenadas: origen en la esquina izquierda del fondo cercano a la cámara,
x a lo ancho (0-10 m), y a lo largo (0-20 m). Red en y = 10 m.
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
CALIBRACION = RAIZ / "data/calibracion"
SILVER = RAIZ / "data/silver"
FIGURAS = RAIZ / "docs/img"

# --- Dimensiones reglamentarias (metros) ---
ANCHO = 10.0
LARGO = 20.0
RED_Y = 10.0
DIST_SAQUE_RED = 6.95


def cargar_calibracion(partido: str) -> tuple[np.ndarray, np.ndarray]:
    """Lee los puntos marcados y devuelve (pts_px, pts_m), ignorando los saltados."""
    datos = json.loads((CALIBRACION / f"{partido}.json").read_text(encoding="utf-8"))
    validos = [p for p in datos["puntos"] if p["u_px"] is not None]
    pts_px = np.array([[p["u_px"], p["v_px"]] for p in validos], dtype=np.float64)
    pts_m = np.array([[p["x_m"], p["y_m"]] for p in validos], dtype=np.float64)
    return pts_px, pts_m


def calcular_homografia(pts_px: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    """Matriz H (3x3) que lleva puntos del suelo en la imagen (px) a la pista (m)."""
    if len(pts_px) < 4:
        raise ValueError(f"Hacen falta al menos 4 puntos y hay {len(pts_px)}")
    H, _ = cv2.findHomography(pts_px, pts_m)  # mínimos cuadrados con todos los puntos
    return H


def px_a_metros(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Convierte puntos (N, 2) en píxeles a (N, 2) en metros."""
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def error_validacion(pts_px: np.ndarray, pts_m: np.ndarray) -> dict:
    """Validación leave-one-out: error al proyectar cada punto con una H que no lo ha visto."""
    errores = []
    for i in range(len(pts_px)):
        H_i = calcular_homografia(np.delete(pts_px, i, axis=0), np.delete(pts_m, i, axis=0))
        proyectado = px_a_metros(H_i, pts_px[i : i + 1])[0]
        errores.append(float(np.linalg.norm(proyectado - pts_m[i])))
    return {
        "error_medio_m": round(float(np.mean(errores)), 3),
        "error_max_m": round(float(np.max(errores)), 3),
        "errores_m": [round(e, 3) for e in errores],
    }


def punto_suelo_px(df: pd.DataFrame) -> np.ndarray:
    """Punto de apoyo en el suelo de cada persona: punto medio de los dos tobillos.

    Si falta un tobillo, usa el otro (pandas ignora los NaN al hacer la media).
    """
    x = df[["left_ankle_x", "right_ankle_x"]].mean(axis=1)
    y = df[["left_ankle_y", "right_ankle_y"]].mean(axis=1)
    return np.column_stack([x, y])


def dibujar_pista(ax=None):
    """Dibuja la pista vista desde arriba."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 8))
    ax.add_patch(plt.Rectangle((0, 0), ANCHO, LARGO, facecolor="#2a5caa", edgecolor="white", lw=2))
    ax.plot([0, ANCHO], [RED_Y, RED_Y], color="white", lw=3)  # red
    for y in (RED_Y - DIST_SAQUE_RED, RED_Y + DIST_SAQUE_RED):  # líneas de saque
        ax.plot([0, ANCHO], [y, y], color="white", lw=1.5)
    ax.plot([ANCHO / 2] * 2, [RED_Y - DIST_SAQUE_RED, RED_Y + DIST_SAQUE_RED],
            color="white", lw=1.5)  # línea central (solo entre las de saque)
    ax.set_xlim(-1, ANCHO + 1)
    ax.set_ylim(-1, LARGO + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)  ← cámara")
    return ax


if __name__ == "__main__":
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, ejes = plt.subplots(1, 2, figsize=(8, 8))

    for partido, ax in zip(["2022_BCN_FinalF_1", "2022_BCN_FinalM_1"], ejes):
        pts_px, pts_m = cargar_calibracion(partido)
        H = calcular_homografia(pts_px, pts_m)

        # Comprobación 1: los puntos marcados deben volver a su sitio
        reproy = px_a_metros(H, pts_px)
        print(f"\n{partido} | puntos usados: {len(pts_px)}")
        print("  error de ajuste máx (m):", round(float(np.max(np.linalg.norm(reproy - pts_m, axis=1))), 3))
        # Comprobación 2: validación leave-one-out
        print("  leave-one-out:", error_validacion(pts_px, pts_m))

        # Jugadores del frame de calibración (ground truth de Silver) proyectados a la pista
        df = pd.read_parquet(SILVER / f"{partido}_pose.parquet")
        frame0 = df[df["frame"] == df["frame"].min()]
        pos_m = px_a_metros(H, punto_suelo_px(frame0))
        print("  posiciones frame inicial (m):\n", np.round(pos_m, 2))

        dibujar_pista(ax)
        ax.scatter(pos_m[:, 0], pos_m[:, 1], s=80, c="yellow", edgecolors="black", zorder=3)
        ax.set_title(partido.replace("2022_BCN_", ""))

    plt.tight_layout()
    salida = FIGURAS / "homografia_frame_inicial.png"
    plt.savefig(salida, dpi=120)
    print("\nFigura guardada en", salida)