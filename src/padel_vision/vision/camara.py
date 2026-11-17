"""Modelo 3D de la cámara (modelo pinhole) estimado a partir de los puntos de la pista.

Sistema de coordenadas del mundo: x a lo ancho (0-10 m), y a lo largo (0-20 m),
z hacia arriba (el suelo es z = 0). Mismo origen que en pista.py.

Para qué sirve: la homografía solo es válida para puntos del suelo. El tobillo está a
unos centímetros de altura, con el modelo de cámara trazamos el rayo que pasa por el
tobillo y calculamos dónde corta el plano horizontal a esa altura.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Camara:
    K: np.ndarray       # matriz intrínseca 3x3 (focal y centro óptico, en píxeles)
    R: np.ndarray       # rotación mundo -> cámara (3x3)
    t: np.ndarray       # traslación mundo -> cámara (3x1)
    rms_px: float       # error de reproyección de los puntos de calibración (píxeles)

    @property
    def centro(self) -> np.ndarray:
        """Posición de la cámara en el mundo (x, y, z) en metros."""
        return (-self.R.T @ self.t).ravel()


def calibrar_camara(pts_px: np.ndarray, pts_m: np.ndarray,
                    ancho_px: int = 1920, alto_px: int = 1080) -> Camara:
    """Estima la cámara a partir de puntos de la pista (en el suelo, z = 0).

    Con una sola imagen y todos los puntos en el mismo plano no se pueden estimar todos
    los parámetros, así que fijamos lo razonable en cualquier cámara moderna: centro
    óptico en el centro de la imagen, píxeles cuadrados y sin distorsión de lente.
    Solo se estiman la distancia focal y la posición y orientación de la cámara.
    """
    obj = np.hstack([pts_m, np.zeros((len(pts_m), 1))]).astype(np.float32)
    K0 = np.array([[1500.0, 0, ancho_px / 2],
                   [0, 1500.0, alto_px / 2],
                   [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
             | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K, _, rvecs, tvecs = cv2.calibrateCamera(
        [obj], [pts_px.astype(np.float32)], (ancho_px, alto_px), K0, None, flags=flags)
    R, _ = cv2.Rodrigues(rvecs[0])
    return Camara(K=K, R=R, t=tvecs[0].reshape(3, 1), rms_px=float(rms))


def px_a_metros_altura(cam: Camara, pts_px: np.ndarray, altura_m: float = 0.0) -> np.ndarray:
    """Convierte puntos (N, 2) de la imagen a (N, 2) en la pista, suponiendo que
    están a `altura_m` metros sobre el suelo (0 = en el suelo, como la homografía)."""
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    homog = np.column_stack([pts, np.ones(len(pts))])            # (u, v, 1)
    rayos = (cam.R.T @ np.linalg.inv(cam.K) @ homog.T).T         # dirección de cada rayo en el mundo
    C = cam.centro
    s = (altura_m - C[2]) / rayos[:, 2]                          # cuánto avanzar hasta z = altura
    return (C + s[:, None] * rayos)[:, :2]