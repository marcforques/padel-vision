"""Herramienta para marcar puntos de referencia de la pista sobre un frame.
"""
import json
import sys
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[3]
VIDEOS = RAIZ / "data/bronze/padeltracker100"
SALIDA = RAIZ / "data/calibracion"
VENTANA = "Marcar pista"

# (nombre, x_m, y_m). Origen: esquina izquierda del fondo cercano a la cámara.
PUNTOS = [
    ("fondo_cerca_izq", 0.0, 0.0),
    ("fondo_cerca_der", 10.0, 0.0),
    ("saque_cerca_izq", 0.0, 3.05),
    ("saque_cerca_T", 5.0, 3.05),
    ("saque_cerca_der", 10.0, 3.05),
    ("saque_lejos_izq", 0.0, 16.95),
    ("saque_lejos_T", 5.0, 16.95),
    ("saque_lejos_der", 10.0, 16.95),
    ("fondo_lejos_izq", 0.0, 20.0),
    ("fondo_lejos_der", 10.0, 20.0),
]


def main(partido: str, frame: int = 0) -> None:
    cap = cv2.VideoCapture(str(VIDEOS / f"{partido}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"No se pudo leer el frame {frame} de {partido}")

    marcados: list = []  # (u, v) en píxeles, o None si se salta

    def dibujar():
        vis = img.copy()
        for (nombre, _, _), p in zip(PUNTOS, marcados):
            if p:
                cv2.circle(vis, p, 5, (0, 0, 255), -1)
                cv2.putText(vis, nombre, (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(marcados) < len(PUNTOS):
            txt = f"Marca: {PUNTOS[len(marcados)][0]}   (s=saltar, z=deshacer, q=guardar)"
        else:
            txt = "Completado. Pulsa q para guardar"
        cv2.putText(vis, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(VENTANA, vis)

    def clic(evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN and len(marcados) < len(PUNTOS):
            marcados.append((x, y))
            dibujar()

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(VENTANA, clic)
    dibujar()
    while True:
        tecla = cv2.waitKey(50) & 0xFF
        if tecla == ord("s") and len(marcados) < len(PUNTOS):
            marcados.append(None)
            dibujar()
        elif tecla == ord("z") and marcados:
            marcados.pop()
            dibujar()
        elif tecla == ord("q"):
            break
    cv2.destroyAllWindows()

    datos = {
        "match_id": partido,
        "frame": frame,
        "sistema": "origen en esquina izquierda del fondo cercano; x a lo ancho (0-10 m), y a lo largo (0-20 m)",
        "puntos": [
            {"nombre": n, "x_m": xm, "y_m": ym,
             "u_px": p[0] if p else None, "v_px": p[1] if p else None}
            for (n, xm, ym), p in zip(PUNTOS, marcados)
        ],
    }
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = SALIDA / f"{partido}.json"
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Guardado en", ruta)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)