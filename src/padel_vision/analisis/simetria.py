"""Ajuste y validación de la corrección del tobillo usando la simetría entre lados.

Las parejas se cambian de lado durante el partido, así que a lo largo de un partido
la distribución de la distancia a la red debería ser parecida en el lado cercano y en el
lejano. Si la medida tiene un sesgo que depende de la distancia a la cámara, los dos lados
se separan.

1. Probamos alturas del tobillo entre 0 y 20 cm y medimos cuánto se parecen los dos lados
   (distancia de Wasserstein: cuántos metros, de media, habría que mover una distribución
   para convertirla en la otra. 0 = idénticas).
2. Elegimos la altura que mejor iguala los lados y comprobamos que sea físicamente creíble.
3. Con la medida corregida, buscamos el valle entre el pico de red y el de fondo para
   proponer un umbral de red robusto.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from padel_vision.pipeline.gold import PARTIDOS, SILVER, asignar_puestos, filtrar_pista, proyectar
from padel_vision.vision.camara import calibrar_camara
from padel_vision.vision.pista import RED_Y, calcular_homografia, cargar_calibracion

RAIZ = Path(__file__).resolve().parents[3]
FIGURAS = RAIZ / "docs/img"
ALTURAS = np.round(np.arange(0.0, 0.205, 0.01), 2)
PASO_FRAMES = 2  # usamos 1 de cada 2 frames para ir más rápido (frames consecutivos son casi iguales)


def distancias(silver: pd.DataFrame, H, cam, altura_m: float) -> pd.DataFrame:
    df = asignar_puestos(filtrar_pista(proyectar(silver, H, cam, altura_m)))
    df["dist_red_m"] = (df["y_m"] - RED_Y).abs()
    return df[["frame", "lado", "dist_red_m"]]


def valle(dist: pd.Series) -> float:
    """Punto más bajo del histograma (suavizado) ENTRE el pico de red y el de fondo."""
    cuentas, bordes = np.histogram(dist, bins=np.arange(0, 11.05, 0.1))
    suave = np.convolve(cuentas, np.ones(5) / 5, mode="same")
    centros = (bordes[:-1] + bordes[1:]) / 2
    zona_red = (centros >= 2) & (centros <= 5.5)
    zona_fondo = (centros >= 6) & (centros <= 10)
    pico_red = centros[zona_red][np.argmax(suave[zona_red])]
    pico_fondo = centros[zona_fondo][np.argmax(suave[zona_fondo])]
    entre = (centros > pico_red) & (centros < pico_fondo)
    return float(centros[entre][np.argmin(suave[entre])])


def main():
    datos = {}
    for partido in PARTIDOS:
        pts_px, pts_m = cargar_calibracion(partido)
        silver = pd.read_parquet(SILVER / f"{partido}_pose.parquet")
        silver = silver[silver["frame"] % PASO_FRAMES == 0]
        datos[partido] = (silver, calcular_homografia(pts_px, pts_m), calibrar_camara(pts_px, pts_m))

    # 1. Barrido de alturas
    filas = []
    for h in ALTURAS:
        for partido, (silver, H, cam) in datos.items():
            d = distancias(silver, H, cam, h)
            cerca = d.loc[d["lado"] == "cerca", "dist_red_m"]
            lejos = d.loc[d["lado"] == "lejos", "dist_red_m"]
            filas.append({
                "altura_m": h, "match_id": partido,
                "wasserstein_m": wasserstein_distance(cerca, lejos),
                "mediana_cerca_m": cerca.median(), "mediana_lejos_m": lejos.median(),
            })
    res = pd.DataFrame(filas)
    tabla = res.pivot(index="altura_m", columns="match_id", values="wasserstein_m")
    tabla["media"] = tabla.mean(axis=1)
    print("Diferencia entre lados (Wasserstein, m) según la altura del tobillo:")
    print(tabla.round(3).to_string())

    h_opt = float(tabla["media"].idxmin())
    print(f"\nAltura que mejor iguala los lados: {h_opt:.2f} m")
    print(f"Diferencia sin corrección (0 m): {tabla.loc[0.0, 'media']:.3f} m"
          f" -> con corrección: {tabla.loc[h_opt, 'media']:.3f} m")
    if not 0.05 <= h_opt <= 0.15:
        print("AVISO: la altura óptima no es físicamente creíble para un tobillo (5-15 cm).")
        print("       La corrección no explica todo el sesgo: hay otras causas (ver documentación).")

    # 2. Figura antes / después y valle para el umbral
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, ejes = plt.subplots(2, len(datos), figsize=(12, 7), sharex=True, sharey=True)
    umbrales = []
    for col, (partido, (silver, H, cam)) in enumerate(datos.items()):
        for fila, (h, titulo) in enumerate([(0.0, "sin corrección"), (h_opt, f"corregido (h = {h_opt:.2f} m)")]):
            d = distancias(silver, H, cam, h)
            ax = ejes[fila, col]
            for lado, color in [("cerca", "#1f77b4"), ("lejos", "#ff7f0e")]:
                ax.hist(d.loc[d["lado"] == lado, "dist_red_m"], bins=88, range=(0, 11),
                        density=True, alpha=0.5, color=color, label=lado)
            if fila == 1:
                u = valle(d["dist_red_m"])
                umbrales.append(u)
                ax.axvline(u, color="black", ls="--", label=f"valle {u:.1f} m")
            ax.set_title(f"{partido.replace('2022_BCN_', '')} — {titulo}")
            ax.legend()
        ejes[1, col].set_xlabel("distancia a la red (m)")
    plt.tight_layout()
    salida = FIGURAS / "simetria_antes_despues.png"
    plt.savefig(salida, dpi=120)
    print(f"\nValle entre red y fondo (umbral sugerido): {', '.join(f'{u:.1f} m' for u in umbrales)}")
    print("Figura guardada en", salida)


if __name__ == "__main__":
    main()