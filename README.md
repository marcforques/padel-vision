# Padel Vision 🎾

**Análisis táctico de pádel con visión artificial:** a partir del vídeo de un partido grabado con una cámara fija, el sistema calcula dónde está cada jugador en la pista (en metros) y genera métricas tácticas objetivas.

> Proyecto final del Máster en Data Science e IA (Evolve) · Marc Forqués Isasi

---

## El problema

En el pádel, el análisis de partidos y entrenamientos es visual y subjetivo, depende del ojo del entrenador. No hay herramientas accesibles para medir aspectos tácticos clave, como **cuánto tiempo controla una pareja la red** o si los dos jugadores **suben y bajan coordinados**. La analítica deportiva avanzada existe en deportes como el fútbol profesional, pero es cara y no está pensada para el pádel.

**Para quién:** entrenadores, clubes y academias, y jugadores de competición o aficionados avanzados.

## Qué hace

```
Vídeo del partido ──► Detección de jugadores ──► Posición en la pista (metros) ──► Métricas tácticas
                                                        ▲
                                     Calibración de la pista (4+ puntos marcados una vez)
```

1. **Calibración:** se marcan una vez puntos conocidos de la pista (esquinas, líneas de saque).
2. **Posición en metros:** el punto de apoyo de cada jugador (sus tobillos) se transforma a una vista cenital de la pista mediante una homografía y un modelo 3D de la cámara.
3. **Métrica táctica:** porcentaje del tiempo en la zona de red frente a la zona de fondo.

[Jugadores proyectados sobre la pista](docs/img/homografia_frame_inicial.png)

## Resultado destacado

Al analizar los datos se detectó un **sesgo sistemático de casi 1 m** entre el lado de la pista cercano y el lejano a la cámara: el tobillo no está en el suelo, y la proyección lo desplazaba. Se corrigió con un modelo 3D de la cámara y se validó usando la propia **simetría del juego** (las parejas se cambian de lado): la diferencia entre lados bajó de **0,81 m a 0,087 m (−89 %)**.

[Antes y después de la corrección](docs/img/simetria_antes_despues.png)

## Estado del proyecto

| Entrega | Contenido | Estado |
|---|---|---|
| 1 | [Ideas de producto](docs/entregas/01_ideas_producto.md) | ✅ |
| 2 | [Datos y preparación](docs/entregas/02_datos_y_preparacion.md) (Bronze → Silver → Gold) | ✅ |
| 3 | Machine Learning: detección de jugadores en vídeo y comparación con el *ground truth* | 🔄 En curso |
| 4 | Diseño del frontal y experiencia de usuario | ⏳ |
| Final | MVP funcional (9 de diciembre de 2026) | ⏳ |

## Estructura del repositorio

```
padel-vision/
├── data/
│   ├── bronze/        # datos originales (no se suben: tamaño y derechos)
│   ├── silver/        # datos limpios (no se suben)
│   ├── calibracion/   # puntos de la pista marcados a mano
│   └── sample/        # muestra pequeña de cada capa
├── db/schema.sql      # esquema de la base de datos Gold (DuckDB, modelo en estrella)
├── docs/
│   ├── entregas/      # documentación de cada entrega
│   └── img/           # figuras
├── notebooks/         # exploración
└── src/padel_vision/
    ├── pipeline/      # silver.py, gold.py
    ├── vision/        # marcar_pista.py, pista.py (homografía), camara.py (modelo 3D)
    └── analisis/      # simetria.py (validación de la corrección y umbral de red)
```

## Cómo ejecutarlo

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.11+. Se recomienda una GPU NVIDIA para la fase de detección.

```bash
# 1. Instalar dependencias
uv sync

# 2. Descargar los datos de PadelTracker100 desde Zenodo
#    https://doi.org/10.5281/zenodo.14653706
#    y descomprimir padel-data-labels.zip en data/bronze/padeltracker100/

# 3. Ejecutar el pipeline de datos
uv run python src/padel_vision/pipeline/silver.py
uv run python -m padel_vision.analisis.simetria
uv run python -m padel_vision.pipeline.gold
```

El resultado es la base de datos `data/gold/padel.duckdb` con las posiciones de los jugadores y las métricas.

## Datos y privacidad

- **Datos de desarrollo:** [PadelTracker100](https://doi.org/10.5281/zenodo.14653706) (Bada-Nerín et al., 2026), licencia CC BY 4.0. Dos finales del World Padel Tour (Barcelona, 2022) con la pose de los jugadores anotada.
- **Minimización:** no se guardan imágenes ni nombres, solo coordenadas. Los jugadores se identifican por su puesto en la pista.
- Los vídeos no se incluyen en el repositorio.

## Tecnologías

Python · OpenCV · Ultralytics YOLO · PyTorch · pandas · DuckDB · Matplotlib