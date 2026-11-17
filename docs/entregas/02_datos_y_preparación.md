# Entrega 02 — Datos y preparación

**Proyecto:** Sistema de visión artificial para análisis táctico y posicionamiento en pádel
**Objetivo de esta entrega:** demostrar que el producto puede construirse con datos reales y dejar preparado un conjunto de datos limpio y trazable (Bronze → Silver → Gold) para la fase de modelado.

---

## 1. Qué datos necesita el producto

El producto convierte el vídeo de un partido, grabado con una cámara fija, en posiciones de los jugadores sobre la pista (en metros) y en métricas tácticas. La primera métrica del MVP es el **porcentaje de tiempo que cada pareja pasa en la zona de red frente a la zona de fondo** (en esta fase se calcula por lado de la pista. Ver Riesgos y limitaciones).

**Qué representa una observación:** un jugador en un frame del vídeo (frame × jugador). A 30 fps, un partido de una hora genera unas 430.000 observaciones (4 jugadores × 108.000 frames).

| Tipo | Dato | Para qué |
|---|---|---|
| Imprescindible | Vídeo de cámara fija con la pista completa a la vista | Es la entrada del producto |
| Imprescindible | Puntos de referencia de la pista (mínimo 4) marcados una vez por vídeo | Calcular la transformación píxeles → metros (homografía) |
| Imprescindible (desarrollo) | Posiciones de los jugadores anotadas por personas (*ground truth*) | Validar que el sistema mide bien |
| Deseable | Vídeos de distintas condiciones (amateur, otras cámaras, otra iluminación) | Comprobar la robustez fuera de condiciones ideales |
| Deseable | Identidad de cada pareja a lo largo del partido | Atribuir la métrica a cada pareja aunque cambien de lado |

**Granularidad e histórico:** la unidad mínima es el frame (1/30 s). No se necesita histórico: cada partido se analiza de forma independiente. Para el desarrollo basta con partidos anotados que permitan validar la medida.

---

## 2. Fuentes reales y viabilidad

| Fuente | Rol | Acceso | Cobertura | Estabilidad |
|---|---|---|---|---|
| **Vídeo del usuario o del club** | Fuente real del producto en producción | El usuario sube el archivo de vídeo | Cualquier partido grabado con cámara fija | Depende del usuario, formato estándar (MP4) |
| **PadelTracker100** | Desarrollo y validación (*ground truth*) | Descarga directa desde Zenodo (https://doi.org/10.5281/zenodo.14653706)) | 2 partidos (finales femenina y masculina del WPT Barcelona 2022): unos 25 y 30 minutos de vídeo editado, 1920×1080 a 30 fps, unos 100.000 frames | Alta: repositorio científico con DOI permanente; dataset estático |
| **Vídeos amateur propios** | Prueba de robustez en condiciones reales | Partidos grabados por un familiar, cedidos con consentimiento | En curso | Datos propios |

**Por qué PadelTracker100 y no otra fuente:** es el único dataset público de pádel con anotaciones manuales de pose (17 puntos del cuerpo por jugador y frame) a esta escala. Las anotaciones se generaron de forma semiautomática y después se **refinaron a mano**, lo que permite usarlas como referencia para medir el error de nuestro sistema.

**Importante:** PadelTracker100 **no es la fuente del producto**, sino la herramienta para validarlo. En producción, el dato es el propio vídeo que aporta el usuario o el club, y es renovable, cada partido nuevo es un dato nuevo.

**Plan B:** si PadelTracker100 dejara de estar disponible, se usaría el dataset PadelVic (partido amateur con posiciones estimadas) o se anotaría manualmente una muestra de frames de vídeos propios (150–300 frames bastan para validar).

---

## 3. Uso, privacidad y datos personales

**Licencias:**
- **PadelTracker100:** licencia CC BY 4.0, que permite el uso y la redistribución citando a los autores (Bada-Nerín et al., *Data in Brief*, 2026).
- **Imágenes de la retransmisión (WPT):** pueden estar sujetas a derechos de la retransmisión. Por precaución, **los vídeos no se suben al repositorio**; se indican las instrucciones para descargarlos de la fuente original.
- **Vídeos de plataformas como YouTube:** si se usan, solo para pruebas internas, sin redistribuirlos.

**Datos personales:** los vídeos muestran a personas identificables. Se aplica **minimización**:
- **No se guardan imágenes** en las capas Silver y Gold: solo coordenadas numéricas (posición del cuerpo en píxeles y posición en la pista en metros).
- **No se guardan nombres.** Cada jugador se identifica por su **puesto en la pista** (`cerca_izq`, `cerca_der`, `lejos_izq`, `lejos_der`), no por su identidad.
- **Vídeos amateur:** solo se usan con el **consentimiento por escrito** de los cuatro jugadores, nunca se suben al repositorio y de ellos solo se conservan las coordenadas calculadas.

---

## 4. Del dato bruto al dato utilizable (Bronze → Silver → Gold)

```
Vídeo + anotaciones (Bronze) ──► Tablas limpias (Silver) ──► Posiciones en metros en DuckDB (Gold)
                                                         ▲
                                     Puntos de la pista marcados a mano (calibración)
```

### Bronze — dato tal como llega

`data/bronze/padeltracker100/` (no se sube al repo por tamaño y derechos)

| Fichero | Contenido |
|---|---|
| `2022_BCN_FinalF_1.mp4`, `2022_BCN_FinalM_1.mp4` | Vídeos originales (2,9 GB y 4,3 GB) |
| `labels/*_pose.json` | Pose de los jugadores en formato COCO: caja y 17 puntos del cuerpo por persona y frame |
| `labels/*_ball.json` | Posición de la bola (no se usa en el MVP) |
| `labels/*_shots.csv` | Golpes por frame (no se usa en el MVP) |

Bronze no se modifica nunca: es la garantía de trazabilidad y reproducibilidad.

**Calibración** (`data/calibracion/*.json`, sí se sube al repo): 10 puntos de referencia de la pista (esquinas y líneas de saque) marcados a mano por partido con `src/padel_vision/vision/marcar_pista.py`. Es un dato de entrada propio: se genera una vez y se versiona.

### Silver — dato limpio y consistente

Script: `src/padel_vision/pipeline/silver.py` → `data/silver/<partido>_pose.parquet`
**Una fila = una persona anotada en un frame**, con su caja y sus 17 puntos del cuerpo.

Problemas encontrados en los datos y decisión tomada:

| Problema | Decisión |
|---|---|
| En el partido masculino faltan **65 frames** sin anotar (a partir del frame 325), el identificador interno de imagen (`image_id`) queda desalineado con el vídeo | El número de frame se obtiene del **nombre del fichero** (`frame_000123.PNG`), nunca del `image_id` |
| El **orden de los 17 puntos del cuerpo es distinto** entre los dos partidos (izquierda y derecha intercambiadas) | Los puntos se normalizan **por nombre** a un orden canónico |
| Puntos del cuerpo no anotados | Se convierten en valores nulos (NaN) |
| Anotaciones **duplicadas** (dos cajas casi idénticas para la misma persona) | Se eliminan si se solapan más de un 70 % (IoU > 0,7) |
| Frames con menos de 4 personas (jugador tapado o fuera de plano) | Se conservan y se marcan, no se inventan datos |

Resultado:

| | Femenino | Masculino |
|---|---|---|
| Anotaciones iniciales | 183.744 | 215.173 |
| Duplicados eliminados | 2 | 1 |
| Frames | 45.934 | 53.888 |
| Frames con menos de 4 personas | 9 | 375 |
| Frames con más de 4 personas (persona extra real) | 15 | 3 |
| Anotaciones con ambos tobillos visibles | 99,74 % | 99,60 % |

### Transformación a metros (homografía)

Script: `src/padel_vision/vision/pista.py`

Para pasar de píxeles a metros se calcula una **homografía**: una transformación que lleva los puntos del suelo en la imagen a sus posiciones en una vista cenital de la pista (20 × 10 m). Se calcula a partir de los 10 puntos de calibración.

- **Punto de apoyo de cada jugador:** punto medio de los dos tobillos. La homografía solo es válida para puntos del suelo. El centro del cuerpo está a casi un metro de altura y se proyectaría en una posición errónea.
- **Validación (*leave-one-out*):** para cada punto de calibración se calcula la homografía sin él y se mide el error al proyectarlo.

| | Error medio | Error máximo |
|---|---|---|
| Femenino | 0,145 m | 0,289 m |
| Masculino | 0,096 m | 0,208 m |

Los errores mayores se concentran en las esquinas de la pista, más difíciles de marcar y más afectadas por la distorsión de la lente.

[Jugadores del primer frame proyectados sobre la pista](../img/homografia_frame_inicial.png)

### Corrección de la altura del tobillo (paralaje)

Módulo: `src/padel_vision/vision/camara.py` · Análisis: `src/padel_vision/analisis/simetria.py`

**Problema detectado.** Al analizar las posiciones, el lado de la pista lejano a la cámara aparecía desplazado casi 1 m respecto al cercano. Como las parejas se cambian de lado durante el partido, ambos lados deberían tener una distribución de posiciones parecida. La diferencia era un error sistemático de la medida, no táctica.

**Causa.** El tobillo no está en el suelo. La homografía lo proyecta a lo largo del rayo de la cámara hasta el suelo y lo sitúa más lejos de la cámara de lo que está. El efecto crece con la distancia, por eso era mayor en el campo lejano.

**Solución.** Se estima un **modelo 3D de la cámara** a partir de los mismos puntos de calibración (focal, posición y orientación). Para cada jugador se calcula dónde corta el rayo de la cámara el plano a la altura del tobillo, y ese desplazamiento se suma a la posición de la homografía. Así se combina la precisión de la homografía en el suelo con la corrección por altura que solo puede dar el modelo 3D.

- **Cámara estimada:** a 16,7 m por detrás del fondo cercano, centrada y a 7,9 m de altura. Calibrando cada partido por separado, la posición coincide con 2 cm de diferencia, lo que confirma que la calibración es consistente.

**Ajuste y validación por simetría.** La altura del tobillo se elige como la que hace más parecidas las distribuciones de ambos lados (distancia de Wasserstein), probando de 0 a 20 cm:

| Altura del tobillo | Diferencia entre lados |
|---|---|
| 0 m (sin corrección) | 0,81 m |
| **0,12 m (óptima)** | **0,087 m (−89 %)** |

La altura óptima es la misma en los dos partidos (11–13 cm) y es físicamente coherente: el punto anotado como tobillo es el maléolo, a unos 7–9 cm del pie, más unos 3 cm de suela de la zapatilla.

[Distancia a la red por lado, antes y después de la corrección](../img/simetria_antes_despues.png)

### Gold — dato modelado para su uso

Script: `src/padel_vision/pipeline/gold.py` → base de datos DuckDB `data/gold/padel.duckdb` (se genera con el script, en el repo está el esquema `db/schema.sql`).

**Modelo en estrella:**

| Tabla | Tipo | Contenido |
|---|---|---|
| `dim_partido` | Dimensión | Partido, categoría, fuente, licencia, fps, resolución |
| `dim_calibracion` | Dimensión | Homografía, modelo de cámara (posición, focal, error) y altura del tobillo usada: trazabilidad de cada posición |
| `dim_puesto` | Dimensión | Puestos en la pista: lado (cerca/lejos de la cámara) y posición (izquierda/derecha) |
| `fact_posicion` | Hechos | **Una fila = un jugador en un frame según una fuente**: posición en píxeles, en metros, distancia a la red y zona (red/fondo) |
| `fact_metrica` | Hechos | Porcentaje de tiempo en red y en fondo por partido, lado y fuente |

**Clave del diseño:** la columna `fuente` de `fact_posicion`. Ahora contiene el *ground truth* (`gt`); en la siguiente fase, los modelos escribirán sus predicciones en la misma tabla (`baseline`, `pose`), lo que permite compararlos con el *ground truth* mediante una simple consulta. Los resultados de los modelos vuelven así a Gold, y el frontal solo leerá de esta capa.

Transformaciones de Silver a Gold:
1. Proyección del punto de apoyo a metros: homografía + corrección de la altura del tobillo (0,12 m).
2. **Eliminación de personas fuera de la pista** (más de 1 m fuera de las líneas): recogepelotas, árbitros y jugadores que salen por las puertas laterales.
3. **Asignación de puestos:** se conservan los frames con exactamente 2 jugadores a cada lado de la red. Los cambios de lado (jugadores cruzando la red) no son juego y se excluyen.
4. **Zona:** red si el jugador está a menos de **5,5 m** de la red; fondo en caso contrario. El umbral se ha decidido con los datos: la distribución de distancias a la red tiene dos picos (jugadores en red, hacia 4 m; y al fondo, hacia 7,5–8 m), y el valle entre ambos está en 5,6–5,7 m en los dos partidos. Poner el umbral en el valle hace la métrica robusta, un pequeño error de posición apenas cambia la zona de nadie. Un umbral inicial de 4 m, elegido por intuición, cortaba el pico de red por la mitad.

| | Femenino | Masculino |
|---|---|---|
| Personas descartadas fuera de pista | 62 | 340 |
| Frames válidos (2 + 2 jugadores) | 45.766 (99,6 %) | 52.830 (98,0 %) |

**Métrica resultante** (`fact_metrica`, porcentaje del tiempo en la zona de red):

| | Lado cercano | Lado lejano |
|---|---|---|
| Femenino | 48,0 % | 47,0 % |
| Masculino | 53,1 % | 52,5 % |

Antes de la corrección, la diferencia entre lados era de 16–21 puntos Y ahora es de menos de 1 punto, como se espera cuando las parejas se reparten el tiempo en ambos lados.

### Cómo reproducirlo

```bash
uv sync
# 1. Descargar padel-data-labels.zip de Zenodo y descomprimirlo en data/bronze/padeltracker100/
# 2. (Opcional) Volver a marcar la calibración:
uv run python src/padel_vision/vision/marcar_pista.py 2022_BCN_FinalF_1
# 3. Pipeline
uv run python src/padel_vision/pipeline/silver.py
uv run python -m padel_vision.analisis.simetria   # ajuste de la altura del tobillo y del umbral
uv run python -m padel_vision.pipeline.gold
```

Muestras de cada capa en `data/sample/`:
 
| Capa | Fichero |
|---|---|
| Bronze | `bronze_2022_BCN_FinalF_1_pose_muestra.json` (primeros 50 frames del JSON original) |
| Silver | `silver_<partido>_pose.csv` (200 filas) |
| Gold | `gold_fact_posicion.csv` (200 filas) y `gold_fact_metrica.csv` |
 
---


## 5. Riesgos y limitaciones

- **Validación indirecta de la posición:** el *ground truth* contiene la pose en píxeles, no la posición en metros. La conversión a metros se valida con la calibración (*leave-one-out*, 10–15 cm) y con la simetría entre lados, pero no con una medición independiente en pista.
- **Distorsión de la lente:** no se modela. Es la causa más probable de que los mayores errores de calibración estén en las esquinas de la pista.
- **Lado frente a pareja:** la métrica actual se calcula por lado de la pista, no por pareja. Para atribuirla a cada pareja se identificará su equipación por color (agrupación K-Means sobre el torso de cada jugador).
- **Poca variedad en el *ground truth*:** solo hay 2 partidos profesionales, en la misma pista y con la misma cámara. La robustez en vídeo amateur se evaluará con vídeos propios.
- **Calibración manual:** cada vídeo requiere marcar a mano los puntos de la pista (unos 2 minutos). Automatizarlo es una mejora futura.
- **Si PadelTracker100 dejara de estar disponible:** se usaría PadelVic o una muestra anotada manualmente de vídeos propios.