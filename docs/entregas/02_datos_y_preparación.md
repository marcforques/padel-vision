# Entrega 02 — Datos y preparación

**Proyecto:** Sistema de visión artificial para análisis táctico y posicionamiento en pádel
**Objetivo de esta entrega:** demostrar que el producto puede construirse con datos reales y dejar preparado un conjunto de datos limpio y trazable (Bronze → Silver → Gold) para la fase de modelado.

---

## 1. Qué datos necesita el producto

El producto convierte el vídeo de un partido, grabado con una cámara fija, en posiciones de los jugadores sobre la pista (en metros) y en métricas tácticas. La primera métrica del MVP es el **porcentaje de tiempo que cada pareja pasa en la zona de red frente a la zona de fondo**.

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
| **Vídeo del usuario o del club** | Fuente real del producto en producción | El usuario sube el archivo de vídeo | Cualquier partido grabado con cámara fija | Depende del usuario; formato estándar (MP4) |
| **PadelTracker100** | Desarrollo y validación (*ground truth*) | Descarga directa desde Zenodo (https://doi.org/10.5281/zenodo.14653706) | 2 partidos completos (finales femenina y masculina del WPT Barcelona 2022), 1920×1080 a 30 fps, unos 100.000 frames | Alta: repositorio científico con DOI permanente; dataset estático |
| **Vídeos amateur propios** | Prueba de robustez en condiciones reales | Partidos grabados por un familiar, cedidos con consentimiento | En curso | Datos propios |

**Por qué PadelTracker100 y no otra fuente:** es el único dataset público de pádel con anotaciones manuales de pose (17 puntos del cuerpo por jugador y frame) a esta escala. Las anotaciones se generaron de forma semiautomática y después se **refinaron a mano**, lo que permite usarlas como referencia para medir el error de nuestro sistema.

**Importante:** PadelTracker100 **no es la fuente del producto**, sino la herramienta para validarlo. En producción, el dato es el propio vídeo que aporta el usuario o el club, y es renovable: cada partido nuevo es un dato nuevo.

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
| En el partido masculino faltan **65 frames** sin anotar (a partir del frame 325). El identificador interno de imagen (`image_id`) queda desalineado con el vídeo | El número de frame se obtiene del **nombre del fichero** (`frame_000123.PNG`), nunca del `image_id` |
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

- **Punto de apoyo de cada jugador:** punto medio de los dos tobillos. La homografía solo es válida para puntos del suelo; el centro del cuerpo está a casi un metro de altura y se proyectaría en una posición errónea.
- **Validación (*leave-one-out*):** para cada punto de calibración se calcula la homografía sin él y se mide el error al proyectarlo.

| | Error medio | Error máximo |
|---|---|---|
| Femenino | 0,145 m | 0,289 m |
| Masculino | 0,096 m | 0,208 m |

Los errores mayores se concentran en las esquinas de la pista, más difíciles de marcar y más afectadas por la distorsión de la lente.

[Jugadores del primer frame proyectados sobre la pista](../img/homografia_frame_inicial.png)

### Gold — dato modelado para su uso

Script: `src/padel_vision/pipeline/gold.py` → base de datos DuckDB `data/gold/padel.duckdb` (se genera con el script, en el repo está el esquema `db/schema.sql`).

**Modelo en estrella:**

| Tabla | Tipo | Contenido |
|---|---|---|
| `dim_partido` | Dimensión | Partido, categoría, fuente, licencia, fps, resolución |
| `dim_calibracion` | Dimensión | Matriz de homografía y su error de validación (trazabilidad de cada posición) |
| `dim_puesto` | Dimensión | Puestos en la pista: lado (cerca/lejos de la cámara) y posición (izquierda/derecha) |
| `fact_posicion` | Hechos | **Una fila = un jugador en un frame según una fuente**: posición en píxeles, en metros, distancia a la red y zona (red/fondo) |
| `fact_metrica` | Hechos | Porcentaje de tiempo en red y en fondo por partido, lado y fuente |

**Clave del diseño:** la columna `fuente` de `fact_posicion`. Ahora contiene el *ground truth* (`gt`). En la siguiente fase, los modelos escribirán sus predicciones en la misma tabla (`baseline`, `pose`), lo que permite compararlos con el *ground truth* mediante una simple consulta. Los resultados de los modelos vuelven así a Gold, y el frontal solo leerá de esta capa.

Transformaciones de Silver a Gold:
1. Proyección del punto de apoyo a metros con la homografía.
2. **Eliminación de personas fuera de la pista** (más de 1 m fuera de las líneas): recogepelotas, árbitros y jugadores que salen por las puertas laterales.
3. **Asignación de puestos:** se conservan los frames con exactamente 2 jugadores a cada lado de la red. Los cambios de lado (jugadores cruzando la red) no son juego y se excluyen.
4. **Zona:** red si el jugador está a menos de 4 m de la red, fondo en caso contrario. *Valor provisional, elegido por criterio de juego. Se revisará con la distribución real de posiciones (ver Riesgos y limitaciones).*

| | Femenino | Masculino |
|---|---|---|
| Personas descartadas fuera de pista | 117 | 381 |
| Frames válidos (2 + 2 jugadores) | 45.703 (99,5 %) | 52.592 (97,6 %) |

### Cómo reproducirlo

```bash
uv sync
# 1. Descargar padel-data-labels.zip de Zenodo y descomprimirlo en data/bronze/padeltracker100/
# 2. (Opcional) Volver a marcar la calibración:
uv run python src/padel_vision/vision/marcar_pista.py 2022_BCN_FinalF_1
# 3. Pipeline
uv run python src/padel_vision/pipeline/silver.py
uv run python -m padel_vision.pipeline.gold
```

Muestras de cada capa en `data/sample/`.

---

## 5. Riesgos y limitaciones

- **Sesgo sistemático entre lados de la pista (detectado, en corrección).** Como las parejas se cambian de lado durante el partido, ambos lados deberían tener una distribución de posiciones parecida. Sin embargo, el lado lejano aparece desplazado casi 1 m respecto al cercano, y con el umbral de 4 m el porcentaje de tiempo en red sale del 30 % en el lado cercano frente al 13 % en el lejano (femenino). La causa principal es que el tobillo no está en el suelo, sino a unos 10 cm de altura, y la homografía lo desplaza a lo largo del rayo de la cámara. En la siguiente fase se corregirá con un modelo 3D de la cámara y se validará comprobando que ambos lados se igualan. El umbral de red también se revisará con los datos.

[Distancia a la red por lado de la pista](../img/distancia_red_por_lado.png)

- **Lado frente a pareja:** la métrica actual se calcula por lado de la pista, no por pareja. Para atribuirla a cada pareja se identificará su equipación por color (agrupación K-Means sobre el torso de cada jugador).
- **Poca variedad en el *ground truth*:** solo hay 2 partidos profesionales, en la misma pista y con la misma cámara. La robustez en vídeo amateur se evaluará con vídeos propios.
- **Calibración manual:** cada vídeo requiere marcar a mano los puntos de la pista (unos 2 minutos). Automatizarlo es una mejora futura.
- **Si PadelTracker100 dejara de estar disponible:** se usaría PadelVic o una muestra anotada manualmente de vídeos propios.