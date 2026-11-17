-- Esquema Gold. Modelo en estrella.
-- Dimensiones: dim_partido, dim_calibracion, dim_puesto
-- Hechos:      fact_posicion (una fila = un jugador en un frame según una fuente)
--              fact_metrica  (métrica táctica agregada)

CREATE TABLE dim_partido (
    match_id   VARCHAR PRIMARY KEY,
    categoria  VARCHAR,          -- femenino / masculino
    fuente     VARCHAR,          -- origen del vídeo
    licencia   VARCHAR,
    fps        DOUBLE,
    ancho_px   INTEGER,
    alto_px    INTEGER,
    n_frames   INTEGER
);

CREATE TABLE dim_calibracion (
    match_id          VARCHAR PRIMARY KEY REFERENCES dim_partido (match_id),
    frame_ref         INTEGER,   -- frame sobre el que se marcaron los puntos
    n_puntos          INTEGER,
    homografia        DOUBLE[],  -- matriz H 3x3 aplanada (9 valores)
    error_loo_medio_m DOUBLE,
    error_loo_max_m   DOUBLE,
    camara_x_m        DOUBLE,    -- posición estimada de la cámara
    camara_y_m        DOUBLE,
    camara_z_m        DOUBLE,
    focal_px          DOUBLE,
    error_reproy_px   DOUBLE,
    altura_tobillo_m  DOUBLE     -- altura usada en la corrección de paralaje
);

CREATE TABLE dim_puesto (
    puesto_id   VARCHAR PRIMARY KEY,  -- cerca_izq, cerca_der, lejos_izq, lejos_der
    lado        VARCHAR,              -- cerca / lejos (respecto a la cámara)
    posicion    VARCHAR,              -- izq / der
    descripcion VARCHAR
);

CREATE TABLE fact_posicion (
    match_id   VARCHAR REFERENCES dim_partido (match_id),
    frame      INTEGER,
    t_s        DOUBLE,
    puesto_id  VARCHAR REFERENCES dim_puesto (puesto_id),
    fuente     VARCHAR,   -- gt (ground truth) | baseline | pose
    u_px       DOUBLE,    -- punto de suelo en la imagen
    v_px       DOUBLE,
    x_m        DOUBLE,    -- posición en la pista
    y_m        DOUBLE,
    dist_red_m DOUBLE,
    zona       VARCHAR,   -- red | fondo
    confianza  DOUBLE,    -- NULL para el ground truth
    PRIMARY KEY (match_id, frame, puesto_id, fuente)
);

CREATE TABLE fact_metrica (
    match_id  VARCHAR REFERENCES dim_partido (match_id),
    lado      VARCHAR,
    fuente    VARCHAR,
    pct_red   DOUBLE,
    pct_fondo DOUBLE,
    n_frames  INTEGER,
    PRIMARY KEY (match_id, lado, fuente)
);