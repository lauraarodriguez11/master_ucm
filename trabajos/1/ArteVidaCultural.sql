/* -------------------------------------------------------------------------------------------
Laura Rodríguez Ropero
---------------------------------------------------------------------------------------------------*/
/* -------------------------------------------------------------------------------------------
Creación de la Base de Datos ArteVidaCultural
---------------------------------------------------------------------------------------------------*/
DROP DATABASE IF EXISTS ArteVidaCultural;
CREATE DATABASE IF NOT EXISTS ArteVidaCultural;
USE ArteVidaCultural;

/* -------------------------------------------------------------------------------------------
Definición de la estructura de la Base de Datos
Creación de las tablas
---------------------------------------------------------------------------------------------------*/
/*DROP TABLE IF EXISTS Evento;
DROP TABLE IF EXISTS Ubicacion;
DROP TABLE IF EXISTS Actividad;
DROP TABLE IF EXISTS Artista;
DROP TABLE IF EXISTS Caracteristica;
DROP TABLE IF EXISTS Asistente;
DROP TABLE IF EXISTS Entrada;
DROP TABLE IF EXISTS participe;
DROP TABLE IF EXISTS valore;
DROP TABLE IF EXISTS valga;
*/

-- Tabla Ubicacion
CREATE TABLE Ubicacion (
    id_ubicacion INT AUTO_INCREMENT PRIMARY KEY,
    n_ubicacion VARCHAR(100) NOT NULL,
    direccion VARCHAR(255),
    ciudad VARCHAR(100),
    aforo INT NOT NULL,
    precio_alquiler DECIMAL(10, 2),
    caracteristicas VARCHAR(255)
);

-- Tabla Actividad
CREATE TABLE Actividad (
    id_actividad INT AUTO_INCREMENT PRIMARY KEY,
    n_actividad VARCHAR(100) NOT NULL,
    tipo VARCHAR(50), 
    coste DECIMAL(10, 2)
);

-- Tabla Evento
CREATE TABLE Evento (
    id_evento INT AUTO_INCREMENT PRIMARY KEY,
    n_evento VARCHAR(100) NOT NULL,
    fecha_hora DATETIME NOT NULL,
    descripcion TEXT,
    id_actividad INT,
    id_ubicacion INT,
    UNIQUE (id_actividad, id_ubicacion),
    FOREIGN KEY (id_actividad) REFERENCES Actividad(id_actividad),
    FOREIGN KEY (id_ubicacion) REFERENCES Ubicacion(id_ubicacion)
);

-- Tabla Artista
CREATE TABLE Artista (
    id_artista INT AUTO_INCREMENT PRIMARY KEY,
    n_artista VARCHAR(150) NOT NULL,
    biografia TEXT
);

-- Tabla Asistente
CREATE TABLE Asistente (
    id_asistente INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    ape1 VARCHAR(50) NOT NULL,
    ape2 VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL
);

-- Tabla Telefono
CREATE TABLE Telefono (
    telefono NUMERIC(9,0) NOT NULL PRIMARY KEY,
    id_asistente INT NOT NULL,
    FOREIGN KEY (id_asistente) REFERENCES Asistente(id_asistente)
);

-- Tabla Categoria
CREATE TABLE Categoria (
    id_categoria INT AUTO_INCREMENT PRIMARY KEY,
    n_categoria VARCHAR(50) NOT NULL
);

-- Tabla intermedia para la relación Actividad-Artista (participe) (N:N)
CREATE TABLE Participe (
    id_actividad INT,
    id_artista INT,
    cache DECIMAL(10, 2),
    PRIMARY KEY (id_actividad, id_artista),
    UNIQUE (id_actividad, id_artista),
    FOREIGN KEY (id_actividad) REFERENCES Actividad(id_actividad),
    FOREIGN KEY (id_artista) REFERENCES Artista(id_artista)
);

-- Tabla intermedia para la relación Evento-Asistente (valore) (N:N) con valoración
CREATE TABLE Valore (
    id_evento INT,
    id_asistente INT,
    valoracion SMALLINT CHECK (valoracion >= 0 AND valoracion <= 5),
    PRIMARY KEY (id_evento, id_asistente),
    UNIQUE (id_evento, id_asistente),
    FOREIGN KEY (id_evento) REFERENCES Evento(id_evento),
    FOREIGN KEY (id_asistente) REFERENCES Asistente(id_asistente)
);

-- Tabla intermedia para la relación Evento-Categoria (valga) (N:N) con precio
CREATE TABLE Valga (
    id_evento INT,
    id_categoria INT,
    precio DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (id_evento, id_categoria),
	UNIQUE (id_evento, id_categoria),
    FOREIGN KEY (id_evento) REFERENCES Evento(id_evento),
    FOREIGN KEY (id_categoria) REFERENCES Categoria(id_categoria)
);

-- Tabla para la entidad Entrada (Evento-Asistente-Categoria)
CREATE TABLE Entrada (
	id_entrada INT AUTO_INCREMENT PRIMARY KEY,
    id_evento INT NOT NULL,
    id_asistente INT NOT NULL,
    id_categoria INT NOT NULL,
    FOREIGN KEY (id_evento) REFERENCES Evento(id_evento),
    FOREIGN KEY (id_asistente) REFERENCES Asistente(id_asistente),
    FOREIGN KEY (id_categoria) REFERENCES Categoria(id_categoria)
);

/* -------------------------------------------------------------------------------------------
Creación de 3 triggers que se encarguen de insertar, actualizar y eliminar
los datos de el atributo de actividad, coste, cuando se inserten, actualizen 
o eliminen datos en la tabla participe.
---------------------------------------------------------------------------------------------------*/

-- Trigger para inserciones en Participe
DELIMITER //
CREATE TRIGGER actualiza_coste_actividad_insert
AFTER INSERT ON Participe
FOR EACH ROW
BEGIN
    UPDATE Actividad
    SET coste = coste + NEW.cache
    WHERE id_actividad = NEW.id_actividad;
END//
DELIMITER ;

-- Trigger para actualizaciones en Participe
DELIMITER //
CREATE TRIGGER actualiza_coste_actividad_update
AFTER UPDATE ON Participe
FOR EACH ROW
BEGIN
    UPDATE Actividad
    SET coste = coste - OLD.cache + NEW.cache
    WHERE id_actividad = NEW.id_actividad;
END//
DELIMITER ;

-- Trigger para eliminaciones en Participe
DELIMITER //
CREATE TRIGGER actualiza_coste_actividad_delete
AFTER DELETE ON Participe
FOR EACH ROW
BEGIN
    UPDATE Actividad
    SET coste = coste - OLD.cache
    WHERE id_actividad = OLD.id_actividad;
END//
DELIMITER ;


/* -------------------------------------------------------------------------------------------
Inserción de Datos
---------------------------------------------------------------------------------------------------*/SHOW VARIABLES LIKE 'secure_file_priv';

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Ubicacion.csv' 
INTO TABLE Ubicacion
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ';'          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(n_ubicacion,direccion,ciudad,aforo,precio_alquiler,caracteristicas);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Actividad.csv' 
INTO TABLE Actividad 
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(n_actividad,tipo);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Evento.csv' 
INTO TABLE Evento
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(n_evento,fecha_hora,descripcion,id_actividad,id_ubicacion);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Artista.csv' 
INTO TABLE Artista
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(n_artista,biografia);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Asistente.csv' 
INTO TABLE Asistente
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(nombre,ape1,ape2,email);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Telefono.csv' 
INTO TABLE Telefono
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(telefono,id_asistente);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Categoria.csv' 
INTO TABLE Categoria
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(n_categoria);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\participe.csv' 
INTO TABLE participe
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(id_actividad,id_artista,cache);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\valore.csv' 
INTO TABLE valore
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(id_evento,id_asistente,valoracion);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\valga.csv' 
INTO TABLE valga
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(id_evento,id_categoria,precio);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Entrada.csv' 
INTO TABLE Entrada
CHARACTER SET utf8mb4 
FIELDS TERMINATED BY ','          -- delimitador de campos 
LINES TERMINATED BY '\r\n'          -- terminador de línea 
IGNORE 1 LINES                    -- ignora la primera línea (cabecera) 
(id_evento,id_asistente,id_categoria);


/* -------------------------------------------------------------------------------------------
Creación de un trigger que se encargue de controlar que el nº de
entradas vendidas para cada evento no supere el aforo respectivo.
---------------------------------------------------------------------------------------------------*/
DELIMITER //
CREATE TRIGGER before_insert_entrada
BEFORE INSERT ON Entrada
FOR EACH ROW
BEGIN
    DECLARE entradas_vendidas INT DEFAULT 0;
    DECLARE aforo_max INT;

    -- Obtener el aforo máximo de la ubicación del evento
    SELECT u.aforo INTO aforo_max
    FROM Evento e
    JOIN Ubicacion u ON e.id_ubicacion = u.id_ubicacion
    WHERE e.id_evento = NEW.id_evento;

    -- Contar cuántas entradas han sido vendidas para este evento
    SELECT COUNT(*) INTO entradas_vendidas
    FROM Entrada
    WHERE id_evento = NEW.id_evento;

    -- Comprobar si el aforo se superará
    IF (entradas_vendidas + 1) > aforo_max THEN
        SIGNAL SQLSTATE '45000' 
        SET MESSAGE_TEXT = 'Error: No se puede vender más entradas. Se ha alcanzado el aforo máximo del evento.';
    END IF;
END//
DELIMITER ;

/* -------------------------------------------------------------------------------------------
Consultas
---------------------------------------------------------------------------------------------------*/

-- 1. Consulta de eventos por el tipo de actividad

-- ¿Qué eventos se han desarollado cuya actividad sea de tipo 'Música'?
SELECT e.id_evento, e.n_evento, a.tipo, a.n_actividad
FROM evento e
JOIN actividad a ON e.id_actividad = a.id_actividad
WHERE a.tipo = 'Música';

-- ¿Qué eventos se han desarollado cuya actividad sea de tipo 'Teatro'?
SELECT e.id_evento, e.n_evento, a.tipo, a.n_actividad
FROM evento e
JOIN actividad a ON e.id_actividad = a.id_actividad
WHERE a.tipo = 'Teatro';

-- 2. Número de eventos de cada tipo de actividad
SELECT a.tipo AS tipo_actividad, COUNT(*) AS numero_eventos
FROM evento e
JOIN actividad a ON e.id_actividad = a.id_actividad
GROUP BY a.tipo;

-- 3. ¿En qué fecha se han realizado más eventos?
SELECT DATE(fecha_hora) AS fecha, COUNT(*) AS numero_eventos
FROM evento
GROUP BY DATE(fecha_hora)
HAVING COUNT(*) = (
    SELECT MAX(eventos_por_dia)
    FROM (
        SELECT DATE(fecha_hora) AS fecha, COUNT(*) AS eventos_por_dia
        FROM evento
        GROUP BY DATE(fecha_hora)
    ) AS subconsulta
)
ORDER BY fecha;

-- 4. ¿En qué ciudad se han realizado más eventos?
SELECT u.ciudad, COUNT(*) AS numero_eventos
FROM evento e
JOIN ubicacion u ON e.id_ubicacion = u.id_ubicacion
GROUP BY u.ciudad
HAVING COUNT(*) = (
    SELECT MAX(eventos_por_ciudad)
    FROM (
        SELECT u.ciudad, COUNT(*) AS eventos_por_ciudad
        FROM evento e
        JOIN ubicacion u ON e.id_ubicacion = u.id_ubicacion
        GROUP BY u.ciudad
    ) AS subconsulta
)
ORDER BY u.ciudad;

-- 5. ¿En qué actividades participa sólo un artista?
SELECT a.n_actividad, COUNT(p.id_artista) AS numero_artistas
FROM actividad a
JOIN participe p ON a.id_actividad = p.id_actividad
GROUP BY a.id_actividad
HAVING COUNT(p.id_artista) = 1;

-- ¿En qué actividades participan dos artistas?
SELECT a.n_actividad, COUNT(p.id_artista) AS numero_artistas
FROM actividad a
JOIN participe p ON a.id_actividad = p.id_actividad
GROUP BY a.id_actividad
HAVING COUNT(p.id_artista) = 2;

-- 6. ¿En qué ciudad se han realizado sólo eventos de teatro?
SELECT u.ciudad
FROM evento e
JOIN actividad a ON e.id_actividad = a.id_actividad
JOIN ubicacion u ON e.id_ubicacion = u.id_ubicacion
GROUP BY u.ciudad
HAVING COUNT(DISTINCT a.tipo) = 1
   AND MAX(a.tipo) = 'Teatro';
   
-- 7. ¿Qué evento ha sido valorado más veces con 1 (mínimo)?
SELECT e.n_evento, COUNT(v.valoracion) AS cantidad_ceros
FROM evento e
LEFT JOIN valore v ON e.id_evento = v.id_evento
WHERE v.valoracion = 1
GROUP BY e.id_evento, e.n_evento
HAVING COUNT(v.valoracion) = (
    SELECT MAX(cantidad_ceros)
    FROM (
        SELECT COUNT(v.valoracion) AS cantidad_ceros
        FROM evento e
        LEFT JOIN valore v ON e.id_evento = v.id_evento
        WHERE v.valoracion = 1
        GROUP BY e.id_evento
    ) AS subconsulta
)
ORDER BY e.n_evento;

-- 8. ¿Cuántas entradas se han vendido para el evento 'Noche de Rock'?
SELECT e.n_evento, COUNT(en.id_entrada) AS entradas_vendidas
FROM Entrada en
JOIN Evento e ON en.id_evento = e.id_evento
WHERE e.n_evento = 'Noche de Rock';

-- 9. ¿Qué artistas han participado en el evento 'Noche de Rock'?
SELECT e.n_evento, ar.n_artista, p.cache
FROM Participe p
JOIN Evento e ON p.id_actividad = e.id_actividad
JOIN Artista ar ON p.id_artista = ar.id_artista
WHERE e.n_evento = 'Noche de Rock';

-- 10. ¿Para cuál evento se han vendido más entradas?
SELECT e.n_evento, COUNT(en.id_entrada) AS entradas_vendidas
FROM Entrada en
JOIN Evento e ON en.id_evento = e.id_evento
GROUP BY e.n_evento
ORDER BY entradas_vendidas DESC;

-- 11. ¿Cuál es el aforo disponible para el evento 'Cine Bajo las Estrellas'?
SELECT e.n_evento, u.aforo - COUNT(en.id_entrada) AS aforo_restante
FROM Evento e
JOIN Ubicacion u ON e.id_ubicacion = u.id_ubicacion
LEFT JOIN Entrada en ON e.id_evento = en.id_evento
WHERE e.n_evento = 'Cine Bajo las Estrellas'
GROUP BY e.n_evento, u.aforo;

-- 12. ¿Qué eventos están programados para el día '2024-11-10'?
SELECT e.n_evento, e.fecha_hora, u.n_ubicacion
FROM Evento e
JOIN Ubicacion u ON e.id_ubicacion = u.id_ubicacion
WHERE DATE(e.fecha_hora) = '2024-11-10';

-- 13. ¿Cuál es la fecha con más eventos?
SELECT DATE(fecha_hora) AS fecha_evento, COUNT(*) AS total_eventos
FROM Evento
GROUP BY DATE(fecha_hora)
HAVING COUNT(*) = (
    SELECT MAX(eventos_por_dia)
    FROM (
        SELECT DATE(fecha_hora) AS fecha_evento, COUNT(*) AS eventos_por_dia
        FROM Evento
        GROUP BY DATE(fecha_hora)
    ) AS subconsulta
)
ORDER BY fecha_evento;

-- 14. ¿Qué eventos están programados para esta fecha?
SELECT e.n_evento, e.fecha_hora, u.n_ubicacion
FROM Evento e
JOIN Ubicacion u ON e.id_ubicacion = u.id_ubicacion
WHERE DATE(e.fecha_hora) = '2024-10-25';

-- 15. ¿Cuáles son las actividades en las que ha participado el artista 'Banda de Jazz'?
SELECT ar.n_artista, a.n_actividad, p.cache
FROM Participe p
JOIN Actividad a ON p.id_actividad = a.id_actividad
JOIN Artista ar ON p.id_artista = ar.id_artista
WHERE ar.n_artista = 'Banda de Jazz';

-- 16. ¿Qué asistentes no han valorado algún evento de los que han ido, y cuál?
SELECT a.nombre, a.ape1, e.n_evento
FROM Entrada en
JOIN Asistente a ON en.id_asistente = a.id_asistente
JOIN Evento e ON en.id_evento = e.id_evento
LEFT JOIN Valore v ON en.id_evento = v.id_evento AND en.id_asistente = v.id_asistente
WHERE v.valoracion IS NULL;

-- 17. ¿Cuál es la valoración media de cada evento?
SELECT e.n_evento, AVG(v.valoracion) AS valoracion_media
FROM Evento e
JOIN Valore v ON e.id_evento = v.id_evento
GROUP BY e.n_evento
ORDER BY valoracion_media DESC;

-- 18. ¿Cuánto cuestan las entradas VIP de cada evento?
SELECT e.n_evento, c.n_categoria, v.precio
FROM Evento e
JOIN Valga v ON e.id_evento = v.id_evento
JOIN Categoria c ON v.id_categoria = c.id_categoria
WHERE c.n_categoria = 'VIP';

-- 19. Vamos a hacer una consulta con ayuda de una vista que muestre el 
-- nombre del asistente, apellido, categoría del evento, y la cantidad 
-- de eventos dentro de esa categoría a los que ha asistido, pero solo si 
-- ha asistido a más de un evento en la misma categoría.
CREATE VIEW Vista_Entrada_Asistente_Categoria AS
SELECT 
    en.id_entrada, 
    a.id_asistente,
    a.nombre AS nombre_asistente,
    a.ape1 AS apellido_asistente,
    e.id_evento,
    e.n_evento AS nombre_evento,
    c.id_categoria,
    c.n_categoria AS nombre_categoria
FROM Entrada en
JOIN Asistente a ON en.id_asistente = a.id_asistente
JOIN Evento e ON en.id_evento = e.id_evento
JOIN Categoria c ON en.id_categoria = c.id_categoria;

SELECT 
    veac.nombre_asistente, 
    veac.apellido_asistente, 
    veac.nombre_categoria, 
    COUNT(DISTINCT veac.id_evento) AS total_eventos
FROM Vista_Entrada_Asistente_Categoria veac
GROUP BY veac.nombre_asistente, veac.apellido_asistente, veac.nombre_categoria
HAVING COUNT(DISTINCT veac.id_evento) > 1;