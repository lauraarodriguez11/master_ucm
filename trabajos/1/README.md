# Trabajo 1: Creación de una Base de Datos

Este trabajo pertenece al módulo de 'Bases de Datos SQL' y tiene el siguiente enunciado: 

##### Enunciado
Tenemos una empresa dedicaba a la organización de eventos culturales únicos “ArteVida Cultural”. Organizamos desde deslumbrantes conciertos de música clásica hasta exposiciones de arte vanguardista, pasando por apasionantes obras de teatro y cautivadoras conferencias, llevamos la cultura a todos los rincones de la comunidad.
Necesitamos gestionar la gran variedad de eventos y detalles, así como las ganancias que obtenemos. Para ello, es necesario llevar un registro adecuado de cada evento, de las actividades que se organizan en el evento, de los artistas que los protagonizan, las ubicaciones donde tienen lugar, la venta de entradas y, por supuesto, el entusiasmo de los visitantes que asisten, si les ha gustado o no, para ello el usuario valorará cada evento con un número del 0 al 5.
Hemos decidido diseñar e implementar una base de datos relacional que no solo simplifique la organización de eventos, sino que también permita analizar datos valiosos para tomar decisiones informadas.
En nuestra empresa ofrecemos una serie de actividades que tienen un nombre, un tipo: concierto de distintos tipos de música (clásica, pop, blues, soul, rock and roll, jazz, reggaeton, góspel, country, …), exposiciones, obras de teatro y conferencias, aunque en un futuro estamos dispuestos a organizar otras actividades. Además, en cada actividad participa uno o varios artistas y tiene un coste (suma del caché de los artistas).
El artista tiene un nombre, no tiene un caché fijo, éste depende de la actividad en la que participe, además tendremos una breve biografía suya. Un artista puede participar en muchas actividades.
La ubicación tendrá un nombre (Teatro Maria Guerrero, Estadio Santiago Bernabeu, …), dirección, ciudad o pueblo, aforo, precio del alquiler y características.
De cada evento tenemos que saber el nombre del evento (p.e. “VI festival de música clásica de Alcobendas”), la ubicación, el precio de la entrada, la fecha y la hora, así como una breve descripción de este. En un evento sólo se realiza una actividad.
También tendremos en cuenta los asistentes a los eventos, de los que sabemos su nombre completo, sus teléfonos de contacto y su email. Una persona puede asistir a más de un evento y a un evento pueden asistir varias personas, hemos de controlar que al evento no asistan más personas que el aforo del que dispone la ubicación.
Nos interesará realizar consultas de eventos que se han realizado por el tipo de actividad, número de eventos de cada actividad, en qué fecha se han realizado más eventos, en qué ciudad realizamos más eventos, actividades con un solo artista, ciudad en la que sólo se han realizado eventos de teatro, evento con más ceros en su valoración, …

##### Diseño conceptual
A partir de los requisitos del apartado anterior se construirá el modelo conceptual de datos, en concreto el modelo entidad-relación. Para ello, puedes utilizar herramientas online que te permiten realizar los diagramas muy similares a los de los apuntes de clase.
Se requiere una describir el porqué de la existencia de las entidades (claves, tipos de los atributos: multivalorados, derivados, ..., dominios, …), y de las relaciones (cardinalidades, …). Además, enumerar los conceptos que no están expresados en la modelización porque el modelo entidad-relación no lo permite.
A veces en esta fase en necesario replantearse la fase anterior, pueden surgir detalles no escritos en la fase de especificación, aprovecha para aportar tus matices al enunciado.

##### Diseño lógico
En esta tercera fase del proyecto se aplican las técnicas aprendidas en el curso en el apartado “paso a tablas”, para transformar el diagrama entidad-relación obtenido en la fase 2 en su correspondiente modelo relacional, es decir, en un conjunto de relaciones con las claves primarias y ajenas.

##### Implementación
La implementación en MySQL deberá estar bien presentada, puesto que el código que vamos a escribir en la creación refleja lo visto en las fases anteriores y es el resultado final del trabajo. Es importante la organización y presentación porque facilita el mantenimiento posterior. Se recomienda utilizar identificadores significativos, comentarios, separando la definición de las tablas, la inserción de datos y las consultas. Pero todo en un solo script.
En la definición de la estructura se utilizarán los tipos de datos apropiados a cada columna, se utilizarán restricciones sobre los atributos, restricciones mediante reglas y restricciones sobre los dominios, se definirán las claves y las relaciones entre las tablas.
Los datos se insertarán de forma variada según se ha indicado en el documento “Cargar datos en workbench”. Si tenéis problemas con la inserción de datos con archivos, podéis insertar los datos con insert into. En cualquiera de los casos han de ser suficientes para poder comprobar que las operaciones son correctas.
Las consultas que se realizarán serán variadas y con distinto nivel de complejidad (unas 10 consultas). Se pide calidad en la consulta. También se realizará alguna vista para ser utilizada alguna consulta, y al menos un trigger.


## Archivos

- **ArteVidaCultural.sql**: Este script comprende un conjunto de instrucciones SQL para crear y estructurar una base de datos llamada ArteVidaCultural. Se trata de una base de datos completa para gestionar eventos culturales, donde se almacenan detalles sobre los eventos, artistas, asistentes, ubicaciones, actividades y categorías, además de implementar mecanismos para gestionar las relaciones entre entidades mediante tablas intermedias. También incluye mecanismos automáticos de actualización (triggers) y consultas SQL para obtener información relevante, como las actividades realizadas por artistas, las valoraciones de los eventos y la relación entre asistentes y eventos. Además, se crea una vista que facilita consultas complejas sobre la participación de los asistentes en los eventos.

- **Archivos.csv**: Esta carpeta contiene una serie de archivos .csv que comprenden las tablas de la base de datos.