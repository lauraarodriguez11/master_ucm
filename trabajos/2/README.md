# Trabajo 2: Filtrado y Agregación de un Catálogo Online

Este trabajo pertenece al módulo de 'Bases de Datos No SQL' y tiene el siguiente enunciado: 

##### Enunciado
Sobre un Dataset de datos a elegir, se debe realizar estos ejercicios:
    1. Cargar / Importar dataset
    2. Ejercicios sobre inserción, actualización, proyección y filtrado
    3. Ejercicios sobre pipeline de agregación
Entrega mínima de un fichero PDF, con al menos estos 3 apartados:
    a. Primero, dar contexto y explicar la estructura del dataset
    b. Segundo, incluir código de las queries de TODOS los ejercicios mostrando además una captura de pantalla con los resultados para añadir análisis y/o explicación de los mismos
    c. Tercero, un apartado de conclusiones sobre los análisis realizados en los
apartados previos


## Archivos

- **Script_fashion_products.js**: Este script en JavaScript se utiliza para interactuar con una base de datos MongoDB que contiene información sobre productos de un catálogo de moda. Cubre una amplia gama de operaciones de base de datos en MongoDB, desde inserciones y eliminaciones hasta complejas agregaciones y consultas de agrupación, permitiendo análisis de datos sobre los productos de moda almacenados.

- **Actualizar_dataset.js**: Script que permite asociarle a cada producto de 'styles.json' una url de una imagen de él mismo de 'images.json'. No hace falta ejecutarlo pues ya está actualizado el dataset en cuestión.

- **styles.json**: Se trata de el conjunto de datos que comprende todos los productos del catálogo (ya actualizado con las url de las imágenes) sobre el que se va a trabajar.

- **images.json**: Se trata del dataset que contiene todas las url de los productos que se han añadido previamente a el dataset styles.

- **Print_images.py**: Script de python que permite visualizar algunas de las imágenes del catálogo.