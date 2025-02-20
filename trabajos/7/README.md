# Trabajo 7: 

Este trabajo pertenece al módulo de 'Minería de Datos y Modelización Predictiva' y tiene el siguiente enunciado:

##### Enunciado
El objetivo de esta tarea es seleccionar una serie temporal con estacionalidad que contenga aproximadamente 150 observaciones y analizarla mediante técnicas de minería de datos y modelización predictiva. 
En este caso he desarrollado un informe sobre el análisis de una serie temporal de anomalías de temperatura oceánica. Utilizo datos del National Centers for Environmental Information (NCEI) de la NOAA para estudiar tendencias y patrones en la temperatura de los océanos. Aplico diferentes métodos de análisis, incluyendo la descomposición de la serie, la reserva de datos para validación, el ajuste de un modelo de suavizado exponencial de Holt y la selección de un modelo ARIMA óptimo. Finalmente, se comparo los resultados de ambos enfoques y se discuto posibles mejoras, como la incorporación de modelos estacionales o de aprendizaje automático.


## Archivos

- **Code_part1.ipynb**: Este notebook contiene la primera parte del código en Python asociado al análisis de la serie temporal. Incluye la carga y visualización de los datos, la exploración gráfica de la serie, la descomposición estacional, y los primeros pasos para modelizar la serie mediante métodos de suavizado exponencial. También contiene código para la diferenciación de la serie y la evaluación de su estacionariedad utilizando pruebas estadísticas como ADF y KPSS.

- **Code_part2.ipynb**: Este notebook complementa el análisis con la implementación del modelo ARIMA. Contiene el ajuste de distintos modelos ARIMA, la validación de sus parámetros mediante criterios estadísticos como AIC y BIC, y la generación de predicciones. Además, incluye la evaluación del rendimiento del modelo mediante métricas como el error cuadrático medio (MSE) y el error absoluto medio (MAE). También se implementa una versión automática del modelo ARIMA (Auto-ARIMA) para optimizar la selección de parámetros.

- **data.csv**: Dataset anomalías temperaturas oceánicas.