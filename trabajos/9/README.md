# Trabajo 9: 

Este trabajo pertenece al módulo de 'Machine Learning' y tiene el siguiente enunciado:

##### Enunciado

El objetivo de este proyecto es desarrollar un modelo predictivo que determine el color original (blanco o negro) de un coche usado a partir de sus características técnicas. Se parte de un dataset real de una empresa de compraventa de vehículos con 4340 registros, que incluye tanto variables numéricas como categóricas. El análisis comienza con una depuración y transformación de los datos. A continuación, se entrena un árbol de decisión, ajustando sus hiperparámetros con GridSearchCV y analizando su estructura e importancia de variables. Posteriormente, se desarrollan modelos más complejos como Random Forest y XGBoost, aplicando una búsqueda sistemática de hiperparámetros mediante validación cruzada de 4 folds y evaluando su rendimiento con métricas como accuracy, precision, recall, F1-score y AUC. 

## Archivos

- **datos_tarea25.xlsx**: Archivo que contiene el conjunto de datos utilizado en la tarea.

- **DecissionTree.ipynb**: Notebook en el que se desarrolla un modelo de árbol de decisión para predecir el color original del vehículo. 

- **RandomForest.ipynb**: Notebook que extiende el análisis con un modelo de Random Forest, comenzando desde el preprocesamiento del dataset y utilizando una búsqueda de hiperparámetros con GridSearchCV.

- **XGBoost.ipynb**: Notebook dedicado al ajuste de un modelo XGBoostClassifier. Se parte del preprocesamiento del dataset y se realiza una búsqueda extensa de hiperparámetros utilizando también validación cruzada.