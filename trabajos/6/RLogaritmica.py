# Importo las librerías necesarias
import os
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Elijo el directorio de trabajo e importo las funciones necesarias
os.chdir(r'C:/Users/lrodr/OneDrive/Documentos/master_ucm/trabajos/6')
from FuncionesMineria import (graficoVcramer, validacion_cruzada_glm, glm_forward, glm_backward, glm_stepwise, 
                              Counter, impVariablesLog, curva_roc, sensEspCorte, crear_data_modelo, pseudoR2)

# Cargo los datos depurados
with open('datosEleccionesDep.pickle', 'rb') as f:
    datos_input = pickle.load(f)
    
# Identifico la variable objetivo y la elimino de mi conjunto de datos.
varObjBin = datos_input['AbstencionAlta']
datos_input = datos_input.drop(['AbstentionPtge', 'AbstencionAlta'], axis = 1)

# Veo el reparto original. Compruebo que la variable objetivo tome valor 1 para el evento y 0 para el no evento
pd.DataFrame({
    'n': varObjBin.value_counts()
    , '%': varObjBin.value_counts(normalize = True)
})

# Genero una lista con los nombres de las variables del cojunto de datos input.
variables_input = list(datos_input.columns)  

# Seleciono las variables numéricas
var_cont = list(datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns)

# Seleciono las variables categóricas
var_categ = [variable for variable in variables_input if variable not in var_cont]

# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(datos_input, varObjBin, test_size = 0.2, random_state = 1234567)
# Indico que la variable respuesta es numérica (hay que introducirla en el algoritmo de phython tal y como la va a tratar)
y_train, y_test = y_train.astype(int), y_test.astype(int)

# Interacciones 2 a 2 sólo de las variables continuas
# He seleccionado las mejores variables continuas basándome en su V de Cramer, priorizando aquellas que tienen la mayor 
# asociación con la variable dependiente. Dado que la capacidad computacional de mi ordenador es limitada, no es viable 
# crear interacciones entre todas las variables continuas disponibles, ya que esto implicaría calcular combinaciones 
# exponencialmente más complejas. Por lo tanto, elegir un subconjunto de las mejores variables permite reducir la complejidad 
# del modelo y optimizar los recursos sin sacrificar demasiada precisión en los resultados. Estas 10 variables representan 
# las que tienen mayor probabilidad de aportar información relevante al modelo, maximizando la eficiencia del análisis.
graficoVcramer(datos_input, varObjBin)
interacciones = ['Pob2010', 'PersonasInmueble', 'Age_under19_Ptge']
var_inter = list(itertools.combinations(interacciones, 2))

# Seleccion de variables Backward, con métrica AIC
modeloBackAIC = glm_backward(y_train, x_train, var_cont, var_categ, var_inter, 'AIC')
pseudoR2(modeloBackAIC['Modelo'], modeloBackAIC['X'], y_train)
x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], modeloBackAIC['Variables']['categ'], modeloBackAIC['Variables']['inter'])
pseudoR2(modeloBackAIC['Modelo'], x_test_modeloBackAIC, y_test)
len(modeloBackAIC['Modelo'].coef_[0])
AUC1 = curva_roc(x_test_modeloBackAIC, y_test, modeloBackAIC)

# Seleccion de variables Backward, con métrica BIC
modeloBackBIC = glm_backward(y_train, x_train, var_cont, var_categ, var_inter, 'BIC')
pseudoR2(modeloBackBIC['Modelo'], modeloBackBIC['X'], y_train)
x_test_modeloBackBIC = crear_data_modelo(x_test, modeloBackBIC['Variables']['cont'], modeloBackBIC['Variables']['categ'], modeloBackBIC['Variables']['inter'])
pseudoR2(modeloBackBIC['Modelo'], x_test_modeloBackBIC, y_test)
len(modeloBackBIC['Modelo'].coef_[0])
AUC2 = curva_roc(x_test_modeloBackBIC, y_test, modeloBackBIC)

# Seleccion de variables Forward, con métrica AIC
modeloForwAIC = glm_forward(y_train, x_train, var_cont, var_categ, var_inter, 'AIC')
pseudoR2(modeloForwAIC['Modelo'], modeloForwAIC['X'], y_train)
x_test_modeloForwAIC = crear_data_modelo(x_test, modeloForwAIC['Variables']['cont'], modeloForwAIC['Variables']['categ'], modeloForwAIC['Variables']['inter'])
pseudoR2(modeloForwAIC['Modelo'], x_test_modeloForwAIC, y_test)
len(modeloForwAIC['Modelo'].coef_[0])
AUC3 = curva_roc(x_test_modeloForwAIC, y_test, modeloForwAIC)

# Seleccion de variables Forward, con métrica BIC
modeloForwBIC = glm_forward(y_train, x_train, var_cont, var_categ, var_inter, 'BIC')
pseudoR2(modeloForwBIC['Modelo'], modeloForwBIC['X'], y_train)
x_test_modeloForwBIC = crear_data_modelo(x_test, modeloForwBIC['Variables']['cont'], modeloForwBIC['Variables']['categ'], modeloForwBIC['Variables']['inter'])
pseudoR2(modeloForwBIC['Modelo'], x_test_modeloForwBIC, y_test)
len(modeloForwBIC['Modelo'].coef_[0])
AUC4 = curva_roc(x_test_modeloForwBIC, y_test, modeloForwBIC)

# Seleccion de variables Stepwise, con métrica AIC
modeloStepAIC = glm_stepwise(y_train, x_train, var_cont, var_categ, var_inter, 'AIC')
pseudoR2(modeloStepAIC['Modelo'], modeloStepAIC['X'], y_train)
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], modeloStepAIC['Variables']['categ'], modeloStepAIC['Variables']['inter'])
pseudoR2(modeloStepAIC['Modelo'], x_test_modeloStepAIC, y_test)
len(modeloStepAIC['Modelo'].coef_[0])
AUC5 = curva_roc(x_test_modeloStepAIC, y_test, modeloStepAIC)

# Seleccion de variables Stepwise, con métrica BIC
modeloStepBIC = glm_stepwise(y_train, x_train, var_cont, var_categ, var_inter, 'BIC')
pseudoR2(modeloStepBIC['Modelo'], modeloStepBIC['X'], y_train)
x_test_modeloStepBIC = crear_data_modelo(x_test, modeloStepBIC['Variables']['cont'], modeloStepBIC['Variables']['categ'], modeloStepBIC['Variables']['inter'])
pseudoR2(modeloStepBIC['Modelo'], x_test_modeloStepBIC, y_test)
len(modeloStepBIC['Modelo'].coef_[0])
AUC6 = curva_roc(x_test_modeloStepBIC, y_test, modeloStepBIC)


# SELECCIÓN ALEATORIA DE VARIABLES 
# Inicializo un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

# Realizo 30 iteraciones de selección aleatoria.
for x in range(20):
    print('---------------------------- iter: ' + str(x))
    # Divido los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size = 0.3, random_state = 1234567 + x)
    # Realizo la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = glm_backward(y_train2.astype(int), x_train, var_cont, var_categ, var_inter, 'BIC')
    # Almaceno las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].feature_names_in_))

# Uno las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calculo la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identifico las dos modelos más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]
var_3 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][2])]

## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results = pd.DataFrame({
    'AUC': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_glm(5, x_train, y_train, modeloBackBIC['Variables']['cont'], modeloBackBIC['Variables']['categ'], modeloBackBIC['Variables']['inter'])
    modelo2 = validacion_cruzada_glm(5, x_train, y_train, var_1['cont'], var_1['categ'], var_1['inter'])
    modelo3 = validacion_cruzada_glm(5, x_train, y_train, var_2['cont'], var_2['categ'], var_2['inter'])
    modelo4 = validacion_cruzada_glm(5, x_train, y_train, var_3['cont'], var_3['categ'], var_3['inter'])   
    results_rep = pd.DataFrame({
        'AUC': modelo1 + modelo2 + modelo3 + modelo4
        , 'Resample': ['Rep' + str((rep + 1))]*5*4
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 +[4]*5
    })
    results = pd.concat([results, results_rep], axis = 0)

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  
plt.grid(True) 
grupo_metrica = results.groupby('Modelo')['AUC']
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys()) 
plt.xlabel('Modelo')
plt.ylabel('AUC')
plt.show()  

# Calculo la media del AUC por modelo
media_r2 = results.groupby('Modelo')['AUC'].mean()
print (media_r2)
# Calcular la desviación estándar del AUC por modelo
std_r2 = results.groupby('Modelo')['AUC'].std()
print(std_r2)
# Cuento el número de parámetros en cada modelo
num_params = [len(modeloForwBIC['Modelo'].coef_[0]), len(frec_ordenada['Formula'][0].split('+')), 
 len(frec_ordenada['Formula'][1].split('+')), len(frec_ordenada['Formula'][2].split('+'))]
print(num_params)

ModeloGanador = glm_backward(y_train, x_train, var_3['cont'], var_3['categ'], var_3['inter'], 'BIC')

# Buscamos el mejor punto de corte
# Generamos una rejilla de puntos de corte
posiblesCortes = np.arange(0, 1.01, 0.01).tolist()  # Generamos puntos de corte de 0 a 1 con intervalo de 0.01
# Creamos un DataFrame para almacenar las métricas para cada punto de corte
rejilla = pd.DataFrame({
    'PtoCorte': [],
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'PosPredValue': [],
    'NegPredValue': []
}) 

for pto_corte in posiblesCortes: 
    rejilla = pd.concat(
        [rejilla, sensEspCorte(modelo5['Modelo'], x_test, y_test, pto_corte, var_cont5, var_categ5)],
        axis=0
    )  # Calculamos las métricas para el punto de corte actual y lo agregamos al DataFrame

rejilla['Youden'] = rejilla['Sensitivity'] + rejilla['Specificity'] - 1  # Calculamos el índice de Youden
rejilla.index = list(range(len(rejilla)))  # Reindexamos el DataFrame para que los índices sean consecutivos

plt.plot(rejilla['PtoCorte'], rejilla['Youden'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Youden')
plt.title('Youden')
plt.show()

plt.plot(rejilla['PtoCorte'], rejilla['Accuracy'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()

# Encuentro el punto de corte que maximiza el índice de Youden
p1 = rejilla['PtoCorte'][rejilla['Youden'].idxmax()]

# Encuentro el punto de corte que maximiza la precisión ( Accuracy )
p2 = rejilla['PtoCorte'][rejilla['Accuracy'].idxmax()]

# El resultado es 0.75 para youden y 0.5 para Accuracy. Los comparamos
sensEspCorte(ModeloGanador['Modelo'], x_test, y_test, p1, var_3['cont'], var_3['categ'], var_3['inter'])
sensEspCorte(ModeloGanador['Modelo'], x_test, y_test, p2, var_3['cont'], var_3['categ'], var_3['inter'])

# Vemos las variables mas importantes del modelo ganador
impVariablesLog(ModeloGanador['Modelo'], y_train, x_train, var_3['cont'], var_3['categ'], var_3['inter'])
# Vemos los coeficientes del modelo ganador
summary_glm(ModeloGanador['Modelo'], y_train, ModeloGanador['X'])
coeficientes = ModeloGanador['Modelo'].coef_

nombres_caracteristicas = crear_data_modelo(x_train, var_3['cont'], var_3['categ'], var_3['inter']).columns  # Suponiendo que X_train es un DataFrame de pandas
# Imprime los nombres de las características junto con sus coeficientes
for nombre, coef in zip(nombres_caracteristicas, coeficientes[0]):
    print(f"Variable: {nombre}, Coeficiente: {coef}")

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
x_test_ModeloGanador = crear_data_modelo(x_test, ModeloGanador['Variables']['cont'], ModeloGanador['Variables']['categ'], ModeloGanador['Variables']['inter'])
pseudoR2(ModeloGanador['Modelo'], x_test_ModeloGanador, y_test)
# Es poca la diferencia, por lo que el modelo se puede considerar robusto
 
# Calculamos la diferencia del Area bajo la curva ROC en train y test
curva_roc(crear_data_modelo(x_train , var_3['cont'], var_3['categ'], var_3['inter'], y_train , ModeloGanador))
curva_roc(x_test_ModeloGanador, y_test, ModeloGanador)

# Calculamos la diferencia de las medidas de calidad entre train y test 
sensEspCorte(ModeloGanador['Modelo'], x_train, y_train, 0.5,  var_3['cont'], var_3['categ'], var_3['inter'])
sensEspCorte(ModeloGanador['Modelo'], x_test, y_test, 0.5,  var_3['cont'], var_3['categ'], var_3['inter'])