# Importo las librerías necesarias
import os
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pickle

# Elijo el directorio de trabajo e importo las funciones necesarias
os.chdir(r'C:/Users/lrodr/OneDrive/Documentos/master_ucm/trabajos/6')
from FuncionesMineria import (analizar_variables_categoricas, cuentaDistintos, atipicosAmissing, 
                              patron_perdidos, ImputacionCuant, ImputacionCuali, Vcramer, graficoVcramer, 
                              mosaico_targetbinaria, boxplot_targetbinaria, hist_targetbinaria, Rsq,
                              summary_glm, validacion_cruzada_lm, validacion_cruzada_glm,
                              lm_forward, lm_backward, lm_stepwise, glm_forward, glm_backward, glm_stepwise, 
                              modelEffectSizes, impVariablesLog, curva_roc, sensEspCorte, crear_data_modelo, 
                              pseudoR2)

# Cargo los datos
datos = pd.read_excel('DatosEleccionesEspaña.xlsx')

# Compruebo el tipo de formato de las variables que se han asignado en la lectura.
datos.dtypes

#Elimino las variables objetivo que no voy a usar
datos.drop(['Dcha_Pct', 'Izda_Pct', 'Otros_Pct', 'Derecha', 'Izquierda'])

# Indico las categóricas que aparecen como numéricas
numericasAcategoricas = ['CodigoProvincia', 'AbstencionAlta', 'Izquierda', 'Derecha']

# Las transformo en categóricas
for var in numericasAcategoricas:
    datos[var] = datos[var].astype(str)

# Genero una lista con los nombres de las variables.
variables = list(datos.columns)  

# Selecciono las columnas numéricas del DataFrame
numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Selecciono las columnas categóricas del DataFrame
categoricas = [variable for variable in variables if variable not in numericas]
 
# Frecuencias de los valores en las variables categóricas
analizar_variables_categoricas(datos)

# Cuento el número de valores distintos de cada una de las variables numéricas de un DataFrame, y compruebo que no hay ninguna que tenga menos de 10 valores diferentes
cuentaDistintos(datos) 

# Compruebo que todas las variables tienen el formato que quiero 
datos.dtypes

# Descriptivos de las variables numéricas
descriptivos_num = datos.describe().T

# Añado más descriptivos a los anteriores
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)

# Muestro valores perdidos
datos[variables].isna().sum()

# A veces los 'nan' vienen como como una cadena de caracteres, los modifico a perdidos
for x in categoricas:
    datos[x] = datos[x].replace('nan', np.nan) 

# Missings no declarados de variables cualitativas (NSNC, ?)
datos['Densidad'] = datos['Densidad'].replace('?', np.nan)

# Missings no declarados variables cuantitativas (-1, 99999)
datos['Explotaciones'] = datos['Explotaciones'].replace(99999, np.nan)

# Valores fuera de rango
datos['AbstentionPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['AbstentionPtge']]
datos['Izda_Pct'] = [x if 0 <= x <= 100 else np.nan for x in datos['Izda_Pct']]
datos['Dcha_Pct'] = [x if 0 <= x <= 100 else np.nan for x in datos['Dcha_Pct']]
datos['Otros_Pct'] = [x if 0 <= x <= 100 else np.nan for x in datos['Otros_Pct']]
datos['Age_0-4_Ptge'] = [x if 0 <= x <= 100 else np.nan for x in datos['Age_0-4_Ptge']]
datos['Age_under19_Ptge'] = [x if 0 <= x <= 100 else np.nan for x in datos['Age_under19_Ptge']]
datos['Age_19_65_pct'] = [x if 0 <= x <= 100 else np.nan for x in datos['Age_19_65_pct']]
datos['Age_over65_pct'] = [x if 0 <= x <= 100 else np.nan for x in datos['Age_over65_pct']]
datos['WomanPopulationPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['WomanPopulationPtge']]
datos['ForeignersPtge'] = [x if -100 <= x <= 100 else np.nan for x in datos['ForeignersPtge']]
datos['SameComAutonPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['SameComAutonPtge']]
datos['SameComAutonDiffProvPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['SameComAutonDiffProvPtge']]
datos['DifComAutonPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['DifComAutonPtge']]
datos['UnemployLess25_Ptge'] = [x if 0 <= x <= 100 else np.nan for x in datos['UnemployLess25_Ptge']]
datos['Unemploy25_40_Ptge'] = [x if 0 <= x <= 100 else np.nan for x in datos['Unemploy25_40_Ptge']]
datos['UnemployMore40_Ptge'] = [x if 0 <= x <= 100 else np.nan for x in datos['UnemployMore40_Ptge']]
datos['AgricultureUnemploymentPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['AgricultureUnemploymentPtge']]
datos['IndustryUnemploymentPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['IndustryUnemploymentPtge']]
datos['ConstructionUnemploymentPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['ConstructionUnemploymentPtge']]
datos['ServicesUnemploymentPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['ServicesUnemploymentPtge']]
datos['PobChange_pct'] = [x if -100 <= x <= 100 else np.nan for x in datos['PobChange_pct']]

# Junto categorías poco representadas de las variables categóricas
datos['ActividadPpal'] = datos['ActividadPpal'].replace({'Construccion': 'Const-Ind', 'Industria': 'Const-Ind'})
# Agrupar "Industria" y "Construcción" en una sola categoría tiene sentido debido a su baja representación relativa 
# (menos del 0.2% del total), lo que podría generar problemas de significancia estadística y sesgos en el análisis. 
# Además, ambas pertenecen al sector secundario, compartiendo similitudes conceptuales relacionadas con la transformación 
#de materiales e infraestructuras. Este agrupamiento también simplifica el análisis, facilita la interpretación de resultados 
# y previene problemas derivados del desequilibrio en las categorías.

# Indico la variableObj, el ID y las Input 
datos = datos.set_index(datos['Name']).drop('Name', axis = 1)
varObjCont = datos['AbstentionPtge']
varObjBin = datos['AbstencionAlta']
datos_input = datos.drop(['AbstentionPtge', 'AbstencionAlta'], axis = 1)

# Genero una lista con los nombres de las variables del cojunto de datos input.
variables_input = list(datos_input.columns)  

# Seleciono las variables numéricas
numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Seleciono las variables categóricas
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]


# ATIPICOS

# Cuento el porcentaje de atipicos de cada variable. 
resultados = {x: atipicosAmissing(datos_input[x])[1] / len(datos_input) for x in numericas_input}

# Modifico los atipicos como missings
for x in numericas_input:
    datos_input[x] = atipicosAmissing(datos_input[x])[0]


# MISSINGS
# Visualizo un mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.
patron_perdidos(datos_input)

# Muestro el total de valores perdidos por cada variable
datos_input[variables_input].isna().sum()

# Muestro la proporción de valores perdidos por cada variable (guardo la información)
prop_missingsVars = datos_input.isna().sum()/len(datos_input)

# Creo la variable prop_missings que recoge el número de valores perdidos por cada observación
datos_input['prop_missings'] = datos_input.isna().mean(axis = 1)

# Realizo un estudio descriptivo básico a la nueva variable
datos_input['prop_missings'].describe()

# Calculo el número de valores distintos que tiene la nueva variable
len(datos_input['prop_missings'].unique())

# Elimino las observaciones con mas de la mitad de datos missings 
eliminar = datos_input['prop_missings'] > 0.5
datos_input = datos_input[~eliminar]
varObjBin = varObjBin[~eliminar]
varObjCont = varObjCont[~eliminar]

# Transformo la nueva variable en categórica (ya que tiene pocos valores diferentes)
datos_input["prop_missings"] = datos_input["prop_missings"].astype(str)

# Agrego 'prop_missings' a la lista de nombres de variables input
variables_input.append('prop_missings')
categoricas_input.append('prop_missings')

# Elimino las variables con mas de la mitad de datos missings (no hay ninguna)
eliminar = [prop_missingsVars.index[x] for x in range(len(prop_missingsVars)) if prop_missingsVars[x] > 0.5]
datos_input = datos_input.drop(eliminar, axis = 1)

# No considero necesario recategorizar las variables categóricas analizadas porque todas presentan una 
# distribución suficientemente representativa o ya están estructuradas de manera adecuada para el análisis. 
# Por ejemplo, variables como CodigoProvincia y CCAA ofrecen información detallada y diferenciada por 
# regiones geográficas, cuya agregación podría llevar a la pérdida de información relevante. Las variables 
# binarias, como AbstencionAlta, Izquierda y Derecha, están correctamente definidas y no presentan niveles 
# adicionales ni valores faltantes que justifiquen ajustes. Además, la variable Densidad, aunque categórica, 
# tiene tres niveles bien distribuidos y suficientemente representativos, lo que no compromete la robustez de 
# los análisis posteriores. Por lo tanto, las variables categóricas actuales no requieren recategorización 
# para cumplir los objetivos del estudio.

## IMPUTACIONES
# Imputo todas las cuantitativas, seleccionando el tipo de imputacion: media, mediana o aleatorio
for x in numericas_input:
    datos_input[x] = ImputacionCuant(datos_input[x], 'aleatorio')

# Imputo todas las cualitativas, seleccionando el tipo de imputacion: moda o aleatorio
for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'aleatorio')

# Reviso que no queden datos missings
datos_input.isna().sum()

# Una vez finalizado este proceso, se puede considerar que los datos estan depurados. Los guardo
datosEleccionesDep = pd.concat([varObjBin, varObjCont, datos_input], axis = 1)
with open('datosEleccionesDep.pickle', 'wb') as archivo:
    pickle.dump(datosEleccionesDep, archivo) 

# Obtengo la importancia de las variables
graficoVcramer(datos_input, varObjBin)
graficoVcramer(datos_input, varObjCont)

# Crear un DataFrame para almacenar los resultados del coeficiente V de Cramer
VCramer = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])

for variable in variables_input:
    v_cramer = Vcramer(datos_input[variable], varObjCont)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjCont.name, 'Vcramer': v_cramer},
                             ignore_index=True)
    
for variable in variables_input:
    v_cramer = Vcramer(datos_input[variable], varObjBin)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjBin.name, 'Vcramer': v_cramer},
                             ignore_index=True)

# Veo graficamente el efecto de dos variables cualitativas sobre la binaria
mosaico_targetbinaria(datos_input['CCAA'], varObjBin, 'CCAA')
mosaico_targetbinaria(datos_input['Izquierda'], varObjBin, 'Izquierda')

# Veo graficamente el efecto de dos variables cuantitativas sobre la binaria
boxplot_targetbinaria(datos_input['Population'], varObjBin,'Objetivo', 'Population')
boxplot_targetbinaria(datos_input['DifComAutonPtge'], varObjBin, 'Objetivo','DifComAutonPtge')

hist_targetbinaria(datos_input['Population'], varObjBin, 'Population')
hist_targetbinaria(datos_input['DifComAutonPtge'], varObjBin, 'DifComAutonPtge')

# Correlación entre todas las variables numéricas frente a la objetivo continua 
numericas = datos_input.select_dtypes(include=['int', 'float']).columns
matriz_corr = pd.concat([varObjCont, datos_input[numericas]], axis = 1).corr(method = 'pearson')
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(matriz_corr, annot=True, annot_kws={"size": 5}, cmap='coolwarm', fmt=".2f", cbar=True, mask=mask)
plt.yticks(
    ticks=np.arange(len(matriz_corr.index)) + 0.5,  # Posición centrada para las etiquetas
    labels=matriz_corr.index,  # Nombres de las variables
    rotation=0,  # Sin rotación
    fontsize=8  # Tamaño de la fuente
)    
plt.xticks(
    ticks=np.arange(len(matriz_corr.columns)) + 0.5,  # Posición centrada para las etiquetas
    labels=matriz_corr.columns,  # Nombres de las variables
    rotation=90,  # Rotación para mejor visibilidad
    fontsize=8  # Tamaño de la fuente
)
plt.title("Matriz de correlación de Pearson")
plt.show()


# CONSTRUCCIÓN DEL MODELO DE REGRESIÓN LINEAL

# Identifico las variables continuas 
var_cont = ['Population', 'TotalCensus', 'Izda_Pct', 'Dcha_Pct', 'Otros_Pct',
            'Age_0-4_Ptge', 'Age_under19_Ptge', 'Age_19_65_pct', 'Age_over65_pct', 
            'WomanPopulationPtge',  'ForeignersPtge', 'SameComAutonPtge', 
            'SameComAutonDiffProvPtge', 'DifComAutonPtge', 'UnemployLess25_Ptge', 
            'Unemploy25_40_Ptge', 'UnemployMore40_Ptge', 'AgricultureUnemploymentPtge', 
            'IndustryUnemploymentPtge', 'ConstructionUnemploymentPtge', 'ServicesUnemploymentPtge',
            'totalEmpresas', 'Industria', 'Construccion', 'ComercTTEHosteleria', 'Servicios',
            'inmuebles', 'Pob2010', 'SUPERFICIE', 'PobChange_pct', 'PersonasInmueble',
            'Explotaciones']

# Identifico las variables categóricas
var_categ = ['CodigoProvincia', 'CCAA', 'Izquierda', 'Derecha', 'ActividadPpal', 'Densidad', 'prop_missings']

# Hago la particion
x_train, x_test, y_train, y_test = train_test_split(datos_input, varObjCont, test_size = 0.2, random_state = 1234567)

# Dado que la capacidad computacional de mi ordenador es limitada, he decidido reducir el número de posibles interacciones entre variables continuas en los modelos. 
# Para ello, utilizo la V de Cramer como criterio para seleccionar las variables más relevantes, enfocándome en aquellas que muestran una relación lógica y 
# potencial entre sí. En este análisis, incluyo variables continuas vinculadas a la estructura económica y demográfica, tales como ComercTTEHostelería, Construcción, 
# Servicios e Industria, ya que representan aspectos económicos clave que podrían interactuar para explicar los niveles de abstención.
# Asimismo, incorporo variables demográficas y de infraestructura como Population, TotalCensus y Densidad, las cuales reflejan la escala poblacional y la distribución 
# de recursos en los municipios. Estas interacciones me permiten explorar cómo la combinación de factores económicos y demográficos influye en los niveles de 
# participación electoral, logrando un equilibrio entre la complejidad y la relevancia del modelo.

# Interacciones 2 a 2 sólo de las variables continuas
interacciones = ['ComercTTEHosteleria', 'Construccion', 'Servicios', 'Industria', 'Population', 'TotalCensus', 'Densidad']
#interacciones = var_cont
interacciones_unicas = list(itertools.combinations(interacciones, 2))

# Seleccion de variables Backward, métrica AIC
modeloBackAIC = lm_backward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
Rsq(modeloBackAIC['Modelo'], y_train, modeloBackAIC['X'])
x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], modeloBackAIC['Variables']['categ'], modeloBackAIC['Variables']['inter'])
Rsq(modeloBackAIC['Modelo'], y_test, x_test_modeloBackAIC)
len(modeloBackAIC['Modelo'].params)

# Seleccion de variables Backward, métrica BIC
modeloBackBIC = lm_backward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
Rsq(modeloBackBIC['Modelo'], y_train, modeloBackBIC['X'])
x_test_modeloBackBIC = crear_data_modelo(x_test, modeloBackBIC['Variables']['cont'], modeloBackBIC['Variables']['categ'], modeloBackBIC['Variables']['inter'])
Rsq(modeloBackBIC['Modelo'], y_test, x_test_modeloBackBIC)
len(modeloBackBIC['Modelo'].params)

# Seleccion de variables Forward, métrica AIC
modeloForwAIC = lm_forward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
Rsq(modeloForwAIC['Modelo'], y_train, modeloForwAIC['X'])
x_test_modeloForwAIC = crear_data_modelo(x_test, modeloForwAIC['Variables']['cont'], modeloForwAIC['Variables']['categ'], modeloForwAIC['Variables']['inter'])
Rsq(modeloForwAIC['Modelo'], y_test, x_test_modeloForwAIC)
len(modeloForwAIC['Modelo'].params)

# Seleccion de variables Forward, métrica BIC
modeloForwBIC = lm_forward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
Rsq(modeloForwBIC['Modelo'], y_train, modeloForwBIC['X'])
x_test_modeloForwBIC = crear_data_modelo(x_test, modeloForwBIC['Variables']['cont'], modeloForwBIC['Variables']['categ'], modeloForwBIC['Variables']['inter'])
Rsq(modeloForwBIC['Modelo'], y_test, x_test_modeloForwBIC)
len(modeloForwBIC['Modelo'].params)

# Seleccion de variables Stepwise, métrica AIC
modeloStepAIC = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], modeloStepAIC['Variables']['categ'], modeloStepAIC['Variables']['inter'])
Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)
len(modeloStepAIC['Modelo'].params)

# Seleccion de variables Stepwise, métrica BIC, con interacciones
modeloStepBIC = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])
x_test_modeloStepBIC = crear_data_modelo(x_test, modeloStepBIC['Variables']['cont'], modeloStepBIC['Variables']['categ'], modeloStepBIC['Variables']['inter'])
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)
len(modeloStepBIC['Modelo'].params)
 
# De acuerdo con los resultados obtenidos, el modelo seleccionado sería el método Forward con BIC, ya que logra un equilibrio óptimo entre rendimiento y simplicidad.

# SELECCIÓN ALEATORIA DE VARIABLES 
# Inicializo un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

# Realizo 30 iteraciones de selección aleatoria.
for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Divido los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size = 0.3, random_state = 1234567 + x)
    
    # Realizo la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_forward(y_train2.astype(int), x_train2, var_cont, var_categ, interacciones_unicas, 'BIC')
    
    # Almaceno las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))

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
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloForwBIC['Variables']['cont']
        , modeloForwBIC['Variables']['categ']
        , modeloForwBIC['Variables']['inter']
    )
    modelo2 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_1['cont']
        , var_1['categ']
        , var_1['inter']
    )
    modelo3 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_2['cont']
        , var_2['categ']
        , var_2['inter']
    )
    modelo4 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_3['cont']
        , var_3['categ']
        , var_3['inter']
    )
        
    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 + modelo4
        , 'Resample': ['Rep' + str((rep + 1))]*5*4
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 +[4]*5
    })
    results = pd.concat([results, results_rep], axis = 0)

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  
plt.grid(True) 
grupo_metrica = results.groupby('Modelo')['Rsquared']
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys()) 
plt.xlabel('Modelo')
plt.ylabel('Rsquared')
plt.show()  

# Calculo la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
print (media_r2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2)
# Cuento el número de parámetros en cada modelo
num_params = [len(modeloForwBIC['Modelo'].params), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+')),
                 len(frec_ordenada['Formula'][2].split('+'))]
print(num_params)

ModeloGanador = modeloForwBIC

# Vemos los coeficientes del modelo ganador
ModeloGanador['Modelo'].summary()
# Todos los parámetros del modelo son significativos

# Evalúo la estabilidad del modelo a partir de las diferencias en train y test
Rsq(ModeloGanador['Modelo'], y_train, ModeloGanador['X'])
x_test_modeloganador = crear_data_modelo(x_test, ModeloGanador['Variables']['cont'], ModeloGanador['Variables']['categ'], ModeloGanador['Variables']['inter'])
Rsq(ModeloGanador['Modelo'], y_test, x_test_modeloganador)
modelEffectSizes(ModeloGanador, y_train, x_train, ModeloGanador['Variables']['cont'], ModeloGanador['Variables']['categ'], ModeloGanador['Variables']['inter'])


# CONSTRUCCIÓN DEL MODELO DE REGRESIÓN LOGÍSTICA
