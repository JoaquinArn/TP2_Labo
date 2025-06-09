# -*- coding: utf-8 -*-
"""
@authors: ARANGO JOAQUIN        342/24
          CARDINALE DANTE       593/24
          HERRERO LUCAS         179/24
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
import duckdb as dd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from scipy.stats import iqr

#%% CARGAMOS EL DATASET FASHION-MNIST
fashion = pd.read_csv('Fashion-MNIST.csv')

#Eliminamos la primer fila pues únicamente es el indice de la fila
fashion = fashion.drop(columns=['Unnamed: 0'])

#%% SEPARAMOS EL DATASET POR PRENDA

remera_top = fashion[fashion['label'] == 0]
pantalon = fashion[fashion['label'] == 1]
pullover = fashion[fashion['label'] == 2]
vestido = fashion[fashion['label'] == 3]
saco = fashion[fashion['label'] == 4]
sandalia = fashion[fashion['label'] == 5]
camisa = fashion[fashion['label'] == 6]
zapatilla = fashion[fashion['label'] == 7]
cartera = fashion[fashion['label'] == 8]
bota = fashion[fashion['label'] == 9]

#Nota: los dataset tienen en mismo tamaño, por lo tanto hay misma cantidad de cada clase de prenda

#%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: VESTIDO

img = vestido.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = vestido.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = vestido.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = vestido.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = vestido.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 
#%% VISUALIZACIÓN IMÁGENES PROMEDIO 

#En este apartado, nos proponemos visualizar la representación genérica de las distintas clases.
#Este "genérico" lo calculamos a partir del cálculo del promedio que cada clase tiene en los píxeles.
#Para ello, usaremos los datasets auxiliares que nos hemos fabricado.

#Para mostrar la imágen, reformateamos el shape de la imágen.
#Notar que nos desprendemos del label al momento de realizar el gráfico.

promedio_bota = bota.mean()
img = promedio_bota.values[:-1].reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()


promedio_sandalia = sandalia.mean()              
img = promedio_sandalia.values[:-1].reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()      


promedio_zapatilla = zapatilla.mean()
img = promedio_zapatilla.values[:-1].reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()   


promedio_remera_top = remera_top.mean()
img = promedio_remera_top[:-1].values.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()


promedio_camisa = camisa.mean()
img = promedio_camisa[:-1].values.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()      


promedio_pullover = pullover.mean()              
img = promedio_pullover[:-1].values.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()  


promedio_saco = saco.mean()
img = promedio_saco[:-1].values.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()  


promedio_vestido = vestido.mean()
img = promedio_vestido[:-1].values.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()  


promedio_pantalon = pantalon.mean()              
img = promedio_pantalon[:-1].values.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()  


promedio_cartera = cartera.mean()
img = promedio_cartera[:-1].values.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()  

#%% PASAMOS LAS SERIES DE LOS PROMEDIOS A DATAFRAMES DE LAS PRENDAS PROMEDIO DE CADA CLASE

#Pasamos las series de los promedios a dataframes
df_promedio_bota = promedio_bota.to_frame().T
df_promedio_camisa = promedio_camisa.to_frame().T
df_promedio_cartera = promedio_cartera.to_frame().T
df_promedio_pantalon = promedio_pantalon.to_frame().T
df_promedio_pullover = promedio_pullover.to_frame().T
df_promedio_remera_top = promedio_remera_top.to_frame().T
df_promedio_saco = promedio_saco.to_frame().T
df_promedio_sandalia = promedio_sandalia.to_frame().T
df_promedio_vestido = promedio_vestido.to_frame().T
df_promedio_zapatilla = promedio_zapatilla.to_frame().T

#Pasamos los promedios a una única tabla a partir de una consultSQL
consultaSQL = """
                SELECT * FROM df_promedio_bota
                UNION
                SELECT * FROM df_promedio_camisa
                UNION
                SELECT * FROM df_promedio_cartera
                UNION
                SELECT * FROM df_promedio_pantalon
                UNION
                SELECT * FROM df_promedio_pullover
                UNION
                SELECT * FROM df_promedio_remera_top
                UNION
                SELECT * FROM df_promedio_saco
                UNION
                SELECT * FROM df_promedio_sandalia
                UNION
                SELECT * FROM df_promedio_vestido
                UNION
                SELECT * FROM df_promedio_zapatilla;

              """

df = dd.sql(consultaSQL).df()

#con el pixel1,2,3,4,5,6,7,8,12,13,18,21,22,23,24,25 se puede distinguir a la zapatilla del resto porque es la única prenda que no tiene nada ahí
#desde el pixel 155 al pixel165, la zapatilla presenta gran diferencia de intensidad
#en los pixeles 169,170 hay una gran diferencia entre la cartera y el resto 

#%% SCATTERPLOT PROMEDIO SANDALIA
sns.scatterplot(x = promedio_sandalia.index, y = promedio_sandalia.values)
#%% BoxPlot para un pixel en especifico y asi observar la distribucion de intensidad por clase
pixel = 'pixel100'

# Creo un dataframe que tenga solo la información de ese pixel
df_de_100 = fashion[['label', pixel]]

# Generar el boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y=pixel, data=df_de_100)
plt.title(f'Distribución de {pixel} por clase')
plt.xlabel('Clase')
plt.ylabel('Intensidad del píxel')
plt.show()

#%% COMPARAMOS LOS PROMEDIOS DE LAS CLASES POR CADA PIXEL
 
# Separamos la columna de clases de los píxeles
labels = fashion['label']
pixels = fashion.drop(columns=['label'])

# Definimos la clase que queremos comparar, por ejemplo, 0
clase_objetivo = 0

# Creamos una máscara para las imágenes que pertenecen a la clase objetivo
mask_clase = (labels == clase_objetivo)

# Creamos un diccionario para guardar la diferencia absoluta de cada píxel
diferencias = {}
# Recorremos cada píxel
for pixel in pixels.columns:
    promedio_clase = pixels.loc[mask_clase, pixel].mean() #Promedio de esa clase
    promedio_restante = pixels.loc[~mask_clase, pixel].mean() #el ~ intercambia los valores de true y false de la mascara
    diferencia_absoluta = abs(promedio_clase - promedio_restante) #Calculamos la diferencia
    diferencias[pixel] = diferencia_absoluta #Guardamos el valor

# Seleccionamos el píxel con la mayor diferencia
mejor_pixel = max(diferencias, key=diferencias.get)
print(f"Para la clase {clase_objetivo}, el píxel con mayor diferencia es '{mejor_pixel}' con diferencia = {diferencias[mejor_pixel]:.3f}")

#Graficamos el boxplot con el esquema anterior
df_mejor_pixel = fashion[['label', mejor_pixel]]

# Generaramos el boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y=mejor_pixel, data=df_mejor_pixel)
plt.title(f'Distribución de {mejor_pixel} por clase')
plt.xlabel('Clase')
plt.ylabel('Intensidad del mejor_pixel')
plt.show()
#%% IMÁGENES A COLOR DE LA DIFERENCIA
promedio_remera_top = remera_top.mean()

img = abs((promedio_remera_top[:-1].values.reshape((28,28))) - (promedio_camisa[:-1].values.reshape((28,28))))
plt.imshow(img, cmap = 'hot')
plt.colorbar(label ='intensidad')
plt.show()


img = ((promedio_remera_top[:-1].values.reshape((28,28))) - ( promedio_pantalon[:-1].values.reshape((28,28))))
plt.imshow(img, cmap = 'hot')
plt.colorbar(label ='intensidad')
plt.show()

#%% SEPARACIÓN CLASE O Y 8 PARA CLASIFICACIÓN BINARIA

clases_seleccionadas = fashion[(fashion['label'] == 0) | (fashion['label'] == 8)]

#Vimos antes que hay 7000 muestras por clase, así que hay balance en cantidades de remeras/tops y carteras en este dataframe.

#Queremos ajustar un modelo en base a una cantidad reducida de atributos.
#Para seleccionar estos atributos, usaremos diferentes métricas:
    #Promedio
    #Mediana
    #IQR
    #Pesudo-rango (cuya definición se presentará más adelante)
#Los píxeles resultantes serán aquellos cuya diferencia absoluta del resultado de la métrica usada sea mayor.

# Separamos la columna de clases de los píxeles
labels = clases_seleccionadas['label']
pixels = clases_seleccionadas.drop(columns=['label'])

# Por último, adjuntamos una función proporcionada por Manuela Cerdeiro que nos será útil más adelante.

def plot_decision_boundary_fashion(X, y, clf):
    """
    Grafica la frontera de decisión de un clasificador (clf) entrenado sobre dos atributos (dos píxeles)
    del dataset FASHION-MNIST, filtrado para las clases 0 y 8.

    Parámetros:
      - X: matriz de características con dos columnas (intensidades de dos píxeles).
      - y: vector de etiquetas (0 y 8).
      - clf: clasificador ya entrenado (por ejemplo, un KNN con k=5).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Debido a que los rangos pueden ser amplios, el paso de la grilla (h) debe escogerse cuidadosamente.
    h = 0.5
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predecir la clase para cada punto de la grilla.
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Definir paletas de colores para dos clases.
    n_classes = len(np.unique(y))  # Debería ser 2 (clases 0 y 8)
    colors = plt.cm.Pastel1.colors[:n_classes]
    cmap_light = ListedColormap(colors)
    cmap_bold = ListedColormap(colors)

    # Graficar la frontera de decisión.
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
    
    # Graficar los puntos de entrenamiento.
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=40, edgecolor='k')

    # Agregar leyenda personalizada.
    clase0_patch = mpatches.Patch(color=colors[0], label="Clase 0")
    clase8_patch = mpatches.Patch(color=colors[1], label="Clase 8")
    ax.legend(handles=[clase0_patch, clase8_patch])
    
    # Ajustar etiquetas y título acorde a los datos.
    ax.set_xlabel("Intensidad del Pixel 1")
    ax.set_ylabel("Intensidad del Pixel 2")
    ax.set_title("Frontera de decisión en Fashion-MNIST (Clases 0 y 8; 2 píxeles)")

    plt.show()

    
#En el próximo bloque, utilizaremos funciones para determinar los píxeles con lo que entrenaremos modelos KNN.

#%% SELECCIÓN DE ATRIBUTOS PARA CLASIFICACIÓN BINARIA

#Iniciamos programando un procedimiento que devuelva aquellos píxeles cuya diferencia absoluta de promedios sea mayor.

#Creamos un diccionario para guardar la diferencia absoluta del promedio de cada píxel
diferenciasPromedio = {}
diferenciasMediana = {}
diferenciasIQR = {}
diferenciasPseudoRango = {}

#Recorremos cada píxel
for pixel in pixels.columns:
    
    #SELECCIÓN POR PROMEDIO
    promedio_pixel_remera = remera_top[pixel].mean() #Calculamos el promedio de cada pixel de la clase remera-top
    promedio_pixel_cartera = cartera[pixel].mean() #Calculamos el promedio de cada pixel de la clase cartera
    diferencia_absoluta = abs(promedio_pixel_remera - promedio_pixel_cartera) #Calculamos la diferencia absoluta
    diferenciasPromedio[pixel] = diferencia_absoluta #Guardamos el valor
    
    #SELECCIÓN POR MEDIANA
    mediana_pixel_remera = remera_top[pixel].median() #Calculamos la mediana de cada pixel de la clase remera-top
    mediana_pixel_cartera = cartera[pixel].median() #Calculamos la mediana de cada pixel de la clase cartera
    diferencia_absoluta = abs(mediana_pixel_remera - mediana_pixel_cartera) #Calculamos la diferencia
    diferenciasMediana[pixel] = diferencia_absoluta #Guardamos el valor
    
    #SELECCIÓN POR IQR
    primer_cuart_remera = remera_top[pixel].quantile(0.25)
    tercer_cuart_remera = remera_top[pixel].quantile(0.75)
    primer_cuart_cartera = cartera[pixel].quantile(0.25)
    tercer_cuart_cartera = cartera[pixel].quantile(0.75)
    if (max(primer_cuart_remera, primer_cuart_cartera) > min(tercer_cuart_remera, tercer_cuart_cartera)):
            #Calculamos la separación entre los rangos
            diferencia_absoluta = max(abs(primer_cuart_cartera - tercer_cuart_remera) , abs(primer_cuart_remera - tercer_cuart_cartera)) #Calculamos la diferencia
            diferenciasIQR[pixel] = diferencia_absoluta #Guardamos el valor
    
    
    #SELECCIÓN POR PSEUDO-RANGO
    #Este pseudo-rango es lo definimos en base a la perspectiva de un gráfico Boxplot.
    #Es la diferencia entre el límite inferior y superior de la extensión máxima que pueden tener los whiskers.
    #Es decir, definimos al pseudo rango como:
        # (Tercer cuartil + 1.5*IQR) - (Primer cuartil - 1.5*IQR)
    #Este pseudo rango nos permite disociarnos de los outliers que puedan llegar a haber.
    limite_inf_remera = remera_top[pixel].quantile(0.25) - 1.5*iqr(remera_top[pixel]) #Calculamos máxima extensión del limite inferior que puede tener el whisker 
    limite_sup_remera = remera_top[pixel].quantile(0.75) + 1.5*iqr(remera_top[pixel])  #Calculamos máxima extensión del limite superior que puede tener el whisker
    limite_inf_cartera = cartera[pixel].quantile(0.25) - 1.5*iqr(cartera[pixel]) #Calculamos máxima extensión del limite inferior que puede tener el whisker 
    limite_sup_cartera = cartera[pixel].quantile(0.75) + 1.5*iqr(cartera[pixel]) #Calculamos máxima extensión del limite superior que puede tener el whisker 
    #Nos interesa únicamente aquellos atributos cuyos rangos sean disjuntos
    if (max(limite_inf_remera, limite_inf_cartera) > min(limite_sup_remera, limite_sup_cartera)):
            #Calculamos la separación entre los rangos
            diferencia_absoluta = max(abs(limite_inf_cartera - limite_sup_remera) , abs(limite_inf_remera - limite_sup_cartera)) #Calculamos la diferencia
            diferenciasPseudoRango[pixel] = diferencia_absoluta #Guardamos el valor


#Fijamos un número de atributos máxima
#Es decir, el máximo número de atributos que estamos dispuestos a usar para construir cada modelo
#Entrenaremos varios modelos, donde usaremos distintas cantidades de atributos.
cant_atributos = 10

#Seleccionamos los píxeles con la mayor diferencia (en base a la cant_atributos máxima determinada; y en base a cada modelo de selección)
mejores_pixeles_por_promedio = sorted(diferenciasPromedio, key = diferenciasPromedio.get, reverse = True)[:cant_atributos]
mejores_pixeles_por_mediana = sorted(diferenciasMediana, key = diferenciasMediana.get, reverse = True)[:cant_atributos]
mejores_pixeles_por_iqr = sorted(diferenciasIQR, key = diferenciasIQR.get, reverse = True)[:cant_atributos]
mejores_pixeles_por_pseudo_rango = sorted(diferenciasPseudoRango, key = diferenciasPseudoRango.get, reverse = True)[:cant_atributos]



#%% DESARROLLO DE MODELOS KNN PARA CLASIFICACIÓN BINARIA: DOS ATRIBUTOS, K = 5.

#En este bloque, nos proponemos realizar los modelos en base a diferentes atributos (y diversas cantidades)
#Nos proponemos comparar los resultados de la clasificación usando métricas como accuracy_score, recall_score, precision_score y f1_score
#Además, nos interesa observar la variación de los resultados de las métricas a partir de la modificación de la cantidad de 'neighbors' tomados por los modelos.

#Empezamos armando modelos KNN con dos atributos y fijando k=5 con k:cantidad de vecinos.
#Es importante observar que en nuestras variables "mejores_pixeles_por_%", los pixeles están ordenados de mayor a menor en función de la diferencia absoluta que presentaron.
#La importancia de esta observación radica que, cuando tomamos menos píxeles de los que establecimos en la cantidad máxima, lo haremos agarrando aquellos del subconjunto cuya diferencia fue superior.

atributos_promedio = mejores_pixeles_por_promedio[:2]
atributos_mediana = mejores_pixeles_por_mediana[:2]
atributos_iqr = mejores_pixeles_por_iqr[:2]
atributos_pseudo_rango = mejores_pixeles_por_pseudo_rango[:2]

#Empezamos separando la totalidad de los pixeles y labels, en conjuntos de train y test (obs: nos resulta importante que sean cantidades balanceadas de cada clase)
X_train, X_test, y_train, y_test = train_test_split(pixels,labels,test_size=0.2, random_state = 20, stratify=labels)

#Ahora, separamos los subconjuntos de datos a usar 
X_promedio_train = X_train[atributos_promedio]
X_promedio_test = X_test[atributos_promedio]
X_mediana_train = X_train[atributos_mediana]
X_mediana_test = X_test[atributos_mediana]
X_iqr_train = X_train[atributos_iqr]
X_iqr_test = X_test[atributos_iqr]
X_pseudo_rango_train = X_train[atributos_pseudo_rango]
X_pseudo_rango_test = X_test[atributos_pseudo_rango]

#Inicializamos y entrenamos los distintos clasificadores
clasificadorPromedio = KNeighborsClassifier(n_neighbors=5)
clasificadorPromedio.fit(X_promedio_train, y_train)
clasificadorMediana = KNeighborsClassifier(n_neighbors=5)
clasificadorMediana.fit(X_mediana_train, y_train)
clasificadorIQR = KNeighborsClassifier(n_neighbors=5)
clasificadorIQR.fit(X_iqr_train, y_train)
clasificadorPseudo_rango = KNeighborsClassifier(n_neighbors=5)
clasificadorPseudo_rango.fit(X_pseudo_rango_train, y_train)

#Luego, nos interesa observar, mediante la función proporcionada por Manuela Cerdeiro, la frontera de decisión de nuestros modelos
plot_decision_boundary_fashion(X_promedio_train.to_numpy(), y_train.to_numpy(), clasificadorPromedio)
plot_decision_boundary_fashion(X_mediana_train.to_numpy(),y_train.to_numpy(), clasificadorMediana)
plot_decision_boundary_fashion(X_iqr_train.to_numpy(), y_train.to_numpy(), clasificadorIQR)
plot_decision_boundary_fashion(X_pseudo_rango_train.to_numpy(), y_train.to_numpy(), clasificadorPseudo_rango)

#Por último, analizamos qué tan bien funcionan nuestros clasificadores en nuestros conjuntos de test
#Las métricas con las que evaluaremos son: accuracy score y f1 score.

resultados_promedio = clasificadorPromedio.predict(X_promedio_test)
resultados_mediana = clasificadorMediana.predict(X_mediana_test)
resultados_iqr = clasificadorIQR.predict(X_iqr_test)
resultados_pseudo_rango = clasificadorPseudo_rango.predict(X_pseudo_rango_test)

print(f'Con el promedio, el accuracy score es de = {accuracy_score(y_test, resultados_promedio)}')
print(f'Con el promedio, el f1 score es de = {f1_score(y_test, resultados_promedio, average = "macro")} \n')
print(f'Con la mediana, el accuracy score es de = {accuracy_score(y_test, resultados_mediana)}')
print(f'Con la mediana, el f1 score es de = {f1_score(y_test, resultados_mediana, average = "macro")}\n')
print(f'Con el iqr, el accuracy score es de = {accuracy_score(y_test, resultados_iqr)}')
print(f'Con el iqr, el f1 score es de = {f1_score(y_test, resultados_iqr, average = "macro")}\n')
print(f'Con el pseudo_rango, el accuracy score es de = {accuracy_score(y_test, resultados_pseudo_rango)}')
print(f'Con el pseudo_rango, el f1 score es de = {f1_score(y_test, resultados_pseudo_rango, average = "macro")}')

# Se observa que, tomando dos atributos y 5 vecinos, el mejor clasificador fue desarrollado a partir de la selección de píxeles cuyo promedio era más diferente en cada clase.

#%% DESARROLLO DE MODELOS KNN PARA CLASIFICACIÓN BINARIA: DIVERSOS ATRIBUTOS Y CANTIDAD DE VECINOS

#En este bloque, nos proponemos analizar diversos modelos KNN, generados a partir de diversas cantidades de atributos y vecinos considerados.
#Cada modelo desarrollado, será evaluado según la métrica 'accuracy_score'
#Tanto el número de píxeles como k que serán tomados en cuenta serán de 3 a 10.
#Es por ello que programamos lo siguiente:
resultados = []

#Primero vamos iterando sobre la cantidad de atributos que tomamos.    
for i in range(3, 10, 1):
    #Seleccionamos los i mejores píxeles según cada criterio
    pixs_promedio = mejores_pixeles_por_promedio[:i]
    pixs_mediana = mejores_pixeles_por_mediana[:i]
    pixs_iqr = mejores_pixeles_por_iqr[:i]
    pixs_pseudo_rango = mejores_pixeles_por_pseudo_rango[:i]
    
    #Ahora, separamos los subconjuntos de datos a usar en el clasificador
    X_prom_train = X_train[pixs_promedio]
    X_prom_test = X_test[pixs_promedio]
    X_median_train = X_train[pixs_mediana]
    X_median_test = X_test[pixs_mediana]
    X_IQR_train = X_train[pixs_iqr]
    X_IQR_test = X_test[pixs_iqr]
    X_pseudorango_train = X_train[pixs_pseudo_rango]
    X_pseudorango_test = X_test[pixs_pseudo_rango]

    for j in range(3, 10, 1):
        #Inicializamos y entrenamos los distintos clasificadores
        clfPromedio = KNeighborsClassifier(n_neighbors=j)
        clfPromedio.fit(X_prom_train, y_train)
        clfMediana = KNeighborsClassifier(n_neighbors=j)
        clfMediana.fit(X_median_train, y_train)
        clfIQR = KNeighborsClassifier(n_neighbors=j)
        clfIQR.fit(X_IQR_train, y_train)
        clfPseudo_rango = KNeighborsClassifier(n_neighbors=j)
        clfPseudo_rango.fit(X_pseudorango_train, y_train)
        
        #Realizamos las predicciones en base a nuestros conjuntos de prueba
        results_prom = clfPromedio.predict(X_prom_test)
        results_median = clfMediana.predict(X_median_test)
        results_iqr = clfIQR.predict(X_IQR_test)
        results_pseudorango = clfPseudo_rango.predict(X_pseudorango_test)
        
        #Analizamos la performance, colocando los resultados en una lista que pasará a formar parte de la variable 'resultados'
        #Estos resultados los redondeamos a tres cifras.
        results = []
        results.append(round(accuracy_score(y_test, results_prom), 3))
        results.append(round(accuracy_score(y_test, results_median),3))
        results.append(round(accuracy_score(y_test, results_iqr),3))
        results.append(round(accuracy_score(y_test, results_pseudorango),3))
        resultados.append(results)
        
#%% CLASIFICACIÓN MULTICLASE - PRIMERA ETAPA: SEPARACIÓN DE CONJUNTOS

#En primer lugar, separamos nuestros conjuntos en:
    #el que utiizaremos como desarrollo del modelo
    #el conjunto hedl-out o de evaluación; que será en el que finalmente presentaremos los resultados (confiabilidad/acertividad) de nuestro modelo.
    
X = fashion.iloc[:, :-1]  #contiene la información correspondiente a los atributos (pixeles) a evaluar
y = fashion['label'] #contiene la información correspondiente a las etiquetas (clases) de las prendas.

#División de conjuntos de manera que se mantenga la distribución de clases original 
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,test_size=0.1, random_state = 20, stratify = y)

#Los conjuntos X_eval, y_val no serán utilizados hasta el final, en donde se evaluará el modelo seleccionado.

#%% CLASIFICACIÓN MULTICLASE - SEGUNDA ETAPA: SELECCIÓN DE ATRIBUTOS

#Separo conjunto de training y test
X_train, X_test, y_train, y_test = train_test_split(X_dev,y_dev,test_size=0.1, random_state = 20, stratify = y_dev)
arbol_decision = tree.DecisionTreeClassifier(criterion = "gini", max_depth= 10)
arbol_decision = arbol_decision.fit(X_dev, y_dev)

y_pred = arbol_decision.predict(X_test)

print(f'accuracy score = {accuracy_score(y_test, y_pred)}')
print(f'recall score = {recall_score(y_test, y_pred, average = "macro")}')
print(f'precision score = {precision_score(y_test, y_pred, average = "macro")}')
print(f'f1 score = {f1_score(y_test, y_pred, average = "macro")}')
#%%
clases = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10']
plt.figure(figsize= [600,50])
tree.plot_tree(arbol_decision,feature_names=fashion.columns[:-1],class_names = clases,filled = True, rounded = True, fontsize = 10)
plt.suptitle('Árbol clasificador tres variables con profundidad 6 y criterio: gini', size = 40)
#%%
matriz_confusion = confusion_matrix(y_test, y_pred)
print(matriz_confusion)

plt.figure(figsize=(8,6))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
#%%
evaluacion = arbol_decision.predict(X_eval)
print(f'accuracy score = {accuracy_score(y_eval, evaluacion)}')
print(f'recall score = {recall_score(y_eval, evaluacion, average = "macro")}')
print(f'precision score = {precision_score(y_eval, evaluacion, average = "macro")}')
print(f'f1 score = {f1_score(y_eval, evaluacion, average = "macro")}')

#%% SELECCIÓN MEJOR ÁRBOL PRIMER FORMA
alturas = [10]
nsplits = 10
kf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=42)

resultados = np.zeros((nsplits, len(alturas)))
# una fila por cada fold, una columna por cada modelo

for i, (train_index, test_index) in enumerate(kf.split(X_dev, y_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, hmax in enumerate(alturas):
        
        arbol = tree.DecisionTreeClassifier(max_depth = hmax, criterion='entropy')
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        score = accuracy_score(kf_y_test,pred)
        
        resultados[i, j] = score

#%% SELECCIÓN MEJOR ÁRBOL SEGUNDA FORMA

# Especificar los hiperparámetros a probar
parametros = {
    'max_depth': [5,6],  # Profundidad máxima del árbol
    'criterion': ['gini', 'entropy'],  # Criterio de división
    'min_samples_leaf': [1]  # Mínimo de muestras en cada hoja
}

# Configurar GridSearchCV
grid_search = GridSearchCV(arbol_decision, parametros, cv=5, scoring='accuracy', verbose=1)

# Entrenar el modelo con búsqueda de hiperparámetros
grid_search.fit(X_eval, y_eval)

# Ver los mejores parámetros encontrados
print(f'Mejores hiperparámetros: {grid_search.best_params_}')
print(f'Mejor precisión obtenida: {grid_search.best_score_}')