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

#%% DESCRIPCIÓN DEL ARCHIVO

#Este documento tiene la finalidad de reproducir los procesos por los cuales fue realizado el informe 'TP-02-INFORME-CTRL'.
#Se realizan diversas técnicas de procesamiento, análisis y visualización de datos en torno a la construcción de modelos predictivos de clasificación.
#El dataset sobre el cual se implementan las técnicas es Fashion-MNIST.

#Cada bloque tiene encabezados descriptivos y comentarios donde se detalla los procedimientos realizados en el mismo.
#Tener en cuenta que este dataset posee gran cantidad de datos, lo que implica que ciertos bloques tengan una larga duración de ejecución.

#%% CARGAMOS EL DATASET FASHION-MNIST
fashion = pd.read_csv('Fashion-MNIST.csv')

#Eliminamos la primer fila pues únicamente es el indice de la fila
fashion = fashion.drop(columns=['Unnamed: 0'])
#%% SEPARAMOS EL DATASET POR PRENDA

remera_top = fashion[fashion['label'] == 0]
pantalon = fashion[fashion['label'] == 1]
pullover = fashion[fashion['label'] == 2]
vestido = fashion[fashion['label'] == 3]
abrigo = fashion[fashion['label'] == 4]
sandalia = fashion[fashion['label'] == 5]
camisa = fashion[fashion['label'] == 6]
zapatilla = fashion[fashion['label'] == 7]
cartera = fashion[fashion['label'] == 8]
bota = fashion[fashion['label'] == 9]

#Nota: los dataset tienen en mismo tamaño, por lo tanto hay misma cantidad de cada clase de prenda

#%% VISUALIZACIÓN IMÁGENES PROMEDIO 

#En este apartado, nos proponemos visualizar la representación genérica de las distintas clases.
#Este "genérico" lo calculamos a partir del cálculo del promedio que cada clase tiene en los píxeles.
#Para ello, usaremos los datasets auxiliares que nos hemos fabricado.

# Listamos las nombres de las clases con su DF correspondiente
clases_con_df = [
    ("Remera_top", remera_top),
    ("Pantalon", pantalon),
    ("Pullover", pullover),
    ("Vestido", vestido),
    ("Abrigo", abrigo),
    ("Sandalia", sandalia),
    ("Camisa", camisa),
    ("Zapatilla", zapatilla),
    ("Cartera", cartera),
    ("Bota", bota)
]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 filas, 5 columnas
axes = axes.flatten()  # Aplanamos para iterar más fácil

for i, (nombre, df) in enumerate(clases_con_df):
    promedio = df.mean()
    img = promedio[:-1].values.reshape((28, 28))
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(nombre)
    axes[i].axis('off')
    
plt.suptitle("Imágenes promedio por clase", fontsize=16)
plt.tight_layout()
plt.show()

#Borramos variables auxiliares usadas en el proceso
del fig, axes, i, nombre, df, img, promedio, clases_con_df

#%% PASAMOS LAS SERIES DE LOS PROMEDIOS A DATAFRAMES DE LAS PRENDAS PROMEDIO DE CADA CLASE

#Pasamos las series de los promedios a dataframes
df_promedio_bota = bota.mean().to_frame().T
df_promedio_camisa = camisa.mean().to_frame().T
df_promedio_cartera = cartera.mean().to_frame().T
df_promedio_pantalon = pantalon.mean().to_frame().T
df_promedio_pullover = pullover.mean().to_frame().T
df_promedio_remera_top = remera_top.mean().to_frame().T
df_promedio_abrigo = abrigo.mean().to_frame().T
df_promedio_sandalia = sandalia.mean().to_frame().T
df_promedio_vestido = vestido.mean().to_frame().T
df_promedio_zapatilla = zapatilla.mean().to_frame().T

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
                SELECT * FROM df_promedio_abrigo
                UNION
                SELECT * FROM df_promedio_sandalia
                UNION
                SELECT * FROM df_promedio_vestido
                UNION
                SELECT * FROM df_promedio_zapatilla;

              """
#Juntamos todos los promedios en un DF
df_promedios_por_clase = dd.sql(consultaSQL).df()

#Borramos variables auxiliares usadas en el proceso
del consultaSQL, df_promedio_bota,df_promedio_camisa,df_promedio_cartera,df_promedio_pantalon,df_promedio_pullover
del df_promedio_remera_top,df_promedio_abrigo,df_promedio_sandalia,df_promedio_vestido,df_promedio_zapatilla

#%% CALCULAMOS LA IMAGEN PROMEDIO DE TODO EL DF
#En base al df anterior, graficamos como se vería una imagen "superpuesta".
#Esto nos da una idea de como se distribuyen los valores de los pixeles en todo el DF
df_promedios = df_promedios_por_clase.mean()
img = df_promedios.iloc[:-1].values.reshape((28,28))
plt.imshow(img, cmap='inferno')
plt.colorbar(label ='intensidad')
plt.title("Promedio visual derivado \n de los promedios clase a clase", ha = 'center')
plt.axis('off')
plt.show()

#Borramos variables auxiliares usadas en el proceso
del img, df_promedios

#%% MOSTRAR PIXELES RELEVANTES
valores = []
promedios_sin_label = df_promedios_por_clase.drop(columns = ["label"])
#Iteramos por cada píxel
for pixel in promedios_sin_label.columns:
    # Para el píxel actual, obtenemos sus 10 valores promedio (uno de cada clase)
    valor_del_pixel_para_cada_clase = promedios_sin_label[pixel]

    # Calculamos el Cuartil 1 (Q1) y el Cuartil 3 (Q3) de esos 10 valores
    q1 = valor_del_pixel_para_cada_clase.quantile(0.25)
    q3 = valor_del_pixel_para_cada_clase.quantile(0.75)

#Calculamos el IQR para este píxel
    iqr_ = q3 - q1

#Agregamos el IQR calculado a nuestra lista
    valores.append(iqr_)

#Reestructurar la lista de valores IQR en una matriz de 28x28
img = np.array(valores).reshape((28, 28))

#Creamos el mapa de calor
plt.figure(figsize=(10, 8))

heatmap = plt.imshow(img, cmap='hot')

plt.colorbar(heatmap, label='IQR del Píxel Promedio entre Clases')

plt.title('IQR de cada píxel entre las 10 imágenes promedio de las clases', fontsize=16)


plt.xticks(np.arange(0, 28, 5))
plt.yticks(np.arange(0, 28, 5))
plt.show()

#Borramos variables auxiliares usadas en el proceso
del img, heatmap, iqr_, q1, q3, pixel, promedios_sin_label, valor_del_pixel_para_cada_clase,valores

#%% BUSCAMOS PIXELES "INUTILES"

#En este bloque, nos proponemos identificar aquellos atributos cuya inferencia en la identificación de las clases es nula.
#Esta nulidad la determinamos en función de la evaluación de los IQR de los píxeles de cada prenda.
#Aquellos píxeles cuyo IQR sea 0 en todas las prendas, son los que definimos como 'atributos inútiles'.

#Nos creamos un diccionario que guarde 
iqr_por_clases_por_pixel = {}

#Diseñamos una función en la que, inicialmente, reciba todos los píxeles y la novena clase
#En la primer iteración, determina aquellos píxeles de la novena clase que poseen IQR = 0
#Luego, de forma recursiva, recibe estos píxeles resultantes y analiza cuáles de ellos poseen IQR = 0 en las clases "anteriores" (las clases del 0 al 8)
#Cuando la clase ingresada sea la 0, la función devolverá los píxeles cuyo IQR fue nulo para cada clase.

def devuelve_pixeles_irrelevantes_iqr(pixeles, clase):

    res = [] #Lista que contendrá los píxeles de IQR = 0

    for pixel in pixeles: #Iteramos sobre la lista de pixeles que recibe la función como parámetro.
        #Analizamos qué píxeles tienen IQR = 0 en la clase pasada como parámetro.
        clase_data = fashion[fashion['label'] == clase].drop(columns='label')
        q1 = clase_data[pixel].quantile(0.25) #Calculamos el primer quartil de los valores del píxel en la clase.
        q3 = clase_data[pixel].quantile(0.75) #Calculamos el tercer quartil de los valores del píxel en la clase.
        iqr_por_clase = q3 - q1 #El IQR es la diferencia entre el tercer y primer cuartil

        if iqr_por_clase == 0: #Si este IQR = 0, lo agregamos a la variable res
            res.append(pixel)

    if clase > 0: #Hasta no llegar a la última clase (la 0), repetimos el proceso de forma recursiva.
        return devuelve_pixeles_irrelevantes_iqr(res, clase-1) #La función se repite con los píxeles de IQR nulo, y con la clase de índice anterior.
    else: #Si ya le hemos aplicado la función a la última clase, finalizamos.
        return res 

#Inicialmente, los píxeles pasados como parámetro serán todos con los que contamos; y la clase será la novena.
pixeles = []

for i in range(0,784):
    pixeles.append(f'pixel{i}')

atributos_inutiles = devuelve_pixeles_irrelevantes_iqr(pixeles, 9)

#Borramos variables auxiliares usadas en el proceso
del i, iqr_por_clases_por_pixel

#%% GRAFICAMOS ATRIBUTOS "INUTILES"

#En este bloque, visualizamos aquellos píxeles inútiles, que poseerán intensidad nula; contrastandolos con aquellos qué no forman parte de este grupo.

#En esta lista, diferenciaremos la intensidad de los píxeles inútiles de los que no lo son.
img=[]

for i in range(0,784): #Recorremos los píxeles
    #Los inútiles poseerán intensidad nula; el resto 1.
    if f'pixel{i}' in atributos_inutiles:
        img.append(0)
    else:
        img.append(1)

#Ploteamos la imágen
img = np.array(img).reshape(28,28)

fig, ax = plt.subplots(figsize=(4.5, 4.5))

# Dibuja la imagen de los píxeles
ax.imshow(img, cmap='gray', vmin=0, vmax=1)
ax.set_title('Relevancia asignada \n a cada píxel')

# Agregamos una leyenda personalizada
patch_relevante = mpatches.Patch(color='white', label='Relevante')
patch_irrelevante = mpatches.Patch(color='black', label='Irrelevante')
plt.legend(handles=[patch_relevante, patch_irrelevante],
           bbox_to_anchor=(0.05, 1),  # Coordenadas (x,y) relativas al eje donde se va a ubicar la leyenda
           loc='lower center',        # Punto de anclaje de la leyenda (el centro de su lado inferior)
           borderaxespad=0.7,         # Para que no quede tan pegado
           facecolor='gray',
           framealpha=0.3)            # Le damos un poco de transparencia para que se aprecie mejor

plt.tight_layout()
plt.show()

#Borramos variables auxiliares usadas en el proceso
del img, i

#%% GRAFICAMOS UN HEATMAP

#En base a la variación iqr de los pixeles por cada clase, determinamos un heatmap.
iqr_matriz = [] #Lista en donde guardaremos el IQR de todos los píxeles de cada clase.
clases = range(0,10) #Lista de clases
for clase in clases:
    #Por cada clase, calculamos el IQR de sus píxeles.
    clase_data = fashion[fashion['label'] == clase].drop(columns='label')
    iqr_ = clase_data.quantile(0.75) - clase_data.quantile(0.25)
    iqr_matriz.append(iqr_.values)

#Toda la información obtenida la pasamos a un dataframe, agregando las clases como índices (para que después se visualicen en el gráfico de salida)
iqr_df = pd.DataFrame(iqr_matriz, index=[f'Clase {i}' for i in range(10)])

#Ploteamos el Heatmap
plt.figure(figsize=(15, 5))
sns.heatmap(iqr_df, cmap='hot', cbar_kws={"label": "IQR"})
plt.xlabel("Índice de píxel")
plt.ylabel("Clase")
plt.title("IQR por píxel y por clase")
plt.show()

#Borramos variables auxiliares usadas en el proceso
del clase, clase_data, clases, iqr_

#%% DIFERENCIAMOS CLASE 2 DE LA 1

#Tomamos el promedio de cada clase y calculamos su diferencia
img = abs((pullover.mean()[:-1].values.reshape((28,28))) - (pantalon.mean()[:-1].values.reshape((28,28))))
plt.imshow(img, cmap = 'hot')
plt.colorbar(label ='intensidad')
plt.title("Diferencia de prenda promedio entre Clases 2 y 1")
plt.show()

#Borramos variables auxiliares usadas en el proceso
del img

#%% DIFERENCIAMOS CLASE 2 DE LA 6
#Tomamos el promedio de cada clase y calculamos su diferencia
img = abs((pullover.mean()[:-1].values.reshape((28,28))) - (camisa.mean()[:-1].values.reshape((28,28))))
plt.imshow(img, cmap = 'hot')
plt.title("Diferencia de prenda promedio entre Clases 2 y 6")
plt.colorbar(label ='intensidad')
plt.show()

#Borramos variables auxiliares usadas en el proceso
del img

#%% DIFERENCIAMOS CLASE 0 DE LA 8
#Tomamos el promedio de cada clase y calculamos su diferencia
img = abs((remera_top.mean()[:-1].values.reshape((28,28))) - (cartera.mean()[:-1].values.reshape((28,28))))
plt.imshow(img, cmap = 'hot')
plt.title("Diferencia de prenda promedio entre Clases 0 y 8")
plt.colorbar(label ='intensidad')
plt.show()

#Borramos variables auxiliares usadas en el proceso
del img

#%% DIFERENCIAMOS CLASE 0 DE LA 3
#Tomamos el promedio de cada clase y calculamos su diferencia
img = abs((remera_top.mean()[:-1].values.reshape((28,28))) - (vestido.mean()[:-1].values.reshape((28,28))))
plt.imshow(img, cmap = 'hot')
plt.title("Diferencia de prenda promedio entre Clases 0 y 3")
plt.colorbar(label ='intensidad')
plt.show()

#Borramos variables auxiliares usadas en el proceso
del img

#%% VARIACION DE LA CLASE 8 CON PSEUDO-RANGO

#PSEUDO-RANGO
#Este pseudo-rango lo definimos en base a la perspectiva de un gráfico Boxplot.
#Es la diferencia entre el límite inferior y superior de la extensión máxima que tienen los whiskers.
#Es decir, definimos al pseudo rango como:
    #Valor máximo alcanzado por el whisker superior - valor mínimo alcanzado por el whisker inferior
    # max(valor_sup <= (Tercer cuartil + 1.5*IQR)) - min(valor_inf >= (Primer cuartil - 1.5*IQR))
#Este pseudo rango nos permite disociarnos de los outliers que puedan llegar a haber.

# Seleccionamos solo la clase 8 y descartamos la columna de etiquetas
pseudo_por_pixel = {}

# Para cada píxel, calculamos sus cuantiles y definimos los límites del whisker
for pixel in cartera.columns:
    q1 = cartera[pixel].quantile(0.25)
    q3 = cartera[pixel].quantile(0.75)
    rango_iqr = iqr(cartera[pixel])
    
    # Límite inferior y superior según la "regla del 1.5 IQR"
    umbral_inf = q1 - 1.5 * rango_iqr
    umbral_sup = q3 + 1.5 * rango_iqr
    
    # El whisker inferior es el mínimo valor que está por encima del umbral inferior
    valor_inferior = cartera.loc[cartera[pixel] >= umbral_inf, pixel].min()
    # El whisker superior es el máximo valor que está por debajo del umbral superior
    valor_superior = cartera.loc[cartera[pixel] <= umbral_sup, pixel].max()
    
    # El pseudo rango es la diferencia entre el whisker superior e inferior
    pseudo_por_pixel[pixel] = valor_superior - valor_inferior

# Convertimos el diccionario en un DataFrame, reestructuramos los datos para que
# tengan la forma 28x28 (asumiendo imágenes de 28x28 pixeles) y lo graficamos
df_pseudo_cartera = pd.DataFrame([pseudo_por_pixel])
img = df_pseudo_cartera.iloc[:, :-1].values.reshape((28, 28))
plt.imshow(img, cmap='hot')
plt.colorbar(label='Intensidad')
plt.title ("Pseudo-rango por pixel - Clase 8 (cartera)")
plt.show()

#Borramos variables auxiliares usadas en el proceso
del df_pseudo_cartera,img,pixel,pseudo_por_pixel,q1,q3,rango_iqr,umbral_inf,umbral_sup,valor_inferior,valor_superior

#%% VARIACION DE LA CLASE 9 CON PSEUDO-RANGO
pseudo_por_pixel = {}

# Para cada píxel, calculamos sus cuantiles y definimos los límites del whisker
for pixel in bota.columns:
    q1 = bota[pixel].quantile(0.25)
    q3 = bota[pixel].quantile(0.75)
    rango_iqr = iqr(bota[pixel])
    
    # Límite inferior y superior según la "regla del 1.5 IQR"
    umbral_inf = q1 - 1.5 * rango_iqr
    umbral_sup = q3 + 1.5 * rango_iqr
    
    # El whisker inferior es el mínimo valor que está por encima del umbral inferior
    valor_inferior = bota.loc[bota[pixel] >= umbral_inf, pixel].min()
    # El whisker superior es el máximo valor que está por debajo del umbral superior
    valor_superior = bota.loc[bota[pixel] <= umbral_sup, pixel].max()
    
    # El pseudo rango es la diferencia entre el whisker superior e inferior
    pseudo_por_pixel[pixel] = valor_superior - valor_inferior

# Convertimos el diccionario en un DataFrame, reestructuramos los datos para que
# tengan la forma 28x28 (asumiendo imágenes de 28x28 pixeles) y lo graficamos
df_pseudo_bota = pd.DataFrame([pseudo_por_pixel])
img = df_pseudo_bota.iloc[:, :-1].values.reshape((28, 28))
plt.imshow(img, cmap='hot')
plt.colorbar(label='Intensidad')
plt.title ("Pseudo-rango por pixel - Clase 9 (bota)")
plt.show()

#Borramos variables auxiliares usadas en el proceso
del df_pseudo_bota,img,pixel,pseudo_por_pixel,q1,q3,rango_iqr,umbral_inf,umbral_sup,valor_inferior,valor_superior

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

# Por último, adjuntamos una función proporcionada por Manuela Cerdeiro (modificada levemente para adaptarlo a nuestras necesidades) que nos será útil más adelante.

def plot_decision_boundary_fashion(X, y, clf, metrica):
    """
    Grafica la frontera de decisión de un clasificador (clf) entrenado sobre dos atributos (dos píxeles)
    del dataset FASHION-MNIST, filtrado para las clases 0 y 8.

    Parámetros:
      - X: matriz de características con dos columnas (intensidades de dos píxeles).
      - y: vector de etiquetas (0 y 8; correspondientes a nuestras clases).
      - clf: clasificador ya entrenado.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    
    pixel1 = X.columns[0]
    pixel2 = X.columns[1]
    
    X = X.to_numpy()
    y = y.to_numpy()
    
    
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
    ax.set_xlabel(f"Intensidad del {pixel1}",fontsize = 14)
    ax.set_ylabel(f"Intensidad del {pixel2}",fontsize = 14)
    ax.set_title(f"Frontera de decisión sobre la clasificación de las clases 0 y 8 usando {metrica}",fontsize = 16)

    plt.show()

#Borramos variables auxiliares usadas en el proceso
del clases_seleccionadas
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
            diferencia_absoluta = min(abs(primer_cuart_cartera - tercer_cuart_remera) , abs(primer_cuart_remera - tercer_cuart_cartera)) #Calculamos la diferencia
            diferenciasIQR[pixel] = diferencia_absoluta #Guardamos el valor
    
    
    #SELECCIÓN POR PSEUDO-RANGO
    umbral_inf_remera = remera_top[pixel].quantile(0.25) - 1.5*iqr(remera_top[pixel]) #Calculamos máxima extensión del limite inferior que puede tener el whisker 
    umbral_sup_remera = remera_top[pixel].quantile(0.75) + 1.5*iqr(remera_top[pixel])  #Calculamos máxima extensión del limite superior que puede tener el whisker
    umbral_inf_cartera = cartera[pixel].quantile(0.25) - 1.5*iqr(cartera[pixel]) #Calculamos máxima extensión del limite inferior que puede tener el whisker 
    umbral_sup_cartera = cartera[pixel].quantile(0.75) + 1.5*iqr(cartera[pixel]) #Calculamos máxima extensión del limite superior que puede tener el whisker 
    
    limite_inf_remera = remera_top.loc[ remera_top[pixel] >= umbral_inf_remera, pixel ].min() #Calculamos extensión real del limite inferior del whisker 
    limite_sup_remera = remera_top.loc[ remera_top[pixel] <= umbral_sup_remera, pixel ].max() #Calculamos extensión real del limite superior del whisker 
    
    limite_inf_cartera = cartera.loc[ cartera[pixel] >= umbral_inf_cartera, pixel ].min() #Calculamos extensión real del limite inferior del whisker 
    limite_sup_cartera = cartera.loc[ cartera[pixel] <= umbral_sup_cartera, pixel ].max() #Calculamos extensión real del limite superior del whisker 
    #Nos interesa únicamente aquellos atributos cuyos rangos sean disjuntos
    if (max(limite_inf_remera, limite_inf_cartera) > min(limite_sup_remera, limite_sup_cartera)):
            #Calculamos la separación entre los rangos
            diferencia_absoluta = min(abs(limite_inf_cartera - limite_sup_remera) , abs(limite_inf_remera - limite_sup_cartera)) #Calculamos la diferencia
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
#Notar que las variables "mejores_pixeles_por_iqr" y "mejores_pixeles_por_pseudo_rango" poseen el mismo conjunto de atributos.
#Es por ello, que optamos por entrenar los modelos a partir de uno de los factores de selección (pues los resultados serían equivalentes)
#Es así, que mantenemos mejores_pixeles_por_iqr y descartamos la otra variable.

#Borramos variables auxiliares usadas en el proceso
del cant_atributos,diferencia_absoluta,diferenciasIQR,diferenciasMediana,diferenciasPromedio,diferenciasPseudoRango,limite_inf_cartera,limite_inf_remera,limite_sup_cartera,limite_sup_remera
del mediana_pixel_cartera,mediana_pixel_remera,primer_cuart_cartera,primer_cuart_remera,promedio_pixel_cartera,promedio_pixel_remera
del tercer_cuart_cartera,tercer_cuart_remera,umbral_inf_cartera,umbral_inf_remera,umbral_sup_cartera,umbral_sup_remera
del pixel ,mejores_pixeles_por_pseudo_rango

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

#Empezamos separando la totalidad de los pixeles y labels, en conjuntos de train y test (obs: nos resulta importante que sean cantidades balanceadas de cada clase)
X_train, X_test, y_train, y_test = train_test_split(pixels,labels,test_size=0.2, random_state = 20, stratify=labels)

#Ahora, separamos los subconjuntos de datos a usar 
X_promedio_train = X_train[atributos_promedio]
X_promedio_test = X_test[atributos_promedio]
X_mediana_train = X_train[atributos_mediana]
X_mediana_test = X_test[atributos_mediana]
X_iqr_train = X_train[atributos_iqr]
X_iqr_test = X_test[atributos_iqr]
#Inicializamos y entrenamos los distintos clasificadores
clasificadorPromedio = KNeighborsClassifier(n_neighbors=20)
clasificadorPromedio.fit(X_promedio_train, y_train)
clasificadorMediana = KNeighborsClassifier(n_neighbors=20)
clasificadorMediana.fit(X_mediana_train, y_train)
clasificadorIQR = KNeighborsClassifier(n_neighbors=20)
clasificadorIQR.fit(X_iqr_train, y_train)

#Luego, nos interesa observar, mediante la función proporcionada por Manuela Cerdeiro, la frontera de decisión de nuestros modelos
plot_decision_boundary_fashion(X_promedio_train, y_train, clasificadorPromedio, 'promedio')
plot_decision_boundary_fashion(X_mediana_train,y_train, clasificadorMediana, 'mediana')
plot_decision_boundary_fashion(X_iqr_train, y_train, clasificadorIQR, 'IQR')


#Por último, analizamos qué tan bien funcionan nuestros clasificadores en nuestros conjuntos de test
#Las métricas con las que evaluaremos son: accuracy score y f1 score.

resultados_promedio = clasificadorPromedio.predict(X_promedio_test)
resultados_mediana = clasificadorMediana.predict(X_mediana_test)
resultados_iqr = clasificadorIQR.predict(X_iqr_test)

print(f'Con el promedio, el accuracy score es de = {accuracy_score(y_test, resultados_promedio)}')
print(f'Con el promedio, el f1 score es de = {f1_score(y_test, resultados_promedio, average = "macro")} \n')
print(f'Con la mediana, el accuracy score es de = {accuracy_score(y_test, resultados_mediana)}')
print(f'Con la mediana, el f1 score es de = {f1_score(y_test, resultados_mediana, average = "macro")}\n')
print(f'Con el iqr, el accuracy score es de = {accuracy_score(y_test, resultados_iqr)}')
print(f'Con el iqr, el f1 score es de = {f1_score(y_test, resultados_iqr, average = "macro")}\n')

# Se observa que, tomando dos atributos y 5 vecinos, el mejor clasificador fue desarrollado a partir de la selección de píxeles cuyo promedio era más diferente en cada clase.

#Borramos variables auxiliares usadas en el proceso
del atributos_iqr,atributos_mediana,atributos_promedio,clasificadorIQR,clasificadorMediana,clasificadorPromedio
del resultados_iqr,resultados_mediana,resultados_promedio,X_iqr_test,X_iqr_train
del X_mediana_test,X_mediana_train,X_promedio_test,X_promedio_train,labels,pixels

#%% DESARROLLO DE MODELOS KNN PARA CLASIFICACIÓN BINARIA: DIVERSOS ATRIBUTOS Y CANTIDAD DE VECINOS

#En este bloque, nos proponemos analizar diversos modelos KNN, generados a partir de diversas cantidades de atributos y vecinos considerados.
#Cada modelo desarrollado, será evaluado según la métrica 'accuracy_score'
#Tanto el número de píxeles como k que serán tomados en cuenta serán de 3 a 10.
#Es por ello que programamos lo siguiente:
resultadosMean = []
resultadosMedian = []
resultadosIQR = []

#Primero vamos iterando sobre la cantidad de atributos que tomamos.    
for i in range(3, 11, 1):
    #Seleccionamos los i mejores píxeles según cada criterio
    pixs_promedio = mejores_pixeles_por_promedio[:i]
    pixs_mediana = mejores_pixeles_por_mediana[:i]
    pixs_iqr = mejores_pixeles_por_iqr[:i]
    
    #Ahora, separamos los subconjuntos de datos a usar en el clasificador
    X_prom_train = X_train[pixs_promedio]
    X_prom_test = X_test[pixs_promedio]
    X_median_train = X_train[pixs_mediana]
    X_median_test = X_test[pixs_mediana]
    X_IQR_train = X_train[pixs_iqr]
    X_IQR_test = X_test[pixs_iqr]
    
    #Inicializamos para cada conjunto de atributos una lista.
    #En estas se guardará para cierta cantidad de atributos, los distintos valores del accuracy_score según la cantidad de vecinos (de 3 a 10)
    resultsMean = []
    resultsMedian = []
    resultsIQR = []

    for j in range(3, 11, 1):
        #Inicializamos y entrenamos los distintos clasificadores
        clfPromedio = KNeighborsClassifier(n_neighbors=j)
        clfPromedio.fit(X_prom_train, y_train)
        clfMediana = KNeighborsClassifier(n_neighbors=j)
        clfMediana.fit(X_median_train, y_train)
        clfIQR = KNeighborsClassifier(n_neighbors=j)
        clfIQR.fit(X_IQR_train, y_train)
        
        #Realizamos las predicciones en base a nuestros conjuntos de prueba
        results_prom = clfPromedio.predict(X_prom_test)
        results_median = clfMediana.predict(X_median_test)
        results_iqr = clfIQR.predict(X_IQR_test)
        
        #Analizamos la performance, colocando los resultados en una lista que pasará a formar parte de la variable 'resultados'
        #Estos resultados los redondeamos a tres cifras.
        resultsMean.append(round(accuracy_score(y_test, results_prom), 4))
        resultsMedian.append(round(accuracy_score(y_test, results_median),4))
        resultsIQR.append(round(accuracy_score(y_test, results_iqr),4))

    resultadosMean.append(resultsMean)
    resultadosMedian.append(resultsMedian)
    resultadosIQR.append(resultsIQR)

#Borramos variables auxiliares usadas en el proceso
del clfIQR,clfMediana,clfPromedio,i,j,mejores_pixeles_por_iqr,mejores_pixeles_por_mediana,mejores_pixeles_por_promedio,pixs_iqr
del pixs_mediana,pixs_promedio,resultsIQR,resultsMean,resultsMedian,results_iqr,results_median,results_prom
del X_IQR_test,X_IQR_train,X_median_test,X_median_train,X_prom_test,X_prom_train

#%% GRÁFICOS RESULTADOS MODELOS KNN PARA CLASIFICACIÓN BINARIA: DIVERSOS ATRIBUTOS Y CANTIDAD DE VECINOS

# Definimos los rangos para las etiquetas de los ejes
num_atributos_labels = list(range(3, 11)) # Para el eje Y
num_vecinos_labels = list(range(3, 11)) # Para el eje X

#Desarrollamos la función para graficar nuestros resultados.
#La idea es graficar cada matriz generada acorde a las distintas cantidades de atributos y vecinos considerados.
#Estas matrices las agrupamos en una misma imágen, que contendrá a los gráficos en posicionadas en dos filas y columnas.
#Se podrán ver los resultados del accuracy_score para cada selección de atributos vista (redondeada a dos decimales)
#Este gráfico permite analizar las diferencias entre la efectividad de predicción de cada modelo desarrolado.

#La función recibe entonces:
    #una matriz de resultados 
    #la posición de la figura en la cual se encontrará el gráfico 
    #un título (que se relaciona con el mecanismo por el cual se eligieron los atributos: Promedio/Mediana/IQR/Pseudo-rango)
def generar_grafico_matriz(matriz, ax, titulo):

    c = ax.imshow(matriz, cmap='viridis', origin='lower') #ploteamos la matriz
    cbar = plt.colorbar(c, ax=ax) # Añadimos una barra de color para interpretar los valores
    for label in cbar.ax.get_yticklabels(): # Para barras de color verticales
        label.set_fontweight('bold')
    
    
    # Configuramos las etiquetas del eje X (Número de Vecinos)
    ax.set_xticks(np.arange(len(num_vecinos_labels)))
    ax.set_xticklabels(num_vecinos_labels)
    ax.set_xlabel("Número de Vecinos (k)", fontsize=15)

    # Configuramos las etiquetas del eje Y (Número de Atributos)
    ax.set_yticks(np.arange(len(num_atributos_labels)))
    ax.set_yticklabels(num_atributos_labels)
    ax.set_ylabel("Número de Atributos", fontsize=15)

    ax.set_title(f"Accuracy Score - Selección por {titulo}", fontsize=19) #Le agregamos título al gráfico.

    #Escribimos los valores del accuracy_score correspondiente a cada posición de la matriz
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, f"{matriz[i, j]:.2f}", ha='center', va='center', color='white', fontsize=11, weight=900)

#Creamos la figura y los subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
fig.suptitle('Accuracy score según las medidas de tendencia/disperción utilizadas', fontsize=24)


#Llamamos a la función para cada matriz de resultados (pasadas a array)
generar_grafico_matriz(np.array(resultadosMean), axes[0], "Promedio")
generar_grafico_matriz(np.array(resultadosMedian), axes[1], "Mediana")
generar_grafico_matriz(np.array(resultadosIQR), axes[2], "IQR")

fig.text(0.5, 0.02, 'Las matrices describen los resultados del score para las distintas cantidades de atributos y vecinos', fontsize=15, color='gray', ha='center', va='bottom')

plt.tight_layout(rect=[0, 0.07, 1, 0.94])
plt.show()

#Borramos variables auxiliares usadas en el proceso
del axes,fig,resultadosIQR,resultadosMean,resultadosMedian

#%% CLASIFICACIÓN MULTICLASE - PRIMERA ETAPA: SEPARACIÓN DE CONJUNTOS EN DEVELOPMENT & EVALUATION

#En primer lugar, separamos nuestros conjuntos en:
    #el que utiizaremos como desarrollo del modelo
    #el conjunto hedl-out o de evaluación; que será en el que finalmente presentaremos los resultados (confiabilidad/acertividad) de nuestro modelo.
    
X = fashion.iloc[:, :-1]  #contiene la información correspondiente a los atributos (pixeles) a evaluar
y = fashion['label'] #contiene la información correspondiente a las etiquetas (clases) de las prendas.

#División de conjuntos de manera que se mantenga la distribución de clases original 
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,test_size=0.1, random_state = 20, stratify = y)

#Los conjuntos X_eval, y_val no serán utilizados hasta el final, en donde se evaluará el modelo seleccionado.

#Borramos variables auxiliares usadas en el proceso
del X,y

#%% CLASIFICACIÓN MULTICLASE - SEGUNDA ETAPA: PRUEBA DE ALTURAS

#Separo conjunto de training y test
X_train, X_test, y_train, y_test = train_test_split(X_dev,y_dev,test_size=0.1, random_state = 50, stratify = y_dev)
#Listamos un conjunto de alturas:
alturas = range(1,11)
#Guardamos los valores  de las metricas para luego graficarla
accuracy_resultados = []
recall_resultados = []
precision_resultados = []
f1_resultados = []
for altura in alturas:
    arbol_decision = tree.DecisionTreeClassifier(criterion = "gini", max_depth= altura)
    arbol_decision = arbol_decision.fit(X_train, y_train)
    
    y_pred = arbol_decision.predict(X_test)
    accuracy_resultados.append(accuracy_score(y_test, y_pred))
    recall_resultados.append(recall_score(y_test, y_pred, average = "macro"))
    precision_resultados.append(precision_score(y_test, y_pred, average = "macro"))
    f1_resultados.append(f1_score(y_test, y_pred, average = "macro"))
    print(f'accuracy score con altura {altura}: {accuracy_score(y_test, y_pred)}')
    print(f'recall score con altura {altura}: {recall_score(y_test, y_pred, average = "macro")}')
    print(f'precision score con altura {altura}: {precision_score(y_test, y_pred, average = "macro")}')
    print(f'f1 score con altura {altura}: {f1_score(y_test, y_pred, average = "macro")}')

#Borramos variables auxiliares usadas en el proceso
del altura,alturas,arbol_decision

#%% GRAFICAMOS SCORE PARA CADA ALTURA
sns.lineplot(x = range(1,11), y = accuracy_resultados, label = 'accuracy',c='blue')
sns.lineplot(x = range(1,11), y = recall_resultados, label = 'recall',c='red',linestyle = '-.')
sns.lineplot(x = range(1,11), y = precision_resultados, label = 'precision',c='green')
sns.lineplot(x = range(1,11), y = f1_resultados, label = 'f1',c = 'gold')
plt.legend(title = 'Metrica:')
plt.title('Evolución metricas por altura con criterio gini')
plt.ylabel('Score')
plt.xlabel('Altura/profundidad del árbol')

#Borramos variables auxiliares usadas en el proceso
del accuracy_resultados,recall_resultados,precision_resultados,f1_resultados

#%% CLASIFICACIÓN MULTICLASE - TERCER ETAPA: SELECCIÓN DE MEJOR ÁRBOL EN BASE A K-FOLDING

#Por favor ser pacientes, este bloque puede tardar en ejecutarse.

#Para conseguir el mejor árbol, desarrollaremos varios modelos, con distintas alturas (de 6 a 10) y criterios (entropía, gini).
#Luego, nos quedaremos aquel que arroje un score mayor tomando como métrica el accuracy_score
#Por ello, aplicaremos el siguiente mecanismo:
    #Usaremos StratifiedKFolding para separar conjuntos de test y desarollo, para los cuales entrenaremos y evaluaremos cada modelo. 
    #Para cada altura y criterio, se sacará el promedio del accuracy_score en los distintos folds
    #Estos promedios se guardarán en un array dependiendo del criterio: 'resultadosPorAlturaConEntropía', 'resultadosPorAlturaConGini'

#Evaluaremos alturas/profundidades del 6 al 10     
alturas = [6,7,8,9,10]
#Utilizaremos 4 folds, donde cada uno contendrá 1750 datos por cada clase (7000 datos en total)
nsplits = 4
kf = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=50)

#Creamos los arrays mencionados
#Una fila por cada altura, donde cada fila contiene el promedio del modelo entrenado y evaluado con KFolding
resultadosPorAlturaConEntropía = np.zeros((len(alturas), 1))
resultadosPorAlturaConGini = np.zeros((len(alturas), 1))

for j, hmax in enumerate(alturas):
    scoreEntropia = 0
    scoreGini = 0
    
    for i, (train_index, test_index) in enumerate(kf.split(X_dev, y_dev)):
        
        kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
        kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
        
        arbolEntropia = tree.DecisionTreeClassifier(max_depth = hmax, criterion='entropy')
        arbolEntropia.fit(kf_X_train, kf_y_train)
        pred = arbolEntropia.predict(kf_X_test)
        scoreEntropia += accuracy_score(kf_y_test,pred)
        
        arbolGini = tree.DecisionTreeClassifier(max_depth = hmax, criterion='gini')
        arbolGini.fit(kf_X_train, kf_y_train)
        pred = arbolGini.predict(kf_X_test)
        scoreGini += accuracy_score(kf_y_test,pred)
        
    resultadosPorAlturaConEntropía[j,0] = scoreEntropia/4
    resultadosPorAlturaConGini[j,0] = scoreGini/4

#Borramos variables auxiliares usadas en el proceso
del alturas,nsplits,kf,arbolEntropia,arbolGini,hmax,i,j,kf_X_test,kf_X_train,kf_y_test,kf_y_train,pred,scoreEntropia,scoreGini,test_index,train_index

#%% GRAFICAMOS LOS ACCURACY SCORE DE LOS CRITERIOS
sns.lineplot(x = range(6,11), y = resultadosPorAlturaConEntropía[:,0], label = 'entropía')
sns.lineplot(x = range(6,11), y = resultadosPorAlturaConGini[:,0], label = 'gini')

plt.xticks(ticks=range(6,11))

plt.legend(title = 'Criterio:')
plt.title('Evolución accuracy_score por altura para cada criterio')
plt.ylabel('Accuracy score')
plt.xlabel('Altura/profundidad del árbol')

#Borramos variables auxiliares usadas en el proceso
del resultadosPorAlturaConEntropía,resultadosPorAlturaConGini

#%% EVALUAMOS LA PERFORMANCE DEL MEJOR ARBOL
#Segun los resultados anteriores, el mejor arbol toma criterio entropy y una maxima altura de 10
mejor_arbol = tree.DecisionTreeClassifier(max_depth = 10, criterion='entropy', random_state=14)
mejor_arbol.fit(X_dev,y_dev)
y_pred = mejor_arbol.predict(X_eval)
matriz_confusion = confusion_matrix(y_eval, y_pred)

clases = [0,1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(8,6))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

print(f'accuracy score = {accuracy_score(y_eval, y_pred)}')
print(f'recall score = {recall_score(y_eval, y_pred, average = "macro")}')
print(f'precision score = {precision_score(y_eval, y_pred, average = "macro")}')
print(f'f1 score = {f1_score(y_eval, y_pred, average = "macro")}')

#Borramos variables auxiliares usadas en el proceso
del matriz_confusion,X_dev,X_eval,X_test,X_train,y_dev,y_eval,y_pred,y_test,y_train

#%% OPCION ALTERNATIVA: ÁRBOL CON MENOS ATRIBUTOS - SELECCIÓN ATRIBUTOS
#Apostamos por desarrollar un árbol con menor cantidad de atributo pero con similares niveles de eficacia en cuanto al nivel de predicción.
#Los atributos que recibirá este árbol serán:
    #Los que el árbol anterior le haya dado 'importancia' (más adelante se enclarecerá este concepto).
    #Y, dentro de este grupo, eliminaremos aquellos que pertenezcan a nuestro conjunto 'atributos_inutiles', que hemos descripto en el análisis exploratorio.
    
#En primer lugar, separamos nuestros conjuntos en:
    #el que utiizaremos como desarrollo del modelo
    #el conjunto held-out o de evaluación; que será en el que finalmente presentaremos los resultados (confiabilidad/acertividad) de nuestro modelo.
    
#tree.feature_importances devuelve una lista en la que se explicita la importancia que tuvo cada atributo pasado al momento de realizar las predicciones.
importancias = mejor_arbol.feature_importances_
#Armamos un dataframe en el que explicitemos la importancia que le da el árbol a cada atributo 
df_importancias = pd.DataFrame({'Atributo': fashion.columns.drop('label'), 'Importancia': importancias})
#Luego, a partir de una consulta SQL, obtendremos solo aquellas quesí fueron consideradas por el árbol, o sea, aquellos cuya importancia sea > 0. 
consultaSQL = """
                SELECT *
                FROM df_importancias
                WHERE Importancia != 0
              """
df_importancias = dd.sql(consultaSQL).df() #Eliminamos aquellas que no poseen importancia.
X_optimo = fashion[df_importancias['Atributo'].tolist()] #Luego, seleccionamos las columnas de fashion que el árbol utilizo.

#Finalmente, sacamos aquellos 'atributos inutiles'
X_optimo = X_optimo[[pixel for pixel in X_optimo.columns if pixel not in atributos_inutiles]]  #contiene la información correspondiente a los atributos no inútiles
y = fashion['label'] #contiene la información correspondiente a las etiquetas (clases) de las prendas.

#División de conjuntos de manera que se mantenga la distribución de clases original 
X_optimo_dev, X_optimo_eval, y_optimo_dev, y_optimo_eval = train_test_split(X_optimo,y,test_size=0.1, random_state = 20, stratify = y)

#Los conjuntos X_eval, y_val no serán utilizados hasta el final, en donde se evaluará el modelo seleccionado.

#%% SELECCIÓN MEJOR ARBOL ALTERNATIVO EN BASE A GRIDSEARCHCV

#Por favor ser pacientes, este bloque puede tardar en ejecutarse.

#Especificamos los hiperparámetros a probar
parametros = {
    'max_depth': [6,7,8,9,10],  # Profundidad máxima del árbol
    'criterion': ['gini', 'entropy'],  # Criterio de división
}

mejor_arbol_optimo = tree.DecisionTreeClassifier(random_state=50)
#Configuramos la función GridSearchCV
grid_search = GridSearchCV(mejor_arbol_optimo, parametros, cv=4, scoring='accuracy', verbose=1)

#Entrenamos el modelo con búsqueda de hiperparámetros
grid_search.fit(X_optimo_dev, y_optimo_dev)

#Vemos los mejores parámetros encontrados
print(f'Mejores hiperparámetros: {grid_search.best_params_}')
print(f'Mejor precisión obtenida: {grid_search.best_score_}')

#%% EVALUAMOS LA PERFORMANCE DEL MEJOR ÁRBOL ALTERNATIVO

#Desarrollamos el árbol en base a las combinación de parámetros que arrojó la función anterior
mejor_arbol_optimo = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 10, random_state=50)
mejor_arbol_optimo.fit(X_optimo_dev, y_optimo_dev)

#Observamos la clasificación que realiza sobre las prendas de nuestro conjunto de evaluación apartado.
y_optimo_pred = mejor_arbol_optimo.predict(X_optimo_eval)
#Desarrollamos y ploteamos la matriz de confusión de los resultados.
matriz_confusion_arbol_optimizado = confusion_matrix(y_optimo_eval, y_optimo_pred)
clases = range(0,11)
plt.figure(figsize=(8,6))
sns.heatmap(matriz_confusion_arbol_optimizado, annot=True, fmt='d', cmap='Reds', xticklabels=clases, yticklabels=clases)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
#Evaluamos al árbol acorde a las métricas accuracy, recall, precision y f1 score.
print(f'accuracy score = {accuracy_score(y_optimo_eval, y_optimo_pred)}')
print(f'recall score = {recall_score(y_optimo_eval, y_optimo_pred, average = "macro")}')
print(f'precision score = {precision_score(y_optimo_eval, y_optimo_pred, average = "macro")}')
print(f'f1 score = {f1_score(y_optimo_eval, y_optimo_pred, average = "macro")}')

#Borramos variables auxiliares usadas en el proceso
del importancias, X_optimo, y, X_optimo_dev, X_optimo_eval, y_optimo_dev, y_optimo_eval, grid_search