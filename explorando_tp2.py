# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 08:25:14 2025

@author: ASUS
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import duckdb as dd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from matplotlib.colors import ListedColormap

#%%
fashion = pd.read_csv('Fashion-MNIST.csv')

#Eliminamos la primer fila pues únicamente es el indice de la fila
fashion = fashion.drop(columns=['Unnamed: 0'])

# Plot imagen 

img = fashion.iloc[32000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

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

#%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: REMERAS/TOPS

img = remera_top.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = remera_top.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = remera_top.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = remera_top.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = remera_top.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 


#%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: PANTALONES

img = pantalon.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = pantalon.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = pantalon.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = pantalon.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = pantalon.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

#Distingir pantalones parece fácil; solo hay que considerar que haya espacio vació en el medio y lineas largas rodeandolo

#%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: PULLOVER

img = pullover.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = pullover.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = pullover.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = pullover.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = pullover.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

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

#%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: SACO

img = saco.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = saco.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = saco.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = saco.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = saco.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

#%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: SANDALIAS

img = sandalia.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = sandalia.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = sandalia.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = sandalia.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = sandalia.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show()

#%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: CAMISAS

img = camisa.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = camisa.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = camisa.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = camisa.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = camisa.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

#%% #%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: ZAPATILLAS

img = zapatilla.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = zapatilla.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = zapatilla.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = zapatilla.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = zapatilla.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

#%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: CARTERA

img = cartera.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = cartera.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = cartera.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = cartera.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = cartera.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

#%% VISUALIZAMOS VARIAS IMÁGENES DE CADA SUBCONJUNTO: BOTAS

img = bota.iloc[0, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = bota.iloc[6999, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = bota.iloc[3500, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = bota.iloc[100, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

img = bota.iloc[1000, :-1].values.reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 

#%%IMÁGENES PROMEDIO CALZADOS
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
              
#%% IMÁGENES PROMEDIO PRENDAS TORSO

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

#%% PROMEDIO PANTALON Y CARTERA

promedio_pantalon = pantalon.mean()
              
img = promedio_pantalon[:-1].values.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()  


promedio_cartera = cartera.mean()
              
img = promedio_cartera[:-1].values.reshape((28,28))
plt.imshow(img, cmap = 'gray')
plt.show()  
#%% PASO LAS SERIES DE LOS PROMEDIOS A DATAFRAMES
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

#%% PONGO LOS PROMEDIOS EN UNA TABLA

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


#%% SCATTERPLOT PROMEDIO PANTALÓN
sns.scatterplot(x = promedio_pantalon.index, y = promedio_pantalon.values)



#%% SCATTERPLOT PROMEDIO ZAPATILLA
sns.scatterplot(x = promedio_zapatilla.index, y = promedio_zapatilla.values)

#%% SCATTERPLOT PROMEDIO CARTERA
sns.scatterplot(x = promedio_cartera.index, y = promedio_cartera.values)

#%% SCATTERPLOT PROMEDIO SANDALIA
sns.scatterplot(x = promedio_sandalia.index, y = promedio_sandalia.values)

#%% SCATTERPLOT PROMEDIO BOTA
sns.scatterplot(x = promedio_bota.index, y = promedio_bota.values) 

#es el que menor alcance de intensidad presenta; con maximos cercanos a 120 

#%% SCATTERPLOT PROMEDIO REMERA-TOP
sns.scatterplot(x = promedio_remera_top.index, y = promedio_remera_top.values)

#%% SCATTERPLOT PROMEDIO CAMISA
sns.scatterplot(x = promedio_camisa.index, y = promedio_camisa.values)

#%% SCATTERPLOT PROMEDIO PULLOVER
sns.scatterplot(x = promedio_pullover.index, y = promedio_pullover.values)

#%% SCATTERPLOT PROMEDIO SACO
sns.scatterplot(x = promedio_saco.index, y = promedio_saco.values)

#%% SCATTERPLOT PROMEDIO VESTIDO
sns.scatterplot(x = promedio_vestido.index, y = promedio_vestido.values)

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

#%%
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

#Grafico el boxplot con el esquema anterior
df_mejor_pixel = fashion[['label', mejor_pixel]]

# Generar el boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y=mejor_pixel, data=df_mejor_pixel)
plt.title(f'Distribución de {mejor_pixel} por clase')
plt.xlabel('Clase')
plt.ylabel('Intensidad del mejor_pixel')
plt.show()

#%%
numeros = range(498,499)
#30-40
#100-110
#120-130
#140-150
#150-160 bastante bueno
#160-180
#193-220-248-249-311-323-324-325-339-367-379
#diferencia entre 350-351 para clase 1
for numero in numeros:
    pixel = 'pixel' + str(numero)
    # Creo un dataframe que tenga solo la información de ese pixel
    df_de_100 = fashion[['label', pixel]]
    
    # Generar el boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y=pixel, data=df_de_100)
    plt.title(f'Distribución de {pixel} por clase')
    plt.xlabel('Clase')
    plt.ylabel('Intensidad del píxel')
    plt.show()
    
#%% CONSULTAS SQL

consultaSQL = """
                SELECT *
                FROM bota
                WHERE (
                    SELECT COUNT (DISTINCT COLUMNS(*))
                    FROM bota) = 1
              """

res = dd.sql(consultaSQL).df()
#%%
#Veré si hay columnas que presentan valores constantes
constant_columns = [col for col in bota.columns if bota[col].nunique() == 1]

print("Columnas con un único valor en bota:", constant_columns)


constant_columns = [col for col in camisa.columns if camisa[col].nunique() == 1]

print("Columnas con un único valor en camisa:", constant_columns)


constant_columns = [col for col in cartera.columns if cartera[col].nunique() == 1]

print("Columnas con un único valor en cartera:", constant_columns)


constant_columns = [col for col in pantalon.columns if pantalon[col].nunique() == 1]

print("Columnas con un único valor en pantalon:", constant_columns)


constant_columns = [col for col in pullover.columns if pullover[col].nunique() == 1]

print("Columnas con un único valor en pullover:", constant_columns)


constant_columns = [col for col in remera_top.columns if remera_top[col].nunique() == 1]

print("Columnas con un único valor remera/top:", constant_columns)


constant_columns = [col for col in saco.columns if saco[col].nunique() == 1]

print("Columnas con un único valor saco:", constant_columns)



constant_columns = [col for col in sandalia.columns if sandalia[col].nunique() == 1]

print("Columnas con un único valor sandalia:", constant_columns)



constant_columns = [col for col in vestido.columns if vestido[col].nunique() == 1]

print("Columnas con un único valor vestido:", constant_columns)



constant_columns = [col for col in zapatilla.columns if zapatilla[col].nunique() == 1]

print("Columnas con un único valor zapatilla:", constant_columns)

#%%
#Veré si hay columnas que no presentan 0
constant_columns = [col for col in bota.columns if bota[col].all() != 0]

print("Columnas con valores no nulos en bota:", constant_columns)


constant_columns = [col for col in camisa.columns if camisa[col].all() != 0]

print("Columnas con valores no nulos en camisa:", constant_columns)


constant_columns = [col for col in cartera.columns if cartera[col].all() != 0]

print("Columnas con valores no nulos en cartera:", constant_columns)


constant_columns = [col for col in pantalon.columns if pantalon[col].all() != 0]

print("Columnas con valores no nulos en pantalon:", constant_columns)


constant_columns = [col for col in pullover.columns if pullover[col].all() != 0]

print("Columnas con valores no nulos en pullover:", constant_columns)


constant_columns = [col for col in remera_top.columns if remera_top[col].all() != 0]

print("Columnas con valores no nulos en remera/top:", constant_columns)


constant_columns = [col for col in saco.columns if saco[col].all() != 0]

print("Columnas con valores no nulos en saco:", constant_columns)



constant_columns = [col for col in sandalia.columns if sandalia[col].all() != 0]

print("Columnas con valores no nulos en sandalia:", constant_columns)



constant_columns = [col for col in vestido.columns if vestido[col].all() != 0]

print("Columnas con valores no nulos en vestido:", constant_columns)



constant_columns = [col for col in zapatilla.columns if zapatilla[col].all() != 0]

print("Columnas con valores no nulos en zapatilla:", constant_columns)

#ninguno 
#%% SEPARACIÓN CLASE O Y 8
clases_seleccionadas = fashion[(fashion['label'] == 0) | (fashion['label'] == 8)]

#ya ví antes que hay 7000 muestras por clase, así que hay balance en cantidades de remeras/tops y carteras

#queremos ajustar un modelo en base a una cantidad reducida de atributos
#luego, usamos las funciones de diferencia en los promedios para determinar los tres máximos

# Separamos la columna de clases de los píxeles
labels = clases_seleccionadas['label']
pixels = clases_seleccionadas.drop(columns=['label'])

# Definimos la clase que queremos comparar, por ejemplo, 0
remeras_tops = 0
carteras = 8

# Creamos un diccionario para guardar la diferencia absoluta de cada píxel
diferencias = {}
# Recorremos cada píxel
for pixel in pixels.columns:
    promedio_pixel_remera = remera_top[pixel].mean()
    promedio_pixel_cartera = cartera[pixel].mean()
    diferencia_absoluta = abs(promedio_pixel_remera - promedio_pixel_cartera) #Calculamos la diferencia
    diferencias[pixel] = diferencia_absoluta #Guardamos el valor

# Fijo un número de atributos
cant_atributos = 5

# Seleccionamos los píxeles con la mayor diferencia
mejores_pixeles = sorted(diferencias, key = diferencias.get, reverse = True)[:cant_atributos]

for pixel in mejores_pixeles:
    print(f"El {pixel} presenta una diferencia de {diferencias[pixel]}")
    print(remera_top[pixel].mean())
    print(cartera[pixel].mean())
    #Grafico el boxplot con el esquema anterior
    df_mejor_pixel = clases_seleccionadas[['label', pixel]]
    
    # Generar el boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y=pixel, data=df_mejor_pixel)
    plt.title(f'Distribución de {pixel} por clase')
    plt.xlabel('Clase')
    plt.ylabel('Intensidad del pixel')
    plt.show()

#%%
X = clases_seleccionadas[['pixel554', 'pixel526']]
y = clases_seleccionadas['label']
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,test_size=0.1, random_state = 20)

#%% CLASIFICADOR KNN
clasificadorPromedio = KNeighborsClassifier(n_neighbors=17)
#17
clasificadorPromedio.fit(X_dev, y_dev)

#%%
res = clasificadorPromedio.predict(X_eval)

print('La precisión del clasificador es ' + str(sum(y_eval == res)/len(y_eval)))

res2 = clasificadorPromedio.predict(X_dev)

print('La precisión del clasificador es ' + str(sum(y_dev == res2)/len(y_dev)))
#%%
def plot_decision_boundary(X, y, clf):
    fig, ax = plt.subplots(figsize=(6, 6))    
    # Crear grilla
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) 
    # Predecir clases en cada punto de la grilla
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Colores
    n_classes = len(np.unique(y))
    colors = plt.cm.Pastel1.colors[:n_classes]
    cmap_light = ListedColormap(colors)
    cmap_bold = ListedColormap(colors)
    # Graficar la frontera de decisión
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=40, edgecolor='k')
    ax.set_xlabel("Largo del pétalo (en centímetros)")
    ax.set_ylabel("Ancho del pétalo (en centímetros)")
    ax.set_title("Frontera de decisión IRIS")
    
#%%
plot_decision_boundary(X_dev.values, y_dev.values,clasificadorPromedio)

plt.show()

#%% SELECCIONAMOS CONJUNTOS DE ATRIBUTOS EN BASE A LA MEDIANA
# Creamos un diccionario para guardar la diferencia absoluta de la mediana de cada píxel
diferencias = {}
# Recorremos cada píxel
for pixel in pixels.columns:
    mediana_pixel_remera = remera_top[pixel].median()
    mediana_pixel_cartera = cartera[pixel].median()
    diferencia_absoluta = abs(mediana_pixel_remera - mediana_pixel_cartera) #Calculamos la diferencia
    diferencias[pixel] = diferencia_absoluta #Guardamos el valor

# Fijo un número de atributos
cant_atributos = 5

# Seleccionamos los píxeles con la mayor diferencia
mejores_pixeles = sorted(diferencias, key = diferencias.get, reverse = True)[:cant_atributos]

for pixel in mejores_pixeles:
    print(f"El {pixel} presenta una diferencia de {diferencias[pixel]}")
    print(remera_top[pixel].median())
    print(cartera[pixel].median())
    #Grafico el boxplot con el esquema anterior
    df_mejor_pixel = clases_seleccionadas[['label', pixel]]

#%%
clasificadorMediana = KNeighborsClassifier(n_neighbors=15)
#17
clasificadorMediana.fit(X_dev, y_dev)

#%%
res3 = clasificadorMediana.predict(X_eval)

print('La precisión del clasificador es ' + str(sum(y_eval == res3)/len(y_eval)))

res4 = clasificadorMediana.predict(X_dev)

print('La precisión del clasificador es ' + str(sum(y_dev == res4)/len(y_dev)))

#%%
diferencias = {}
# Recorremos cada píxel
for pixel in pixels.columns:
    primercuart_pixel_remera = remera_top[pixel].quantile(0.25)
    tercercuart_pixel_remera = remera_top[pixel].quantile(0.75)
    rango_iqr_pixel_remera = primercuart_pixel_remera - tercercuart_pixel_remera
    primercuart_pixel_cartera = cartera[pixel].quantile(0.25)
    tercercuart_pixel_cartera = cartera[pixel].quantile(0.75)
    rango_iqr_pixel_cartera = primercuart_pixel_cartera - tercercuart_pixel_cartera
    diferencia_absoluta = abs(rango_iqr_pixel_remera - rango_iqr_pixel_cartera) #Calculamos la diferencia
    diferencias[pixel] = diferencia_absoluta #Guardamos el valor

# Fijo un número de atributos
cant_atributos = 5

# Seleccionamos los píxeles con la mayor diferencia
mejores_pixeles = sorted(diferencias, key = diferencias.get, reverse = True)[:cant_atributos]

for pixel in mejores_pixeles:
    print(f"El {pixel} presenta una diferencia de {diferencias[pixel]}")

#%%
clasificadorIQR = KNeighborsClassifier(n_neighbors=17)
#17
clasificadorIQR.fit(X_dev, y_dev)

#%%
res5 = clasificadorIQR.predict(X_eval)

print('La precisión del clasificador es ' + str(sum(y_eval == res5)/len(y_eval)))

res6 = clasificadorIQR.predict(X_dev)

print('La precisión del clasificador es ' + str(sum(y_dev == res6)/len(y_dev)))
