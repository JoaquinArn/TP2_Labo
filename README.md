# 👕 Fashion MNIST: Clasificación Binaria y Multiclase

Este repositorio presenta un análisis integral del dataset Fashion MNIST, abarcando desde la exploración de datos hasta el desarrollo de modelos de clasificación binaria y multiclase. Es un proyecto colaborativo desarrollado por estudiantes de Ciencia de Datos (UBA).

## 📌 Objetivos
- Realizar análisis exploratorio para entender distribución y correlación entre clases.
- Implementar modelos de clasificación binaria (`cartera` vs `remera-top`) utilizando KNN.
- Desarrollar modelos multiclase con Árboles de Decisión, evaluando la influencia de la altura del árbol.
- Comparar performance usando atributos completos vs seleccionados.
- Aplicar técnicas de validación como K-Folding y ajuste de hiperparámetros con GridSearchCV.

## 🧰 Herramientas utilizadas
- **Python 3**, **NumPy**, **Pandas**
- **Seaborn**, **Matplotlib**
- **Scikit-Learn**
- Visualizaciones como `decision_boundaries`, `lineplots`, matrices de confusión, curvas de accuracy.

## 📊 Contenido del repositorio
- `TP02-CTRL.py`: Análisis exploratorio de datos y visualización, implementación de KNN para clasificación binaria con selección de atributos y modelos multiclase con Árboles de Decisión + GridSearchCV.
- `TP-02-Informe-CTRL.pdf`: Informe detallado con resultados, gráficos y reflexiones.

## 🧠 Resultados destacados
- Se logró una clasificación binaria con alta precisión tras seleccionar atributos relevantes.
- En la multiclase, se observó una mejora significativa al ajustar la altura de los árboles.
- La validación cruzada permitió evaluar robustez y evitar overfitting.

## ✍️ Contribuyentes
- Arango Joaquín, Cardinale Dante y Herrero Lucas (estudiantes de Ciencia de Datos - UBA)

## 📥 Dataset

El dataset Fashion MNIST no está incluido en este repositorio. Puede descargarse desde:

- [GitHub de Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- O bien directamente a través de librerías como `tensorflow.keras.datasets` o `torchvision.datasets`.

