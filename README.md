# Proyecto de Machine Learning para la estimación de precios de vehículos de segunda mano

## Autoría

Este proyecto fue desarrollado por Daniel Almería Lapeña (https://github.com/DanielAL22). 
Si tienes alguna pregunta o consulta, no dudes en contactarme a través de [daniel.alm92@gmail.com](mailto:daniel.alm92@gmail.com).

## Introducción e idea de negocio
Este proyecto pretende crear una herramienta que permita al usuario, ya sea un particular o una empresa, obtener una estimación rápida del precio del vehículo que desea comprar o vender.
Esta estimación se lleva a cabo a partir de técnicas de ciencia de datos y aprendizaje automático en base a datos históricos de ventas de vehículos. La idea es poder calcular el precio de un 
determinado vehículo a partir de sus características y otros datos de ventas históricas de forma sencilla y accesible desde cualquier lugar con conexión a internet mediante una aplicación web.

Nuestro producto está ideado para cualquier empresa o particular involucrado en el mercado de vehículos de segunda mano o en la industria automotriz en general.
Pretende dotar a las organizaciones y personas dedicadas a actividades de compraventa, tasación, valoración, arrendamiento o aseguramiento de vehículos usados de un estimador de precios 
preciso y confiable que aporte a la mejora en la toma de decisiones y la generación de transacciones más justas y satisfactorias para todas las partes involucradas.

## Explicación del problema
Para desarrollar la idea citada anteriormente contamos con una base de datos que contiene información acerca de ventas online de vehículos. Estos datos se circunscriben a Alemania en el 
año 2016 y contienen características tales como el nombre del vehículo, la marca, el modelo, el año de registro, los kilómetros recorridos o los caballos de potencia entre otros; y el precio.

Nuestro vector objetivo, la variable que queremos estudiar, es el precio del vehículo. Se trata de una variable continua. 
Una variable continua es aquella que puede tomar cualquier valor dentro de un rango infinito de números reales. Esto significa que nos encontramos ante un problema de regresión donde lo
que queremos es predecir un valor. De acuerdo a esta idea, utilizaremos las métricas para evaluar los modelos predictivos que consideramos adecuadas
- R2
- RMSE
- MAE
- MAPE

## Estructura del proyecto
Se ha intentado desarrollar el proyecto de una forma modular, dividido en 4 grandes partes de desarrollo y una última de puesta en producción. Cada una de ellas recibe unos inputs y genera unos
outputs que a su vez son los inputs de la siguiente fase:

### 1. Limpieza de datos
El objetivo de esta etapa es generar un dataset listo para la fase de EDA (Analisis Exploratorio de los Datos).
Se han llevado a cabo las siguientes tareas:
- Selección de las columnas relevantes para el estudio
- Búsqueda y eliminación de filas duplicadas
- Uniformización del texto de los registros de la variable `name` y la variable `model`
- Recuperación de valores desconocidos y perdidos de las variables `brand` y `model`
- Transformación y creación de nuevas variables `postalZone` y `cubic_centimeters` respectivamente
- Eliminación de registros con datos manifiestamente erróneos o extremos
- Exportación del dataset resultante de la limpieza

### 2. Análisis exploratorio de datos (EDA)
El objetivo de esta etapa es entender, visualizar y extraer información relevante del set de datos para decidir y efectuar las tranformaciones necesarias que lo preparen para la fase de preprocesamiento.

Se han llevado a cabo las siguientes tareas:
- Análisis univariado de cada variable independiente y el vector objetivo
- Análisis bivariado de la relación entre cada variable independiente y el vector objetivo
- Análisis multivariado entre pares de variables y el vector objetivo o entre pares de variables y otras variables independientes
- Transformaciones llevadas a cabo a partir del análisis de variables:
    - Eliminación de registros con precios superiores a los 35.000 euros
    - Eliminación de registros con años inferiores a 1995
    - Eliminación de registros con potencias superiores a 250 PS
    - Transfomración en dato perdido de aquellas cilindradas superiores a 4.0 que se constataron como errores.
    - Unificación de categorías de gas y eléctrico/híbrido para la variable `fuelType`
    - Eliminación en la variable `brand` de los registros pertenecientes a marcas de superdeportivos, con escasos registros o catalogados como otros
    - Eliminación de la variable `model`
    - Eliminacion de la columna `cubic_centimeters` por excesiva correlación
     
- Eliminacion de la columna `name`
- Exportación del dataset resultante

### 3. Análisis y tratamiento de datos perdidos
El objetivo de esta etapa es comprobar la existencia de datos perdidos, analizar su distribución entre las variables y tipología y llevar a cabo los tratamientos pertinentes.

Se han llevado a cabo las siguientes tareas:
- Análisis de datos perdidos
    - Conteo y distribución
    - Análisis del tipo de dato perdido por variable
- Imputación de datos perdidos
    - Preparación de los datos
    - Imputación
    - Análisis de los resultados
- Eliminación de la variable notRepairedDamage
- Exportación del dataset

### 4. Modelamiento predictivo
El objetivo de esta etapa es generar un modelo predictivo utilizando técnicas de machine learning que permita dar respuesta a la necesidad planteada de crear un estimador de precios de vehículos de segunda mano

Se han llevado a cabo las siguientes tareas:
- Creación de subconjuntos de entrenamiento y prueba
- Transformación logarítmica del vector objetivo
- Estandarización
- Modelación predictiva
    - Entrenamiento sobre los datos con imputación de valores perdidos 
    - Entrenamiento sobre los datos con eliminación de valores perdidos
    - Generación de predicciones y análisis de métricas
    - Análisis de coeficientes / importancias de los modelos entrenados
- Serialización del modelo ganador, su escalador y sus columnas de entrenamiento para la puesta en producción

### 5. Puesta en producción y despliegue de la aplicación
El objetivo de esta etapa es generar una aplicación web que consuma el modelo predictivo resultante del análisis anterior accesible en la nube

Se han llevado a cabo las siguientes tareas:
- Desarrollo de la aplicación con Streamlit y Python
- Despliegue de la misma en la nube

## Uso
La aplicación resultante del análisis y desarrollo de las distintas partes del proyecto se encuentra en el siguiente [enlace](https://motorpricingermany.streamlit.app/).

Se pueden consultar las distintas fases de limpieza y análisis de datos en los archivos .ipynb. Es posible que GitHub de problemas para visualizar los archivos completos por su extensión, en este caso deberían descargarse y abrirse con algún editor de código como Jupyter Notebook o Visual Studio Code

## Conclusiones
Este proyecto abordó un problema de regresión donde buscábamos predecir el precio de vehículos de segunda mano a partir de datos históricos de ventas. El análisis de las variables, su transformación y el tratamiento de los datos nulos permitieron llevar a cabo una modelación predictiva satisfactoria. El modelo resultante fue insertado dentro de una aplicación para lo cual se utilizó Streamlit. Despues de las adaptaciones necesarias para el tratamiento de los datos de entrada y de salida al modelo; y del desarrollo de la interfaz visual de la aplicación, esta fue desplegada en la nube, desde donde es accesible de manera remota. Como ideas de mejora se podría plantear el reentrenamiento con data actualizada, la inclusión de más variables, explorar el funcionamiento de otros modelos u combinaciones de hiperparámetros o probar con nuevas estrategias de tratamiento de los datos.

