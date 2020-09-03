# Análisis de Sentimientos en Twitter para Descubrir Contenido Xenófobo hacia los Inmigrantes Venezolanos en Ecuador

_El presente Trabajo de Titulación (TT) tuvo el propósito de determinar la existencia de contenido xenófobo en un conjunto de tuits, recolectados entorno a los inmigrantes venezolanos en Ecuador, se lo llevó a cabo mediante las fases de la metodología para el Descubrimiento de Conocimiento en Texto (KDT)._

## Comenzando 🚀

_Estas instrucciones te permitirán obtener una copia del proyecto y reproducirlo de manera local en tu computador._

_**Dataset 1** equivale al conjunto de tuits clasificados por crowdsourcing, obtenido del repositorio de [T-Davidson](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv)._

_**Dataset 2** equivale al conjunto de tuits de interés recolectados entorno a los inmigrantes venezolanos en Ecuador._

### Pre-requisitos 📋

_Software utilizado para realizar el presente TT_

```
Python 3.7
Scikit-learn 0.23.2
NLTK 3.5
Twitterscraper 1.6.1
Pandas 1.0.5
Imbalanced-learn 0.7.0
RegEx 2020.7.14
Joblib 0.16.0
Matplotlib 3.2.2
Googletrans 3.0.0
```
_Nota importante: Si Twitterscraper devuelve una lista vacía, es necesario modificar el archivo query.py agregando este Agente de Usuario ```'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.93 Safari/537.36'``` a HEADERS_LIST[]._ 

### Instalación 🔧

_Preparar un entorno de desarrollo, por ejemplo Anaconda 3 y Spyder._

_**Etapas del proceso KDT**_
* **Comprender el dominio de la aplicación y los objetivos del proceso KDT**

  _Determinar y analizar los sentimientos de un conjunto de datos recolectados de la red social Twitter._ 

  _Interpretar los resultados obtenidos por los algoritmos ejecutados en la fase de minería de datos._

* **Adquisición o selección de un conjunto de datos objetivo**

  _En la carpeta /data se encuentran el dataset recolectado "dataset_2_inicial", se trata de los tuits con mensajes hacia los venezolanos inmigrantes en Ecuador._

* **Limpieza de datos, preprocesamiento y transformación**

  _A través de la librería Imbalanced-learn se realizó el sobremuestreo de las clases minoritarias mediante la técnica SMOTE._

  _Para la limpieza de los textos se empleó la librería RegEx, la misma que utiliza Expresiones Regulares reconocer y tratar dichos textos._

  _Se aplicó Machine Translation para el proceso de traducción de los tuits, se utilizó la librería Googletrans._

* **Desarrollo de modelos y construcción de hipótesis**

  _Se utilizó la librería Scikit-learn para aplicar tres algoritmos de clasificación: **Regresión Logística, Máquinas de Soporte Vectorial y Naive Bayes**. Mediante el fine-tuning de los modelos se obtuvieron modelos ajustados para mejorar las predicciones posteriores._

* **Elección y ejecución de algoritmos de minería de datos adecuados**

  _Se aplicó los algoritmos de clasificación ya mencionados anteriormente, lo que dio como resultado la clasificación del dataset de interés /data llamado "dataset_2_clasificado", en él se encuentran los tuits clasificados por los 3 algoritmos._

  _Finalmente, se ha exportado los modelos resultantes de este último dataset, así mismo se realizó un fine-tuning de estos modelos finales. Se encuentran en la carpeta /modelos._

* **Interpretación y visualización de resultados**

  _El algoritmo de Máquinas de Soporte Vectorial dio el mejor rendimiento con un 94% de puntuación F1, seguido de la Regresión Logística con un 93% y Naive Bayes con un 89%._

  _Respecto a los sentimientos encontrados, se tiene una media de 570 tuits clasificados como xenófobos, 3088 como ofensivos y 6230 como otro sentimiento. O lo que es lo mismo, en valores porcentuales son: 5,76% xenófobos, 31,23% de lenguaje ofensivo y 63,01% de otros sentimientos._

## Estudio completo 📖

Puede encontrar los detalles de este proyecto en: [link_proyecto]()

## Autores ✒️

* **Raúl Romero** - *Tesista* - [raul.romero@unl.edu.ec](raul.romero@unl.edu.ec)
* **Oscar Cumbicus** - *Director del Trabajo de Titulación* - [oscar.cumbicus@unl.edu.ec](oscar.cumbicus@unl.edu.ec)


---
