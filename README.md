# Análisis de Sentimientos en Twitter para Descubrir Contenido Xenófobo hacia los Inmigrantes Venezolanos en Ecuador

_El presente Trabajo de Titulación (TT) tuvo el propósito de determinar la existencia de contenido xenófobo en un conjunto de tuits, recolectados entorno a los inmigrantes venezolanos en Ecuador, se lo llevó a cabo mediante las fases de la metodología para el Descubrimiento de Conocimiento en Texto (KDT)._

_Para determinar la importancia de llevar a cabo el presente estudio, se realizaron entrevistas tanto a la Ing. Celia Jara Galdeman (Gestora de la carrera de Trabajo Social) y al Dr. Paúl Palacios (Especialista en el Área de de Psicología de la Unidad de Bienestar Universitario) que desempeñan su profesión en la Universidad Nacional de Loja, dichas entrevistas están disponibles en el siguiente enlace: https://drive.google.com/open?id=1gNqSMraM3y-aaevO4w7Cbps1MeIJdNKW
Para asegurar su disponibilidad, también se encuentran aquí: https://1drv.ms/u/s!AkrnWa5sI-tphp8lAFdQASrma-FWlg?e=RT31WY_

## Inicio 🚀

_Estas instrucciones te permitirán obtener una copia del proyecto y reproducirlo de manera local en tu computador._

_Están disponibles un lista de videos en donde se explica la correcta utilización de los recursos de este repositorio para garantizar la reproducibilidad de este trabajo. Visitar https://www.youtube.com/playlist?list=PLiM6EEUFTvE0r5pbn13MNRML_sgFajscs_

_**Dataset 1** equivale al conjunto de tuits clasificados por crowdsourcing, obtenido del repositorio de [T-Davidson](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv)._

_**Dataset 2** equivale al conjunto de tuits de interés recolectados entorno a los inmigrantes venezolanos en Ecuador._

### Pre-requisitos 📋

_Software utilizado para realizar el presente TT_

_[Python 3.7](https://www.python.org/downloads/release/python-376/)_

_[Anaconda 3-2020.02](https://repo.anaconda.com/archive/)_

_Spyder IDE_

_Scikit-learn 0.23.2_

_NLTK 3.5_

_Pandas 1.0.5_

_RegEx 2020.7.14_

_Joblib 0.16.0_

_Matplotlib 3.2.2_

_[Twitterscraper 1.6.1](https://pypi.org/project/twitterscraper/)_

_[Imbalanced-learn 0.7.0](https://pypi.org/project/imbalanced-learn/)_

_[Googletrans 3.0.0](https://pypi.org/project/googletrans/) (Utilizar actualización [PyGoogleTranslation](https://pypi.org/project/pygoogletranslation/))_

_[Google Colab](https://colab.research.google.com/)_

_Nota importante: Si Twitterscraper devuelve una lista vacía, es necesario modificar el archivo query.py agregando este Agente de Usuario ```'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.93 Safari/537.36'``` a HEADERS_LIST[]._ 

### Desarrollo y Resultados 🔧

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

Puede encontrar los detalles de este proyecto en: [https://dspace.unl.edu.ec/jspui/handle/123456789/23796](dspace UNL)

## Autores ✒️

* **Raúl Romero** - *Tesista* - [raul.romero@unl.edu.ec](raul.romero@unl.edu.ec)
* **Oscar Cumbicus** - *Director del Trabajo de Titulación* - [oscar.cumbicus@unl.edu.ec](oscar.cumbicus@unl.edu.ec)


---
