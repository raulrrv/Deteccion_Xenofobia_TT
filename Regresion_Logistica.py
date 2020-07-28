# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import re      #expresiones regulares
import nltk

stopwords = nltk.corpus.stopwords.words("english")

#lbfgs: Algoritmo de Broyden – Fletcher – Goldfarb – Shanno de memoria limitada
#leer dataset sin etiquetas
df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_clean_es_nodupl_en_complet.xlsx")
"""
df = pd.DataFrame({"text": ["Venezuelan we don't want you in our country",
                            "damn venezuelan",
                            "Venezuelans are welcome",
                            "Venezuela is a big country", 
                            "Venezuelans are not wanted in any country", 
                            "it is time for them to leave my country"]})
"""
tweets = df.text_traducido
tweets_origin = df.text_original

print(tweets.head())
##vectorizacion de los tweets
#importar vectorizacion
vector_tfidf = joblib.load('C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/labeled_data_clean_tfidf.pkl')

vector_tfidf = vector_tfidf.transform(tweets).todense()
print(vector_tfidf.shape)
X = vector_tfidf

"""
print(df.describe())
print(df.groupby('clase').size())

df.hist()
plt.show()

#graficar clases de los tweets 
import seaborn as sns
sns.countplot(x='clase', data=df)
print("vec_tweets shape: "+str(vec_tweets.shape))
X = vec_tweets
y = np.array(df['clase'])
print("X shape: "+str(X.shape))
"""

#leer o cargar el modelo de regresión logística anteriormente creado
modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/modelo_reg_log.pkl")
print("modelo leído: "+str(modelo))

#clasificacion del dataset sin etiquetar

predicciones = modelo.predict(X)
print(predicciones)
df_temp = pd.DataFrame(columns=["text_original","text_traducido", "clase"])
for i, prediccion in enumerate(predicciones): df_temp.loc[len(df_temp)] = [tweets_origin[i],tweets[i],prediccion]
#exportar dataset etiquetado
df_temp.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_results_orig_trad.xlsx")
df_temp.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_results_orig_trad.csv")

print(df_temp.describe())
print(df_temp.groupby('clase').size()) #contador de clases
import seaborn as sns
sns.countplot(x="clase", data=df_temp)


