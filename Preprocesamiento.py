# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re      #expresiones regulares
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from sklearn import linear_model

dataset = "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_busq1y2_rep.xlsx"
df = pd.read_excel(dataset)
tweets = df.text #obtiene el texto del tweet

#preprocesamiento  de los tweets
processed_tweets = []
#re.sub("cadena a buscar", "con la que se reemplaza", cadena_leida)
url = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
menciones = '@[\w\-]+'
hashtag = '#[\w\-]+'
caracteres_especiales = r'\W'
caracter_individual=r'\s+[a-zA-Z]\s+'
caracter_individual_inicio= r'\^[a-zA-Z]\s+'
varios_espacios= r'\s+'
prefijo_b = r'^b\s+'

for tweet in tweets:
    tweet_procesado = tweet.lower()  #Convertir a minúsculas
    tweet_procesado = re.sub(menciones, ' ', tweet_procesado)
    #tweet_procesado = re.sub(hashtag, ' ', tweet_procesado)
    tweet_procesado = re.sub(url, ' ', tweet_procesado)
    tweet_procesado = re.sub(caracteres_especiales, ' ', tweet_procesado)
    tweet_procesado = re.sub(caracter_individual, ' ', tweet_procesado)
    tweet_procesado = re.sub(caracter_individual_inicio, ' ', tweet_procesado) 
    tweet_procesado = re.sub(prefijo_b, '', tweet_procesado)
    tweet_procesado = re.sub(" rt | amp ", ' ', tweet_procesado)
    tweet_procesado = re.sub(" q ", ' que ', tweet_procesado)
    tweet_procesado = re.sub(" sr ", ' señor ', tweet_procesado)
    tweet_procesado = re.sub(" x ", ' por ', tweet_procesado)
    tweet_procesado = re.sub(" d ", ' de ', tweet_procesado)
    tweet_procesado = re.sub(" xq ", ' porque ', tweet_procesado)
    tweet_procesado = re.sub(varios_espacios, ' ', tweet_procesado, flags=re.I)
    
    processed_tweets.append(tweet_procesado)   #agregar a la lista de tweets procesados

#crear y guardar dataset etiquetado limpio
tweets = processed_tweets
df_temp = pd.DataFrame(columns=["text"])
for tweet in tweets: df_temp.loc[len(df_temp)] = [tweet]

df_temp.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_clean_es.xlsx")
df_temp.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_clean_es.csv")
print(df_temp.describe())

