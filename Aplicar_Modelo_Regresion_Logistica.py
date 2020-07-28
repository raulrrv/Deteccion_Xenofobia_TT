# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:19:59 2020

@author: unknown
"""
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

df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_filter_text_dupl_all_dupl_en_part1.xlsx")
tweets = df.text
print(tweets)
print(tweets.shape)

##vectorizacion de los tweets
vectorizer = TfidfVectorizer()
vector_tfidf = vectorizer.fit_transform(tweets)
print(vector_tfidf.shape)

X = vector_tfidf
print(X.shape)

#leer o cargar el modelo anteriormente creado
modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/modelo_regresion_logistica.pkl")

#modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/final_model.pkl")
print("modelo leído: "+str(modelo))
predictions = modelo.predict(X)
modelo.score(X,predictions)