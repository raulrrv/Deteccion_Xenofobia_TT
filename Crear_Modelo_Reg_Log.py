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
#leer dataset smoted o equilibrado (está vectorizado)

df = pd.read_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean_smoted_k8.csv")
vec_tweets = df.tweet

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

#creamos el modelo                                  
modelo = linear_model.LogisticRegression(solver='lbfgs', 
            max_iter = 400).fit(X,y) 
#modelo_reg = modelo.fit(X,y)                     #hacemos que se ajuste (fit) a nuestro conjunto de entradas X
                                    #y salidas ‘y’
#print("modelo creado: "+str(modelo_reg))
#predicciones
predictions = modelo.predict(X)
print(predictions)
print(predictions.shape)
report = classification_report(y, predictions)
print(report)


#revisar la validez del modelo, su exactitud
#devuelve la precisión media de las predicciones
modelo.score(X,y)
print(accuracy_score(y, predictions))

#matriz de confusión
print(confusion_matrix(y, predictions))


#exportar el modelo
#joblib.dump(modelo, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/modelo_regresion_logistica.pkl")

#leer o cargar el modelo anteriormente creado
#modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/modelo_regresion_logistica.pkl")
#print("modelo leído: "+str(modelo))

##VALIDACION DEL MODELO
#la primera validación será usando 
#el mismo dataset etiquetado para el train y test
#80% para entrenamiento y 20% para validar de forma aleatoria
test_size = 0.20
semilla = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=semilla)
print(classification_report(y, predictions))
modelo.score(X_test, Y_test)

#calcular metricas de precisión con validacion cruzada
"""
k_pliegues=10
kfold = model_selection.KFold(n_splits=k_pliegues, random_state=semilla)
#cv_results = model_selection.cross_val_score(modelo, X_train, Y_train, cv=kfold, scoring='accuracy')
cv_results = model_selection.cross_val_score(modelo, X_train, Y_train, cv=kfold, scoring='f1_macro')
msg = "%s: %f (%f)" % ('Logistic Regression', cv_results.mean(), cv_results.std())
msg2 = "%s: %f (%f)" % ('Logistic Regression', cv_results.mean(), cv_results.std())
print(msg2)
"""

#clasificamos del set-test
predictions = modelo.predict(X_test)
print(accuracy_score(Y_test, predictions))
