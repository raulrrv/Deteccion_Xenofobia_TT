# -*- coding: utf-8 -*-
"""

@author: Raúl Romero
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
import re      #expresiones regulares
import nltk
import seaborn as sns

stopwords = nltk.corpus.stopwords.words("english")

#lbfgs: Algoritmo de Broyden – Fletcher – Goldfarb – Shanno de memoria limitada
#leer dataset sin etiquetas
#df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_clean_es_nodupl_en_complet.xlsx")
#leer dataset 2 etiquetado
#df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_results_orig_trad_.xlsx")
#leer dataset 2 etiquetado y equilibrado
df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/xeno_data_reglog_smoted_k13.xlsx")
#leer dataset 1 etiquetado por crowdsourcing (limpio)
#df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean.xlsx")

"""
df = pd.DataFrame({"text": ["Venezuelan we don't want you in our country",
                            "damn venezuelan",
                            "Venezuelans are welcome",
                            "Venezuela is a big country", 
                            "Venezuelans are not wanted in any country", 
                            "it is time for them to leave my country"]})
"""
df.columns
tweets = df.tweet
X = tweets
#tweets = df.text_traducido

tweets_origin = df.text_original
y = df.clase
print(df.head()) #imprime las 5 primeras filas
print(y.shape) #devuelve el numero total de filas 
print(tweets)
print(X.head())
#print(tweets_origin)

#graficar 1 - clases de los tweets 
sns.countplot(x='clase', data=df)

#graficar 2 - clases de los tweets 
def graficar_clases(y_, titulo):
    count_clases = pd.value_counts(df["clase"])  #devuelve el numero total de cada clase
    porc0 = round(float((count_clases[0] / y_.shape)*100), 2) #obtiene los porcentajes de cada clase
    porc1 = round(float((count_clases[1] / y_.shape)*100), 2)
    porc2 = round(float((count_clases[2] / y_.shape)*100), 2)
    
    verde = "#13CD1E"
    naranja = "#FAB62E"
    rojo = "#F73B15"
    colores=(rojo, naranja, verde)
    plt.figure()
    clases_x= [0,1,2]
    labels_x = ("Xenófobo", "Ofensivo", "Otro")
    valores_y = (count_clases[0], count_clases[1],count_clases[2])
    barras = plt.bar(clases_x, valores_y, align='center', color=colores, edgecolor='none')
    ax = plt.axes()
    ax.set_xticks(clases_x)  #posiciones en eje X
    ax.set_xticklabels(labels_x)  #etiquetas en eje X
    #ax.set_yticklabels([])  #oculta valores de y
    plt.xlabel('\nSentimientos')
    plt.ylabel('Tweets\n')
    
    for i, n in enumerate(valores_y):   #imprime numero de tweets encima de cada barra, de acuerdo a su clase
        ax.text(i, n + 10, n, ha='center', va='bottom')
    
    plt.title(titulo)     
    handles = barras[:3]
    plt.legend(handles, ["0: "+labels_x[0]+"  "+str(porc0)+"%", 
                         "1: "+labels_x[1]+"  "+str(porc1)+"%",
                         "2: "+labels_x[2]+"  "+str(porc2)+"%"])  #imprime leyenda en gráfico
    plt.show()
graficar_clases(y, "Dataset 2 con clases equilibradas")

#tweets.hist()
##vectorizacion de los tweets

##vectorizacion de los tweets
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3), #analiza hasta 3 palabras juntas
    stop_words=stopwords, 
    max_features=7666,
    min_df=5,
    max_df=0.7
    )

vector_tfidf = vectorizer.fit_transform(tweets)
#print(vector_tfidf.shape)

#usar fit para luego exportar esta vectorizacion
vector_tfidf = vectorizer.fit(tweets)
vector_aux = vectorizer.fit(tweets)
vector_aux = vector_aux.transform(tweets).todense()
X= vector_aux
""""""#exportar vectorizacion fit
#joblib.dump(vector_tfidf, 'C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/xeno_labeled_data_clean_tfidf.pkl')
#importar vectorizacion fit
vector_tfidf = joblib.load('C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/xeno_labeled_data_clean_tfidf.pkl')
#importar vectorizacion fit
vector_tfidf = joblib.load('C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/labeled_data_clean_tfidf.pkl')

vector_tfidf = vector_tfidf.transform(tweets).todense()
print(vector_tfidf.shape)
X = vector_tfidf

#importar modelo SMOTE
modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/smote_modelo_reg_log.pkl")
print("modelo leído: "+str(modelo))
predicciones = modelo.predict(X)

print(predicciones)
print(pd.value_counts(predicciones))

#Crear y exportar modelo de nuevo dataset
modelo_xeno = linear_model.LogisticRegression(solver='lbfgs', max_iter = 400).fit(X,predicciones) 
#exportar el modelo
joblib.dump(modelo_xeno, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/modelo_xeno_reg_log.pkl")

#Dividir datos
#80% para entrenamiento y 20% para testear de forma aleatoria
test_size = 0.20
semilla = 7
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=semilla)
len_X_train = X_train.shape[0]
len_X_test = X_test.shape[0]

print(X_train.shape[0])
print(X_test.shape[0])
#graficar set de train y test
verde = "#13CD1E"
naranja = "#FAB62E"
labels = ("Training Set", "Testing Set")
slices=(len_X_train, len_X_test)
colores=(verde,naranja)
importancia = (0.1,0)
plt.pie(slices, colors=colores, explode=importancia, labels=labels,
        autopct="%1.1f%%")
plt.axis("equal")
plt.title("División del dataset 2 en conjuntos de entrenamiento y prueba") 
plt.legend(labels=slices)  #valores flotantes

#80% para entrenamiento y 20% para validar de forma aleatoria
test_size = 0.20
semilla = 7
X_train_, X_validate, y_train_, y_validate = model_selection.train_test_split(X_train, y_train, test_size=test_size, random_state=semilla)
len_X_train_ = X_train_.shape[0]
len_X_validate = X_validate.shape[0]

#graficar división de train y validation
print(X_train_.shape[0])
print(X_validate.shape[0])
#graficar set de train y test
verde = "#13CD1E"
naranja = "#FAB62E"
labels = ("Training Set", "Validation Set")
slices=(len_X_train_, len_X_validate)
colores=(verde,naranja)
importancia = (0.1,0)
plt.pie(slices, colors=colores, explode=importancia, labels=labels,
        autopct="%1.1f%%")
plt.axis("equal")
plt.title("División del dataset 2 en conjuntos de entrenamiento y validación") 
plt.legend(labels=slices)  #valores flotantes

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

#importar modelo de regresión logística anteriormente creado
modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/smote_xeno_modelo_train_reg_log.pkl")
modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/modelo_smote_labeled_reg_log.pkl")

#print("modelo leído: "+str(modelo))
predicciones = modelo_xeno.predict(X)
print(predicciones)
print(accuracy_score(y, predicciones))
print(predicciones.shape)
report = classification_report(y, predicciones)
print(report)



#Crear el modelo de regresión logística                                  
modelo_xeno_train = linear_model.LogisticRegression(solver='lbfgs', max_iter = 400).fit(X_train,y_train) 
#exportar el modelo
joblib.dump(modelo_xeno_train, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/xeno_modelo_train_reg_log.pkl")

predicciones = modelo_xeno_train.predict(X_test)
print(predicciones)
print(predicciones.shape)
report = classification_report(y_test, predicciones)
print(report)


#graficar la matriz confusion
labels = ['Xenofobia', 'Ofensivo', 'Otro']
matriz_confusion = confusion_matrix(y_test, predicciones)
print(matriz_confusion)
plt.matshow(matriz_confusion) #grafica matriz
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(matriz_confusion)
plt.title('Matriz de confusión del modelo de Regresión Logística\n')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicción')
plt.ylabel('Reales')
plt.show()

df_temp = pd.DataFrame(columns=["text_original","text_traducido", "clase"])
for i, prediccion in enumerate(predicciones): df_temp.loc[len(df_temp)] = [tweets_origin[i],tweets[i],prediccion]
#exportar dataset etiquetado
df_temp.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_results_orig_trad.xlsx")
df_temp.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_results_orig_trad.csv")

#print(df_temp.describe()) #Generar estadísticas descriptivas
print(df_temp.groupby('clase').size()) #contador de clases
#graficar barras de las clases según la predicción
sns.countplot(x="clase", data=df_temp)



