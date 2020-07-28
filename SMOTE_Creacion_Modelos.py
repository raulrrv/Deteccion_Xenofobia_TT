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
#graficar clases de los tweets 
import seaborn as sns
stopwords = nltk.corpus.stopwords.words("english")

##lectura de dataset de entrenamiento
df = pd.read_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data.csv")
tweets = df.tweet  #obtener texto de tuits
clases = df.clase  #obtener columna de clases
#tweets=["@hola #esta ~es una I prueba de .start."]
#graficar valores iniciales del dataset
clases.hist()   
plt.show()      #gráficar clases
#graficar clases de los tweets 
sns.countplot(x='clase', data=df)

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
    tweet_procesado = re.sub(varios_espacios, ' ', tweet_procesado, flags=re.I)
    
    processed_tweets.append(tweet_procesado)   #agregar a la lista de tweets procesados

#crear y guardar dataset etiquetado limpio
tweets = processed_tweets
df_temp = pd.DataFrame(columns=["text", "clase"])
for i, tweet in enumerate(tweets): df_temp.loc[len(df_temp)] = [tweet,clases[i]]

df_temp.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean.xlsx")
df_temp.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean.csv")
print(df_temp.describe())
print(df_temp.groupby('clase').size()) #contador de clases


##lectura de dataset de entrenamiento limpio
df = pd.read_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean.csv")
tweets = df.text  #obtener texto de tuits
clases = df.clase  #obtener columna de clases
##vectorizacion de los tweets
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3), #analiza hasta 3 palabras juntas
    stop_words=stopwords, 
    max_features=30000,
    min_df=5,
    max_df=0.7
    )

vector_tfidf = vectorizer.fit_transform(tweets)
print(vector_tfidf.shape)

#usar fit para luego exportar esta vectorizacion
"""
vector_tfidf = vectorizer.fit(tweets)
print(vector_tfidf.shape)

"""
"""#exportar vectorizacion fit
joblib.dump(vector_tfidf, 'C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/labeled_data_clean_tfidf.pkl')
#importar vectorizacion fit
vector_tfidf = joblib.load('C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/labeled_data_clean_tfidf.pkl')
print(vector_tfidf.shape)
"""

##SMOTE Técnica de sobremuestreo de minorías sintéticas
semilla = 100      
k=13     #numero de vecinos SMOTE
#X = df.loc[:, df.columns != "clase"]    #todas las columnas a excepcion de la clase
X = vector_tfidf
y = clases    #obtiene los valores de las clases
                #(.iloc) Selección de datos por números de fila 
                #(.loc) Seleccionar datos por etiqueta o por un enunciado condicional 

"""
Clases:
    0: Xenofobia 
    1: Lenguaje ofensivo
    2: Ninguno
"""
print ("X tweets vectorizados: "+str(X))
print ("y clases [0 1 2]: "+str(y))
print(y.shape)

sm = SMOTE(sampling_strategy="auto", k_neighbors=k, random_state=semilla)

X_sm, y_sm = sm.fit_resample(X,y)
#exportar X_sm y y_sm
joblib.dump(X_sm, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_labeled_smote_X_sm.pkl")
joblib.dump(y_sm, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_labeled_smote_y_sm.pkl")

#importar X_sm y y_sm
X_sm = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_labeled_smote_X_sm.pkl")
y_sm = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_labeled_smote_y_sm.pkl")


print ("X_sm: "+str(X_sm))
print ("y_sm: "+str(y_sm))
print("Filas, Columnas de x: "+str(X_sm.shape))
print("Filas, Columnas de y: "+str(y_sm.shape))
y_sm.hist()   
plt.show()      #gráficar clases
#graficar clases de los tweets 
df_graf = pd.concat([pd.DataFrame(X_sm), pd.DataFrame(y_sm)], axis=1)
df_graf.columns = ["tweet","clase"]
sns.countplot(x="clase", data=df_graf)
#Graficar los puntos de datos resultante
"""
plt.title("Dataset balanced with synthetic or SMOTE'd data ({} neighbors)".format(k))
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X_sm[:, 0], X_sm[:, 1], marker='o', c=y_sm,
           s=25, edgecolor='k', cmap=plt.cm.coolwarm)
plt.show()
"""
#Transformar de vector a su texto original
"""
inverso = vectorizer.inverse_transform(X_sm)
print("Transformacion de vector a texto realizada correctamente")
print(inverso)
df = pd.DataFrame(inverso)
#df.columns = ["tweet"]
df.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_smote_inverso.xlsx", index=False, encoding="utf-8")
"""
#Guardar nuevo dataset equilibrado
"""
df = pd.concat([pd.DataFrame(X_sm), pd.DataFrame(y_sm)], axis=1)
df.columns = ["tweet","clase"]
df.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean_smoted_k8.xlsx", index=False, encoding="utf-8")
df.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean_smoted_k8.csv", index=False, encoding="utf-8")
"""

#Crear el modelo de regresión logística                                  
modelo = linear_model.LogisticRegression(solver='lbfgs', max_iter = 400).fit(X_sm,y_sm) 
#modelo_reg = modelo.fit(X,y)                     #hacemos que se ajuste (fit) a nuestro conjunto de entradas X
                                    #y salidas ‘y’
print("modelo creado: "+str(modelo))
predictions = modelo.predict(X_sm)
print("Predicciones: "+str(predictions))
print(predictions.shape)
report = classification_report(y_sm, predictions)
print(report)

#revisar la validez del modelo, su exactitud
#devuelve la precisión media de las predicciones
modelo.score(X_sm,y_sm)
print(accuracy_score(y_sm, predictions))

#matriz de confusión
print(confusion_matrix(y_sm, predictions))


#exportar el modelo
joblib.dump(modelo, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/smote_modelo_reg_log.pkl")

#importar modelo
modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/smote_modelo_reg_log.pkl")
print("modelo leído: "+str(modelo))

#Validar modelo
#80% para entrenamiento y 20% para validar de forma aleatoria
test_size = 0.20
semilla = 7
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_sm, y_sm, test_size=test_size, random_state=semilla)
predicciones = modelo.predict(X_test)
print(predicciones)
print(accuracy_score(y_test, predicciones))
print(classification_report(y_test, predicciones))

#graficarla
labels = ['Xenofobia', 'Ofensivo', 'Ninguno']
cm = confusion_matrix(y_test, predicciones)
print(cm)
plt.matshow(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Matriz de confusión del modelo de Regresión Logística\n')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicción')
plt.ylabel('Reales')
plt.show()

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

#Crear el modelo de regresión logística                                  
modelo_train = linear_model.LogisticRegression(solver='lbfgs', max_iter = 400).fit(X_train,y_train) 
predicciones = modelo_train.predict(X_test)
print(predicciones)
print(accuracy_score(y_test, predicciones))
print(classification_report(y_test, predicciones))
