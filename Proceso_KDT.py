# -*- coding: utf-8 -*-
"""

@author: Raúl Romero
"""
import pandas as pd
import numpy as np
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
import seaborn as sns #crear gráficas
from matplotlib import pyplot as plt
stopwords = nltk.corpus.stopwords.words("english")

##lectura de dataset de entrenamiento sin limpiar
df = pd.read_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data.csv")
#lectura dasate 2 equilibrado 
#df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/xeno_data_reglog_smoted_k13.xlsx")

tweets = df.tweet  #obtener texto de tuits
y = df.clase  #obtener columna de clases

#graficar clases de los tweets 
y.hist()   
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
for i, tweet in enumerate(tweets): df_temp.loc[len(df_temp)] = [tweet,y[i]]
df_temp.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean.xlsx")
df_temp.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean.csv")
print(df_temp.describe())
print(df_temp.groupby('clase').size()) #contador de clases

def graficar_clases(y_, titulo): 
    #graficar 2 - clases de los tweets 
    count_clases = pd.value_counts(y_)  #devuelve el numero total de cada clase
    porc0 = round(float((count_clases[0] / y_.shape)*100), 2) #obtiene los porcentajes de cada clase
    porc1 = round(float((count_clases[1] / y_.shape)*100), 2)
    porc2 = round(float((count_clases[2] / y_.shape)*100), 2)
    
    verde = "#13CD1E"
    naranja = "#FAB62E"
    rojo = "#F73B15"
    colores=(rojo, naranja, verde)
    plt.figure()
    clases_x= [0,1,2]
    labels_x = ("Discursos Odio", "Ofensivo", "Ninguno")
    #labels_x = ("Xenofobia", "Ofensivo", "Otro")
    valores_y = (count_clases[0], count_clases[1],count_clases[2])
    barras = plt.bar(clases_x, valores_y, align='center', color=colores, edgecolor='none')
    ax = plt.axes()
    ax.set_xticks(clases_x)  #posiciones en eje X
    ax.set_xticklabels(labels_x)  #etiquetas en eje X
    #ax.set_yticklabels([])  #oculta valores de y
    plt.xlabel('\nSentimientos')
    plt.ylabel('Tweets')
    
    for i, n in enumerate(valores_y):   #imprime numero de tweets encima de cada barra, de acuerdo a su clase
        ax.text(i, n + 10, n, ha='center', va='bottom')
    
    plt.title(titulo)     
    handles = barras[:3]
    plt.legend(handles, ["0: "+labels_x[0]+"  "+str(porc0)+"%", 
                         "1: "+labels_x[1]+"  "+str(porc1)+"%",
                         "2: "+labels_x[2]+"  "+str(porc2)+"%"])  #imprime leyenda en gráfico
    plt.show()
    
graficar_clases(y, "Desequilibrio de clases de dataset\nclasificado por crowdsourcing")

##lectura de dataset de entrenamiento y test limpio
#df = pd.read_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/labeled_data_clean.csv")
#tweets = df.text  #obtener texto de tuits
df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_results_orig_trad.xlsx")
tweets = df.text_traducido
clases = df.clase  #obtener columna de clases
##vectorizacion de los tweets
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3), #analiza hasta 3 palabras juntas
    stop_words=stopwords, 
    max_features=7000,
    min_df=5,
    max_df=0.7
    )
vector_tfidf = vectorizer.fit_transform(tweets)
print(vector_tfidf)
print(vector_tfidf.shape)


#usar fit para luego exportar esta vectorizacion
"""
"""
vector_tfidf = vectorizer.fit(tweets)
print(vector_tfidf.shape)

"""#exportar vectorizacion fit
joblib.dump(vector_tfidf, 'C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/labeled_data_clean_tfidf.pkl')
#importar vectorizacion fit
vector_tfidf = joblib.load('C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/labeled_data_clean_tfidf.pkl')
print(vector_tfidf.shape)
"""

##SMOTE Técnica de sobremuestreo de minorías sintéticas
semilla = 100      
k=10     #numero de vecinos SMOTE
#X = df.loc[:, df.columns != "clase"]    #todas las columnas a excepcion de la clase
#X = vector_tfidf
#y = clases    #obtiene los valores de las clases
                #(.iloc) Selección de datos por números de fila 
                #(.loc) Seleccionar datos por etiqueta o por un enunciado condicional 
print ("X tweets vectorizados: "+str(X))
print ("y clases [0 1 2]: "+str(y))
print(y.shape)
#sampling_strategy = {0: 10000, 1: 19190, 2: 10000}
#sampling_strategy = {0: 14392, 1: 19190, 2: 14392}
#sampling_strategy = {0: 9595, 1: 19190, 2: 9595}
sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=semilla)
X_sm, y_sm = sm.fit_resample(X,y)
#exportar modelo X_sm y y_sm
joblib.dump(X_sm, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_xeno_smote_reglog_X_sm.pkl")
joblib.dump(y_sm, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_xeno_smote_reglog_y_sm.pkl")

#importar modelo X_sm y y_sm
""""""
#X_sm = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_labeled_smote_X_sm.pkl")
#y_sm = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_labeled_smote_y_sm.pkl")

X_sm = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/nuevos/data_smote_X_sm_full_feats.pkl")
y_sm = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/nuevos/data_smote_y_sm_full_feats.pkl")

print ("X_sm: "+str(X_sm))
print ("y_sm: "+str(y_sm))
print("Filas, Columnas de x: "+str(X_sm.shape))
print("Filas, Columnas de y: "+str(y_sm.shape))
y_sm.hist() 

graficar_clases(y_sm, "Dataset clasificado y equilibrado\n")
#gráficar clases
"""
plt.show()      
#graficar clases de los tweets 
df_graf = pd.concat([pd.DataFrame(X_sm), pd.DataFrame(y_sm)], axis=1)
df_graf.columns = ["tweet","clase"]
sns.countplot(x="clase", data=df_graf)
"""
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
""""""
df = pd.concat([pd.DataFrame(X_sm), pd.DataFrame(y_sm)], axis=1)
df.columns = ["tweet","clase"]
df.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/xeno_data_reglog_smoted_k13.xlsx", index=False, encoding="utf-8")
df.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/xeno_data_reglog_smoted_k13.csv", index=False, encoding="utf-8")

print(y_sm.hist)

#importar dataset equilibrado y vectorizado X_sm y y_sm
""""""
X_sm = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_xeno_smote_reglog_X_sm.pkl")
y_sm = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/data_xeno_smote_reglog_y_sm.pkl")
#importar modelo
def importarModelo(algoritmo, nombre):
    modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre)
    print("modelo leído: "+str(modelo))
    return modelo

#exportar dataset vectorizado X_sm y y_sm
def exportarDSVector(X_, y_, algoritmo, nombre):
    joblib.dump(X_sm, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_X_sm.pkl")
    joblib.dump(y_sm, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_y_sm.pkl")

#importar dataset vectorizado 
def importarDSVector(algoritmo, nombre):
    X_ = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_X_sm.pkl")
    y_ = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_y_sm.pkl")
    return X_, y_

#calcular metricas de precisión con validacion cruzada
def validacion_cruzada(X_, y_, modelo):    
    k_pliegues=10
    kfold = model_selection.KFold(n_splits=k_pliegues, random_state=42)
    #cv_results = model_selection.cross_val_score(modelo, X_train, Y_train, cv=kfold, scoring='accuracy')
    cv_results = model_selection.cross_val_score(modelo, X_, y_, cv=kfold, scoring='f1_macro')
    msg = "%s: Puntuación media %f, Puntuación estándar (%f)" % ('Logistic Regression', cv_results.mean(), cv_results.std())
    print(msg)
    cv_results
    
def predicciones(X_, modelo):
    predicciones = modelo.predict(X_)
    return predicciones

def reporteClasificacion(y_, predicciones):
    print(predicciones)
    print(accuracy_score(y_, predicciones))
    print(classification_report(y_, predicciones))

#DIVIDIR DATOS
#80% para entrenamiento y 20% para validar de forma aleatoria
def dividirDatos(X_, y_):
    test_size = 0.20
    semilla = 7
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_, y_, test_size=test_size, random_state=semilla)
    return X_train, X_test, y_train, y_test

def exportarModelo(modelo_, algoritmo, nombre):
    joblib.dump(modelo_, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre)
    print(str(modelo_)+" exportado")
#Como dice, % s (de cadena) es para reemplazar una cadena, % f (de flotante) es para reemplazar un flotante, y % d (de entero) es para reemplazar un entero
#grafica 2 matriz confusion
def graficar_matConfusion(y_, predicciones_, titulo):
    labels = ['Xenófobo', 'Ofensivo', 'Otro']
    matriz_confusion = confusion_matrix(y_, predicciones_) #crea matriz de confusion
    clases = ["%s"%i for i in labels[0:len(np.unique(predicciones_))]] 
    df_matConf = pd.DataFrame(matriz_confusion, index=clases, columns=clases) #matriz de confusion con los nombres de clases
    grafica = sns.heatmap(df_matConf, cmap="coolwarm", annot=True, fmt="d") #crea la grafica y si annot es true = se muestra valores en los cuadros
    grafica.set(xlabel="\nReales", ylabel="Predicciones\n")
    plt.title(titulo)
    plt.show()
    
def crearModelo_RegLog(X_, y_):
    modelo = linear_model.LogisticRegression(solver='lbfgs', max_iter = 400).fit(X_, y_)  
    print("Modelo creado: "+modelo)
    return modelo

X_sm, y_sm = importarDSVector("RegLog", "data_xeno_smote_reglog")
print(X_sm)
X_train, X_test, y_train, y_test = dividirDatos(X_sm, y_sm)
X_sub_train, X_validate, y_sub_train, y_validate = dividirDatos(X_train, y_train)
modelo = crearModelo_RegLog(X_sm, y_sm)
exportarModelo(modelo, "RegLog", "modelo_xeno_smote_reglog_final.pkl")

exportarDSVector(X_sub_train, y_sub_train, "RegLog", "xeno_smote_subtrain_reglog")
exportarDSVector(X_validate, y_validate, "RegLog", "xeno_smote_validate_reglog")
modelo = importarModelo("RegLog", "smote_xeno_subtrain_modelo_reg_log.pkl")

prediccions = predicciones(X_test, modelo)
reporteClasificacion(y_test, prediccions)
graficar_matConfusion(y_test, prediccions, 
                      'Matriz de confusión del modelo de entrenamiento de \nRegresión Logística con el conjunto de test')

validacion_cruzada(X_sub_train, y_sub_train, modelo)

#Crear el modelo de regresión logística del subTrain Set                                 
#modelo = linear_model.LogisticRegression(solver='lbfgs', max_iter = 400).fit(X_sub_train,y_sub_train) 

#modelo_reg = modelo.fit(X,y)       #hacemos que se ajuste (fit) a nuestro conjunto de entradas X
                                    #y salidas ‘y’
modelo = importarModelo("RegLog", "smote_xeno_subtrain_modelo_reg_log.pkl")

X_train, X_test, y_train, y_test = dividirDatos(X_sm,y_sm)
X_sub_train, X_validate, y_sub_train, y_validate = dividirDatos(X_train,y_train)

X_sub_train, y_sub_train = importarDSVector("RegLog", "xeno_smote_subtrain_reglog")
predicciones = predicciones(X_validate, modelo)
reporteClasificacion(y_test, predicciones)

#exportar el modelo

#joblib.dump(modelo, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/smote_xeno_modelo_reg_log.pkl")
#joblib.dump(modelo, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/modelo_smote_labeled_reg_log.pkl")
joblib.dump(modelo, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/modelo_smote_labeled_reg_log_full_feats.pkl")
#exportar el modelo de subTrain set
joblib.dump(modelo, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/smote_xeno_subtrain_modelo_reg_log.pkl")

#graficar set de train y test
verde = "#13CD1E"
naranja = "#FAB62E"
labels = ("Training Set", "Testing Set")
slices=(len(y_train), len(y_test)) #cantidad del total de train set y test set
colores=(verde,naranja)
importancia = (0.1,0)
plt.pie(slices, colors=colores, explode=importancia, labels=labels,
        autopct="%1.1f%%")
plt.axis("equal")
plt.title("División del dataset en nuevos conjuntos de entrenamiento y prueba") 
plt.legend(labels=slices)  #valores flotantes

#graficar SUB set de train y validate
verde = "#13CD1E"
naranja = "#FAB62E"
labels = ("Training Set", "Validation Set")
slices=(len(y_sub_train), len(y_validate)) #cantidad del total de train set y test set
colores=(verde,naranja)
importancia = (0.1,0)
plt.pie(slices, colors=colores, explode=importancia, labels=labels,
        autopct="%1.1f%%")
plt.axis("equal")
plt.title("División del Training Set \nen nuevos conjuntos de entrenamiento y validación") 
plt.legend(labels=slices)  #valores flotantes


"""
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
"""

#Crear y testear el modelo de regresión logística, desde el dataset train                                  
"""
modelo_train = linear_model.LogisticRegression(solver='lbfgs', max_iter = 400).fit(X_train,y_train) 
predicciones = modelo_train.predict(X_test)"""

#Crear y testear el modelo de regresión logística, desde el dataset sub train hacia sub_test                                 
modelo_sub_train = linear_model.LogisticRegression(solver='lbfgs', max_iter = 400).fit(X_sub_train, y_sub_train)  
predicciones = modelo_sub_train.predict(X_validate)
print(predicciones)
print(accuracy_score(y_validate, predicciones))
print(classification_report(y_validate, predicciones))

#testear el modelo de regresión logística, desde el dataset sub_train hacia test_principal                                 
predicciones = modelo_sub_train.predict(X_test)
print(predicciones)
print(accuracy_score(y_test, predicciones))
print(classification_report(y_test, predicciones))

#graficar 1 a matriz de confusión
labels = ['Xenofobia', 'Ofensivo', 'Otro']
matriz_confusion = confusion_matrix(y_test, predicciones)
print(matriz_confusion)
plt.matshow(matriz_confusion) #grafica matriz
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(matriz_confusion)
plt.title('Matriz de confusión del modelo de Regresión Logística \nvalidado con el Conjunto Principal de Test\n')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Reales')
plt.ylabel('Predicciones')
plt.show()


graficar_matConfusion()

#exportar el modelo
#joblib.dump(modelo_train, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/smote_xeno_modelo_train_reg_log.pkl")
