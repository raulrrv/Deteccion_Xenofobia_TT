# -*- coding: utf-8 -*-
"""

@author: Raúl Romero
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn import svm
import re      #expresiones regulares
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from sklearn import linear_model
#graficar clases de los tweets 
from scipy.sparse import csr_matrix
import seaborn as sns #crear gráficas
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
naive_bayes = MultinomialNB()
#naive_bayes = GaussianNB()

stopwords = nltk.corpus.stopwords.words("english")
#importar dataset sin clasificar, sin vectorizar
def importarDS(nombre):
    df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/Nuevos/"+nombre)
    return df

def importarDataset(nombre):
    df = pd.read_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/"+nombre)
    return df

#importar dataset vectorizado 
def importarDSVector(algoritmo, nombre):
    X_ = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_X_sm.pkl")
    y_ = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_y_sm.pkl")
    return X_, y_

#DIVIDIR DATOS
#80% para entrenamiento y 20% para validar de forma aleatoria
def dividirDatos(X_, y_):
    test_size = 0.20
    semilla = 7
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_, y_, test_size=test_size, random_state=semilla)
    return X_train, X_test, y_train, y_test

#exportar dataset vectorizado X_sm y y_sm
def exportarDSVector(X_, y_, algoritmo, nombre):
    joblib.dump(X_, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_X_sm.pkl")
    joblib.dump(y_, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_y_sm.pkl")
    print("Datos exportados")
    
def predicciones(X_, modelo_):
    predicciones = modelo_.predict(X_)
    return predicciones

def reporteClasificacion(y_, predicciones_):
    print(predicciones)
    print(accuracy_score(y_, predicciones_))
    print(classification_report(y_, predicciones_))
    
def crearModelo_NB(X_, y_):
    modelo = naive_bayes.fit(X_, y_)
    print("Modelo creado: "+str(modelo))
    return modelo
#calcular metricas de precisión con validacion cruzada
def validacion_cruzada(X_, y_, modelo):    
    k_pliegues=10
    kfold = model_selection.KFold(n_splits=k_pliegues, random_state=42, shuffle = True) #shuffle habilita el random state
    cv_results = model_selection.cross_val_score(modelo, X_, y_, cv=kfold, scoring='f1_macro')
    msg = "%s: Modelo, %s: Puntuación media %f, Puntuación estándar (%f)" % ( "", modelo_nb, cv_results.mean(), cv_results.std())
    print(msg)
    print(cv_results)

def exportarModelo(modelo_, algoritmo, nombre):
    joblib.dump(modelo_, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre)
    print(str(modelo_)+" exportado")
    
#importar modelo
def importarModelo(algoritmo, nombre):
    modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre)
    print("modelo leído: "+str(modelo))
    return modelo

#grafica 2 matriz confusion
def graficar_matConfusion(y_, predicciones_, titulo):
    #labels = ['Discursos odio', 'Ofensivo', 'Ninguno']
    labels = ['Xenófobo', 'Ofensivo', 'Otro']
    matriz_confusion = confusion_matrix(y_, predicciones_) #crea matriz de confusion
    clases = ["%s"%i for i in labels[0:len(np.unique(predicciones_))]] 
    df_matConf = pd.DataFrame(matriz_confusion, index=clases, columns=clases) #matriz de confusion con los nombres de clases
    grafica = sns.heatmap(df_matConf, cmap="coolwarm", annot=True, fmt="d") #crea la grafica y si annot es true = se muestra valores en los cuadros
    grafica.set(xlabel="\nReales", ylabel="Predicciones\n")
    plt.title(titulo)
    plt.show()
    
def exportarDSPredic(tweets_origin, tweets_, y_reglog, y_svm, predicciones, nombre):
    df_temp = pd.DataFrame(columns=["text_original","text_traducido", "clase_RegLog", "clase_SVM", "clase_NavieBayes"])
    for i, prediccion in enumerate(predicciones): df_temp.loc[len(df_temp)] = [tweets_origin[i],tweets[i],y_reglog[i], y_svm[i],prediccion]
    #exportar dataset etiquetado
    df_temp.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/Nuevos/"+nombre+".xlsx")
    df_temp.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/Nuevos/"+nombre+".csv")
    print("Datasets exportados")

def importarVectorTFIDF(nombre):
    vector_tfidf_ = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+nombre)
    return vector_tfidf_
###########
def transformTFIDF(vector_tfidf_, tweets_):
    vector_tfidf_ = vector_tfidf_.transform(tweets_)
    return vector_tfidf_

def fitTFIDF(tweets_):
    vectorizer = TfidfVectorizer(
    ngram_range=(1, 3), #analiza hasta 3 palabras juntas
    stop_words=stopwords, 
    max_features=7666,
    min_df=5,
    max_df=0.7
    )
    vector_tfidf_fit_ = vectorizer.fit(tweets_)
    print(vector_tfidf_fit_)
    return vector_tfidf_fit_

def exportarDSPredic(tweets_origin, tweets_, y_reglog, predicciones, nombre):
    df_temp = pd.DataFrame(columns=["text_original","text_traducido", "clase_RegLog", "clase_SVM", "clase_NavieBayes"])
    for i, prediccion in enumerate(predicciones): df_temp.loc[len(df_temp)] = [tweets_origin[i],tweets[i],y_reglog[i], prediccion, ""]
    #exportar dataset etiquetado
    df_temp.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/"+nombre+".xlsx")
    df_temp.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/"+nombre+".csv")
    print("Datasets exportados")

def exportarModelo(modelo_, algoritmo, nombre):
    joblib.dump(modelo_, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre)
    print(str(modelo_)+" exportado")
    
#calcular metricas de precisión con validacion cruzada
def validacion_cruzada(X_, y_, modelo_):    
    k_pliegues=10
    kfold = model_selection.KFold(n_splits=k_pliegues, random_state=42, shuffle = True) #shuffle habilita el random state
    cv_results = model_selection.cross_val_score(modelo_, X_, y_, cv=kfold, scoring='f1_macro')
    msg = "%s: Puntuación media %f, Puntuación estándar (%f)" % ('Logistic Regression', cv_results.mean(), cv_results.std())
    print(msg)
    print(cv_results)
    return cv_results

#exportar vectorizacion fit 
def exportarTFIDF(vector_tfidf, nombre):
    joblib.dump(vector_tfidf, 'C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/'+nombre)
    print("Vector exportado")
    
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
    labels_x = ("Xenófobo", "Ofensivo", "Otro")
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
    
##SMOTE Técnica de sobremuestreo de minorías sintéticas  
def aplicarSMOTE(X_, y_):
    semilla = 100      
    k=10     #numero de vecinos SMOTE
    print ("X tweets vectorizados: "+str(X_))
    print ("y clases [0 1 2]: "+str(y_))
    print(y_.shape)
    sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=semilla)
    X_sm, y_sm = sm.fit_resample(X_,y_)
    return X_sm, y_sm

#graficar set de train y test
def graficarDivisionDS(train, test, titulo):
    verde = "#13CD1E"
    naranja = "#FAB62E"
    #labels = ("Training Set", "Testing Set")
    labels = ("Training Set", "Validation Set")
    slices=(len(train), len(test)) #cantidad del total de train set y test set
    colores=(verde,naranja)
    importancia = (0.1,0)
    plt.pie(slices, colors=colores, explode=importancia, labels=labels,
            autopct="%1.1f%%")
    plt.axis("equal")
    plt.title(titulo) 
    plt.legend(labels=slices)  #valores flotantes
#exportar dataset vectorizado X_sm y y_sm
def exportarDSVectorSM(X_, y_, algoritmo, nombre):
    joblib.dump(X_, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_X_sm.pkl")
    joblib.dump(y_, "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre+"_y_sm.pkl")
    print("Datos exportados")
    
modelo_nb = importarModelo("NavBayes", "modelo_labeled_smote_navbayes.pkl")
df = importarDS("tweets_results_orig_trad_clasf_rl_svm_navbayes.xlsx")
df.columns
tweets = df.text_traducido
print(tweets)
tweets_origin = df.text_original
y_reglog = df.clase_RegLog
y_svm = df.clase_SVM
y = df.clase_NavieBayes
X = transformTFIDF(fitTFIDF(tweets), tweets)
print(X)

X_sm, y_sm = importarDSVector("NavBayes", "data_labeled_smote")

#X_sm = csr_matrix.toarray(X_sm)
X_sm

X_train, X_test, y_train, y_test = dividirDatos(X_sm, y_sm)

modelo_nb = crearModelo_NB(X_sm, y_sm)
exportarModelo(modelo_nb, "NavBayes", "modelo_labeled_smote_navbayes.pkl")

validacion_cruzada(X_train, y_train, modelo_nb)

predictions = predicciones(X_test, modelo_nb_sb)
reporteClasificacion(y_test,predictions)

exportarDSVector(X_train, y_train, "NavBayes", "data_labeled_smote_train")
graficar_matConfusion(y_test, predictions,
     'Matriz de confusión del modelo de entrenamiento de \nNavie Bayes con el conjunto de test')

exportarDSPredic(tweets_origin, tweets, y_reglog, y_svm, predictions,"tweets_results_orig_trad_clasf_rl_svm_navbayes")

graficar_clases(y_sm, "Clasificación del dataset 2 por Naive Bayes\ncon las clases equilibradas")

X_sm, y_sm = aplicarSMOTE(X, predictions)
exportarDSVectorSM(X_sm, y_sm, "NavBayes", "xeno_labeled_smote_navbayes")
graficarDivisionDS(y_subTrain, y_validate, "División del dataset 2 en conjuntos de entrenamiento y validación")

X_subTrain, X_validate, y_subTrain, y_validate = dividirDatos(X_train, y_train)
exportarDSVectorSM(X_subTrain, y_subTrain, "NavBayes", "xeno_labeled_smote_subtrain_navbayes")

modelo_nb_sb = crearModelo_NB(X_subTrain, y_subTrain)
exportarModelo(modelo_nb_sb, "NavBayes", "modelo_xeno_labeled_smote_subtrain_navbayes.pkl")

exportarModelo(crearModelo_NB(X_train, y_train), "NavBayes", "modelo_xeno_labeled_smote_train_navbayes.pkl")

exportarModelo(crearModelo_NB(X_sm, y_sm), "NavBayes", "modelo_xeno_labeled_smote_navbayes_final.pkl")

