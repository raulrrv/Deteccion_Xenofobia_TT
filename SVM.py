# -*- coding: utf-8 -*-
"""

@author: Raúl Romero
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
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
import seaborn as sns #crear gráficas
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()

stopwords = nltk.corpus.stopwords.words("english")
#importar dataset clasificado, sin vectorizar - excel
def importarDSExcel(nombre):
    df = pd.read_excel("E:/Tutoriales_TT/3 Limpieza/"+nombre)
    return df

#importar dataset clasificado, sin vectorizar - csv
def importarDScsv(nombre):
    df = pd.read_csv("E:/Tutoriales_TT/3 Limpieza/"+nombre)
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
def exportarDSVectorSM(X_, y_, algoritmo, nombre):
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
    
#importar modelo
def importarModelo(algoritmo, nombre):
    modelo = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+algoritmo+"/"+nombre)
    print("modelo leído: "+str(modelo))
    return modelo

#Como dice, % s (de cadena) es para reemplazar una cadena, % f (de flotante) es para reemplazar un flotante, y % d (de entero) es para reemplazar un entero
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
    
print(len(np.unique(predictions)))  
def importarVectorTFIDF(nombre):
    vector_tfidf_ = joblib.load("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/modelo/"+nombre)
    return vector_tfidf_

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

def crearModelo_SVM(X_, y_):
    modelo = svm.SVC(kernel='linear', C = 1).fit(X_, y_)  
    print("Modelo creado: "+str(modelo))
    return modelo

def exportarModelo(modelo_, algoritmo, nombre):
    joblib.dump(modelo_, "E:/Tutoriales_TT/6 Modelos/"+algoritmo+"/"+nombre)
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
    labels = ("Training Set", "Validation Set")
    slices=(len(train), len(test)) #cantidad del total de train set y test set
    colores=(verde,naranja)
    importancia = (0.1,0)
    plt.pie(slices, colors=colores, explode=importancia, labels=labels,
            autopct="%1.1f%%")
    plt.axis("equal")
    plt.title(titulo) 
    plt.legend(labels=slices)  #valores flotantes

def crearModelo_NB(X_, y_):
    modelo = naive_bayes.fit(X_, y_)
    print("Modelo creado: "+str(modelo))
    return modelo

def crearModelo_RForest(X_, y_):
    modelo_ranForest = RandomForest.fit(X_, y_)
    print("Modelo creado: "+str(modelo_ranForest))
    return modelo_ranForest
###########################################################

support_vectors = modelo_svm.support_vectors_

X_train_ = csr_matrix.toarray(X_train)	
support_vectors_ = csr_matrix.toarray(support_vectors)
plt.scatter(X_train_[:,0], X_train_[:,1])
plt.scatter(support_vectors_[:,0], support_vectors_[:,1], color='red')
plt.title('Grafica de matriz de confusión')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


X_test_ = csr_matrix.toarray(X_test)
y_test_ = pd.Series.to_numpy(y_test)
value=1.5
width=0.75
plot_decision_regions(X_test_, y_test_, clf=modelo_svm,
                feature_index=[0,7665],
                  filler_feature_values={0: value, 7665:value},
                  filler_feature_ranges={0: width, 7665: width},
                  legend=2)
plt.show()
X_sm, y_sm = importarDSVector("SVM", "data_labeled_smote")
y_sm
modelo = crearModelo_SVM(X_train, y_train)
exportarModelo(modelo, "SVM", "modelo_labeled_smote_train_svm.pkl")

print(tweets)
exportarTFIDF(vector_tfidf_fit, "xeno_data_clean_tfidf_.pkl")

##########################################################################
########__________VECTORIZACIÓN______###########
df = importarDScsv("labeled_data_clean2.csv")


print(stopwords)
df.columns
df.head()
df.clase.hist()

tweets = df.text
vector_tfidf_fit = fitTFIDF(tweets)
X = transformTFIDF(vector_tfidf_fit, tweets)

y=df.clase
print(y)

######_______________SMOTE__________#############

X_sm, y_sm = aplicarSMOTE(X, y)
y_sm.hist()


#####_____CREACIÓN DE MODELO SVM_____#####
X_train, X_test, y_train, y_test = dividirDatos(X_sm, y_sm)
modelo_svm = crearModelo_SVM(X_train, y_train)
predictions = predicciones(X_test, modelo_svm)

exportarModelo(modelo_svm, "SVM", "modelo_labeled_smote_train_svm.pkl")


#####_____CREACIÓN DE MODELO RandomForest_____#####
X_train, X_test, y_train, y_test = dividirDatos(X, y)
modelo_RanForest = crearModelo_RForest(X_train, y_train)

predictions = predicciones(X_test, modelo_RanForest)
print(predictions)
reporteClasificacion(y_test, predictions)

#####_____CREACIÓN DE MODELO NAVIE BAYES_____#####

modelo_navi = crearModelo_NB(X_train, y_train)

exportarModelo(modelo_navi, "RegLog_NaiveBayes", "modelo_labeled_smote_train_navi.pkl")

predictions = predicciones(X_test, modelo_navi)
print(predictions)
reporteClasificacion(y_test, predictions)

graficar_matConfusion(y_test, predictions, 'Matriz de confusión del modelo de entrenamiento de \nNavie Bayes con el conjunto de test')

#####__CREACIÓN DE MODELO REGRESION LOGÍSTICA_____#####

modelo_reglog = linear_model.LogisticRegression(solver='lbfgs', max_iter = 400).fit(X_train,y_train) 

exportarModelo(modelo_reglog, "RegLog_NaiveBayes", "modelo_labeled_smote_train_reglog.pkl")

predictions = predicciones(X_test, modelo_reglog)
print(predictions)
reporteClasificacion(y_test, predictions)

graficar_matConfusion(y_test, predictions, 'Matriz de confusión del modelo de entrenamiento de \nRandom Forest con el conjunto de test')



exportarDSVectorSM(X_train, y_train, "SVM", "xeno_labeled_smote_train_svm")

exportarDSVector(X_train, y_train, "SVM", "data_labeled_smote_train")

modelo_svm = importarModelo("SVM", "modelo_labeled_smote_svm.pkl")

predictions = predicciones(X_test, modelo_svm_sub)
print(predictions)
reporteClasificacion(y_test, predictions)

graficar_clases(y_sm, "Clasificación del dataset 2 con las clases equilibradas\n")

graficar_matConfusion(y_test, predictions, 'Matriz de confusión del modelo de entrenamiento de \nSVM con el conjunto de test')

exportarDSPredic(tweets_origin, tweets, y_reglog, predictions,"tweets_results_orig_trad_clasf_svm")

validacion_cruzada(X_train, y_train, modelo)

X_sm, y_sm = aplicarSMOTE(X, predictions)

exportarDSVectorSM(X_sm, y_sm, "SVM", "xeno_labeled_smote_svm")

graficarDivisionDS(y_subTrain, y_validate, "División del dataset 2 en conjuntos de entrenamiento y validación")

X_subTrain, X_validate, y_subTrain, y_validate = dividirDatos(X_train, y_train)
exportarDSVectorSM(X_subTrain, y_subTrain, "SVM", "xeno_labeled_smote_subtrain_svm")

modelo_svm_sub = crearModelo_SVM(X_subTrain, y_subTrain)
exportarModelo(modelo_svm_sub, "SVM", "modelo_xeno_labeled_smote_subtrain_svm.pkl")
exportarModelo(crearModelo_SVM(X_train, y_train), "SVM", "modelo_xeno_labeled_smote_train_svm.pkl")

exportarModelo(crearModelo_SVM(X_sm, y_sm), "SVM", "modelo_xeno_labeled_smote_svm_final.pkl")

#predicciones con datos inventados
df = pd.DataFrame({"text": ["Venezuelan we don't want you in our country",
                            "damn venezuelan",
                            "Venezuelans are welcome",
                            "Venezuela is a big country", 
                            "Venezuelans are not wanted in any country", 
                            "it is time for them to leave my country"]})

textos = df.text
print(textos)
vector = importarVectorTFIDF("labeled_data_clean_tfidf.pkl")
print(vector)
vector_t = transformTFIDF(vector, textos)
print(vector_t)
predictions = predicciones(vector_t, importarModelo("SVM", "modelo_xeno_labeled_smote_svm_final.pkl"))
print(predictions)
