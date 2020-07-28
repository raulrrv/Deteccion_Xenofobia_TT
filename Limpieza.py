# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize #separa palabras

nltk.download('punkt')
import string   #para separar los signos de puntuación

##Limpieza de datos
#Eliminar signos de puntuación, menciones, hashtag y enlaces
df = pd.read_excel('C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Recopilacion_informacion/Limpieza/tweets_filter_text_nodupl.xlsx')
Tweets = df.text #obtiene el texto del tweet
df_temp = pd.DataFrame(columns=['text'])

for i, tweet in enumerate(Tweets):
    tweet = tweet.lower()       #convierte a minusculas
    #palabras = tweet.split()    #crea un array con todas las palabras del tweet
    palabras = word_tokenize(tweet)    #crea un array con todas las palabras del tweet
    text=""    
    for word in palabras: 
        try:
            if (word.find("#") != -1 or word.find("@") != -1 or  word.find("http") != -1 or  word.find("//") != -1  or  word.find("/") != -1 or  word.find(".com") != -1 or  word.find(".net") != -1 or  word.find(".org") != -1 or word=="..." or word == "``" or word == "•" or word == "rt" or word == "''" or word == "”" or word == "“" or word[0:1] == "¿" or word == "►" ):
                #print("Palabra eliminada: "+word)
                if(word[0:1] == "#"):    #eliminar el numeral de los hashtag
                    word = word[1:]     #descarta el primer caracter (#)
                    print("hashtag: "+str(word))
                    text = text + word+" "
                elif (word[0:1] == "¿"):
                    word = word[1:]     #descarta el primer caracter (¿)
                    print("signo: "+str(word))
                    text = text + word+" "
                    
            else:
                text = text + word+" "
        except:
            print("Error al leer palabra "+word)

    text_temp = word_tokenize(text)    #crea un array con todas las palabras del tweet  filtrado      
    #eliminar signos de puntuacion
    text_temp = list(filter(lambda token: token not in string.punctuation, text_temp)) #eliminar signos de puntuación
    txt = ""
    for w in text_temp:
        txt = txt + w + " "

    df_temp.loc[i] = [txt]

print (df_temp)

#guardar en un archivo
df_temp.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Recopilacion_informacion/Limpieza/tweets_filter_text_nodupl_clr.xlsx")
