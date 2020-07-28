# -*- coding: utf-8 -*-

import pandas as pd
from googletrans import Translator

traductor = Translator()
idioma_orig ="es"
idioma_dest = "en"

##Lectura de dataset
df = pd.read_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_clean_es_nodupl - copia.xlsx", error_bad_lines=False)
print("Parte del texto leído: "+str(df.head(10)))
tweets = df.text
df_Transl = pd.DataFrame(columns=["text_orignal", "text_traducido"])

#metodo de traduccion recorriendo la lista de tweets
for i, tweet in enumerate(tweets):
    traduccion = traductor.translate(tweet, src=idioma_orig, dest=idioma_dest)
    #print(traduccion.origin, " -> ", traduccion.text)
    cc=0
    #en caso de traducirse, lo reintenta varias veces
    while (traduccion.origin == traduccion.text):
        print("no se pudo traducir")
        traductor = Translator(service_urls=[
              'translate.google.com',
              'translate.google.co.kr',
              'translate.google.com.ec',
              'translate.google.com.mx',
              'translate.google.com.uy',
              'translate.google.cn',
            ])
        traduccion = traductor.translate(tweet, src=idioma_orig, dest=idioma_dest)
        
        cc+=1
        if(cc > 5): break
    #verifica si el texto se tradujo
    if(traduccion.origin == traduccion.text): 
        print("Se intentó traducir:"+str(traduccion.origin) +" pero devolvió: " +str(traduccion.text))
        print("Todo el proceso se detuvo en el indice: "+str(i)+" y el tweet: "+str(tweet))
        break
    
    #agrega texto traducido a un dataframe
    df_Transl.loc[i] = [traduccion.origin, traduccion.text]

#Guargar archivo

df_Transl.to_excel("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_clean_es_nodupl_en_es_test.xlsx", index=False)
df_Transl.to_csv("C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Trabajo_Titulacion/data/tweets_clean_es_nodupl_en_es_test.csv", index=False)