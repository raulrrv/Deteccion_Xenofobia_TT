# -*- coding: utf-8 -*-
"""
@author: Raúl Romero

"""
from twitterscraper import query_tweets
import datetime as dt
import pandas as pd
ruta_OUT=""
 
def recopilarTweets():
    global ruta_OUT
    begin_date = dt.date(2016,1,1)
    end_date = dt.date(2019,12,31)
    
    limit = 20000
    lang = 'spanish'
    
    #tweets = query_tweets("venezolanos ecuador (racismo OR fuera OR odio OR chavistas OR asesinos OR xenofobia OR inmigrantes OR migrantes OR migración)", begindate=begin_date, enddate = end_date, limit = limit, lang= lang)
    #tweets = query_tweets("venezolanos ecuador", begindate=begin_date, enddate = end_date, limit = limit, lang= lang)
    #tweets = query_tweets("venezolanos ec (racismo OR fuera OR odio OR chavistas OR asesinos OR xenofobia OR inmigrantes OR migrantes OR migración)", begindate=begin_date, enddate = end_date, limit = limit, lang= lang)
    #tweets = query_tweets("(venezuela OR venezolano OR venezolana OR venezolanos OR chamo OR chama OR chamos OR chamas) ?geocode:-1.490348,-78.457462,500km")
    #tweets = query_tweets("(venezuela OR venezolano OR venezolana OR venezolanos OR chamo OR chama OR chamos OR chamas) ?geocode:-1.490348,-78.457462,500km")
    
    ## BUSQUEDA 1 
    tweets = query_tweets("((ecuador) AND (venezolano OR venezolana OR venezolanos OR chamo OR chama OR chamos OR chamas)) AND (racismo OR fuera OR odio OR asesinos OR xenofobia OR inmigrantes OR migrantes OR migración)", begindate=begin_date, enddate = end_date)
    
    listTweets = pd.DataFrame(tuit.__dict__ for tuit in tweets)  #__dict__ contiende los atributos del objeto
    
    ## BUSQUEDA 2 con ubicacion
    tweets = query_tweets("(venezolano OR venezolana OR venezolanos OR chamo OR chama OR chamos OR chamas) AND (racismo OR fuera OR odio OR asesinos OR xenofobia OR inmigrantes OR migrantes OR migración) ?geocode:-1.490348,-78.457462,500km", begindate=begin_date, enddate = end_date)

    listTweets = listTweets.append(pd.DataFrame(tuit.__dict__ for tuit in tweets))  #__dict__ contiende los atributos del objeto
    
   # ruta=input("Ingrese nombre del dataset de salida: ") 
    ruta_OUT = "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Recopilacion_informacion/tweets_bus1y2.csv"
    listTweets.to_csv(ruta_OUT)
    
    ruta_OUT = "C:/Users/unknown/OneDrive/UNL/X/Trabajo de Titulacion/Recopilacion_informacion/tweets_bus1y2.xlsx"
    listTweets.to_excel(ruta_OUT)
    
    return listTweets

def getTweetsCSV():
    dataset_completo = pd.read_csv(ruta_OUT, header=0)
    return dataset_completo

recopilarTweets()