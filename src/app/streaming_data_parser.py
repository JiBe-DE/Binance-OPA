import pandas as pd
import json
import asyncio
import websockets
import requests
import math
import datetime as dt

from fastapi import HTTPException

# Module MongoDB
from pymongo import MongoClient

# Module Sqlite
from sqlalchemy import create_engine, text


def get_data_streaming(token, window):
    ''' Récupérer les x (window) dernières données de streaming reçues pour le token choisi
        :param token (str)
        :param window (int) 
    '''
    #Connexion au MongoDB
    client = MongoClient(
        host='172.24.0.1',
        port=27017
    )

    # Récupération ou Création de la collection 
    db = client["binance"]

    if(token + 'Stream' not in db.list_collection_names()):
        raise HTTPException(status_code=403, detail= 'Token inconnu : ' + token)

    col = db[token + 'Stream']
    if(col.count_documents({}) < 3000):
        populate_mongo(token, col)

    pipeline = [
        {
            "$group" : {
                "_id" : "$Open_time",
                "Open_time" : {"$last":"$Open_time"},
                "Open_price" : {"$last":"$Open_price"},
                "High_price" : {"$last":"$High_price"},
                "Low_price" : {"$last":"$Low_price"},
                "Close_price" : {"$last":"$Close_price"},
                "Volume" : {"$last":"$Volume"},
                "Kline_Close_time" : {"$last":"$Kline_Close_time"},
                "Quote_asset_volume" : {"$last":"$Quote_asset_volume"},
                "Number_of_trades" : {"$last":"$Number_of_trades"},
                "TBBA_volume" : {"$last":"$TBBA_volume"},
                "TBQA_volume" : {"$last":"$TBQA_volume"},
                "Unused" : {"$last":"$Unused"}
            },
        },
        {"$limit" : window}
    ]

    df = pd.DataFrame(list(col.aggregate(pipeline=pipeline)))
    df = df.drop('_id', axis = 1)

    return df


def get_col(token, db):
    ''' Retourne la collection contenant les documents du token correspondant
        :param token (str)
        :param db (BDD Mongo) 
    '''
    if(token + 'Stream' not in db.list_collection_names()):
        db.create_collection(name = token + 'Stream', capped=True, size=5242880, max=6000)
        print("Liste des collections : ", db.list_collection_names())
    return db[token + 'Stream']


def populate_mongo(token, col):
    ''' Ajoute les dernières données historiques disponibles à la collection MongoDB
        :param token (str)
        :param col (Collection Mongo) 
    '''
    startTime = math.floor(dt.datetime.now().timestamp()*1000 - 7200000)
    r = requests.get(
        url='https://api.binance.com/api/v3/klines?symbol={TOKEN}&interval=5m&startTime={startTime}&limit=200'.format(TOKEN=token.upper(), startTime=startTime),
    )
    klines = r.json()
    klines.reverse()
    for kline in klines:
        data = {
            "Open_time": kline[0], 
            "Kline_Close_time":kline[6],
            "Open_price":kline[1],
            "Close_price":kline[4],
            "High_price":kline[2],
            "Low_price":kline[3],
            "Volume":kline[5],
            "Quote_asset_volume":kline[7],
            "Number_of_trades":kline[8],
            "TBBA_volume":kline[9],
            "TBQA_volume":kline[10],
            "Unused":kline[11]
                }
        col.insert_one(data) 