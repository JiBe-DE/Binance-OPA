import pandas as pd
import datetime as dt
import base64
import os
import io

# Modules FastAPI
from fastapi import FastAPI, HTTPException, Response, BackgroundTasks, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from enum import Enum

# Modules SQL
import sqlite3, sqlalchemy
from sqlalchemy import create_engine, text

# Module MongoDB
from pymongo import MongoClient

# Modules project
import lstm_model
import historical_data_parser as hdp
import streaming_data_parser as sdp

# Module TensorFlow
import tensorflow as tf
from keras.models import load_model

# Matplotlib
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.dates as md
import mplfinance as mpf

# Définition des utilisateurs 
users = {
    "admin" : {
        "username": "admin",
        "password" : "admin"
    }
}

# DataFrame Column
klines_col=['Open_time', 'Open_price', 'High_price', 'Low_price', 'Close_price', 'Volume','Kline_Close_time', 'Quote_asset_volume', 'Number_of_trades', 'TBBA_volume','TBQA_volume', 'Unused']

# Tags 
api_tags = [
    {'name' : 'User', 'description' : 'services dédiés à l\'utilisateur final'},
    {'name' : 'Admin', 'description' : 'services dédiés à l\'utilisateur admin'}
]

# Initialisation de l'API
api = FastAPI(title = "API CryptoBot", description = "API de prédiction de cours des cryptomonnaies", openapi_tags = api_tags, version = "1.0")
security = HTTPBasic()

# Fonction de vérification BasicAuth
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    if not(users.get(username)) or not(credentials.password == users[username]['password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
    

# Class API 
class futures(str, Enum):
      f1 ='30m' 
      f2 = '1h' 
      f3 = '2h' 

class token_id(str, Enum):
      BTC ='btceur' 
      ETH = 'etheur' 
      BNB = 'bnbeur' 

##########
# METHODES 
##########

@api.get('/status', name = "API Status", tags = ['Admin'])
async def get_status(username: str = Depends(get_current_username)):
    ''' Retourne l'état de santé de l'API : 1 si l'API fonctionne
    '''
    return 1


@api.get('/token/sql', name = "Available tables in SQL", tags = ['Admin'])
async def get_sql_token(username: str = Depends(get_current_username)):
    ''' Retourne les tables disponibles dans la base de données SQL
    '''
    engine = create_engine('sqlite:///./historicDataBinance.db', echo=False)

    return engine.table_names()


@api.get('/token/sql/{token_id}', name = "Last 5 SQL token entries", tags = ['Admin'])
async def get_sql_token(token_id:str, username: str = Depends(get_current_username)):
    ''' Retourne les 5 dernières entrées stockées dans SQL pour le token entré en paramètre
    '''
    engine = create_engine('sqlite:///./historicDataBinance.db', echo=False)
    if(token_id not in engine.table_names()):
            raise HTTPException(status_code=403, detail= 'Token inconnu ' + token_id)
    conn = engine.connect()
    stmt = text("SELECT * FROM " + token_id + " ORDER BY Open_time DESC LIMIT 5;")
    result = conn.execute(stmt)

    return result.fetchall()


@api.post('/token/sql/update/{token_id}', name = "Update token data", tags = ['Admin'])
async def update_token_data(token_id:str, username: str = Depends(get_current_username)):
    ''' Met à jour les données du token choisi. Si aucune donnée n'existe dans la base, une nouvelle
        table est créée avec toutes les données historiques.
    '''
    engine = create_engine('sqlite:///./historicDataBinance.db', echo=False)
    conn = engine.connect()
    # Vérification de la présence de data du token dans la base
    TOKEN = token_id.upper()
    df = pd.DataFrame()
    if token_id in engine.table_names():
        hdp.update_token_data(TOKEN)
        df = pd.read_sql_query("SELECT * FROM " + token_id, conn)
        
    # Sinon téléchargement des données historiques
    else:
        links = hdp.parseIt('klines',TOKEN, '5m')
        df = hdp.create_Dataset(links, klines_col)
        df.to_sql(token_id, con=engine, if_exists='append')
    
    conn.close()

    return {"Downloaded" : token_id}


@api.post('/token/train/{token_id}', name = "Launch Model training", tags = ['Admin'])
async def train_model(token_id:token_id, forecast_horizon:futures, username: str = Depends(get_current_username)):
    ''' Lance l'apprentissage de l'algorithme de ML sur le token choisi
    '''
    engine = create_engine('sqlite:///./historicDataBinance.db', echo=False)
    conn = engine.connect()
    # Vérification de la présence de data du token dans la base
    TOKEN = token_id.upper()
    df = pd.DataFrame()
    if token_id in engine.table_names():
        hdp.update_token_data(TOKEN)
        df = pd.read_sql_query("SELECT * FROM " + token_id, conn)
        
    # Sinon téléchargement des données historiques
    else:
        links = hdp.parseIt('klines',TOKEN, '5m')
        df = hdp.create_Dataset(links, klines_col)
        df.to_sql(token_id, con=engine, if_exists='append')
    
    conn.close()

    #
    # Paramètres du modèle
    #
    if forecast_horizon=="30m":
        future=6
    elif forecast_horizon=="1h":
        future=12
    else:
        future=24
    
    if len(df)>300000:
        df=df[-300000:]

    hist_window=24 # A window of 2 past hours
    target='Close_price'
   
    #
    # Entrainement du modèle
    #
    train_data = lstm_model.prepareData(df)
    model = lstm_model.train_model(train_data, hist_window, future, target, 3)
    model.save(TOKEN + "_"+forecast_horizon+".h5")
    
    return {"LSTM model " + token_id  : " saved"}


@api.get('/token/predict/{token_id}', name = "Launch Model prediction", tags = ['User'])
async def predict(token_id:token_id, forecast_horizon:futures):
    ''' Lance la prédiction avec l'algorithme de ML sur le token choisi
    '''
    if os.path.isfile(token_id + "_"+forecast_horizon+".h5"):

        window=24 # A window of 5 past hours
        if forecast_horizon=="30m":
            future=6
        elif forecast_horizon=="1h":
            future=12
        else:
            future=24

        model = load_model(token_id + "_" + forecast_horizon + ".h5")
        model.summary()
        
        df = sdp.get_data_streaming(token_id.lower(), window)
        df.info()

        df = lstm_model.pre_process(df)
        df.info()
        y=lstm_model.predict(df, model)
        return {'Predicted {token} price in {horizon}'.format(token=token_id, horizon=forecast_horizon) : y.tolist()[0]}

    else :
        raise HTTPException(status_code=403, detail= 'Modèle pour ' + token_id + ' non présent')


@api.get('/graph/{token_id}/{sample_size}', name = "Plot Token price and SMA", tags = ['User'])
def get_img(token_id:str, sample_size:int, background_tasks: BackgroundTasks):
    ''' Retourne un graphe du cours du token choisi ainsi que les courbes des moyennes glissantes.
        :sample_size permet de définir le nombres de données utilisées pour tracer le cours.
    '''
    engine = create_engine('sqlite:///./historicDataBinance.db', echo=False)
    conn = engine.connect()
    if(token_id not in engine.table_names()):
            raise HTTPException(status_code=403, detail= 'Token inconnu ' + token_id)
    df = pd.read_sql_query("SELECT * FROM " + token_id + " ORDER BY Open_time DESC LIMIT " + str(sample_size), conn)
    df = df.iloc[::-1]
    sma_20 = df.Close_price.rolling(20).mean()
    sma_50 = df.Close_price.rolling(50).mean()
    df_mpl = df[["Open_time", "Open_price", "High_price", "Low_price", "Close_price"]]
    df_mpl.columns = ["Open_time", "Open", "High", "Low", "Close"]
    df_mpl["Open_time"] = pd.to_datetime(df_mpl["Open_time"], unit = 'ms')
    df_mpl.index = pd.DatetimeIndex(df_mpl['Open_time'])

    plt.rcParams['figure.figsize'] = [9.50, 6.50]
    plt.rcParams['figure.autolayout'] = True
    fig, ax = plt.subplots()
    mpf.plot(df_mpl, style='yahoo', type='candle', mav=(20,50), ax=ax)
    # ax.plot(pd.to_datetime(df.Kline_Close_time, unit = 'ms'), df.Close_price)
    # ax.plot(pd.to_datetime(df.Kline_Close_time, unit = 'ms'), sma_20, color = 'navy', label = 'sma20')
    # ax.plot(pd.to_datetime(df.Kline_Close_time, unit = 'ms'), sma_50, color = 'red', label = 'sma50')
    plt.tick_params(rotation=45)
    plt.title(token_id.upper())
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()

    background_tasks.add_task(img_buf.close)
    headers = {'Content-Disposition': 'inline; filename="out.png"'}

    return Response(img_buf.getvalue(), headers=headers, media_type='image/png')

# 
# FUNCTIONS
#

def get_sma(serie, window_size):
    ''' Calcul de la moyenne glissante sur une Series
        param: serie (Series) _ Series utilisée pour le calcul de la SMA
        param: window_size (Integer) _ fenêtre de calcul
    '''
    numbers_series = pd.Series(serie)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window_size - 1:]
    
    return final_list
