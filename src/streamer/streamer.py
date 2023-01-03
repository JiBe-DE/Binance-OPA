#!/usr/bin/python3 
import json
import websockets
import asyncio

# Module MongoDB
from pymongo import MongoClient

# Mongo Functions
def mongo_connect():
    ''' Instanciation du client MongoDB
    '''
    return MongoClient(host='172.24.0.1', port=27017)

def get_mongo_db(client):
    ''' Retourne la base de données Binance contenue dans MongoDB
    '''
    return client["binance"]

def get_col(token, db):
    ''' Retourne la collection contenant les documents du token correspondant
        :param token (str)
        :param db (BDD Mongo) 
    '''
    if(token + 'Stream' not in db.list_collection_names()):
        db.create_collection(name = token + 'Stream', capped=True, size=5242880, max=6000)
        print("Liste des collections : ", db.list_collection_names())
    return db[token + 'Stream']

# Streams 
connections = set()
connections.add('wss://stream.binance.com:9443/ws/btceur@kline_5m')
connections.add('wss://stream.binance.com:9443/ws/etheur@kline_5m')
connections.add('wss://stream.binance.com:9443/ws/bnbeur@kline_5m')

client = mongo_connect()
db = get_mongo_db(client)

async def handle_socket(uri, ):
    ''' Gestionnaire de socket utilisé pour recevoir, traiter et stocker les données de streaming Binance
    '''
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            json_kline = json.loads(message)['k']
            token = json_kline['s'].lower()
            json_kline.pop('s')
            json_kline.pop('i')
            json_kline.pop('f')
            json_kline.pop('L')
            json_kline.pop('x')
            json_kline['Open_time'] = json_kline.pop('t')
            json_kline['Kline_Close_time'] = json_kline.pop('T')
            json_kline['Open_price'] = json_kline.pop('o')
            json_kline['Open_price'] = float(json_kline['Open_price'])
            json_kline['Close_price'] = json_kline.pop('c')
            json_kline['Close_price'] = float(json_kline['Close_price'])
            json_kline['High_price'] = json_kline.pop('h')
            json_kline['High_price'] = float(json_kline['High_price'])
            json_kline['Low_price'] = json_kline.pop('l')
            json_kline['Low_price'] = float(json_kline['Low_price'])
            json_kline['Volume'] = json_kline.pop('v')
            json_kline['Volume'] = float(json_kline['Volume'])
            json_kline['Quote_asset_volume'] = json_kline.pop('q')
            json_kline['Quote_asset_volume'] = float(json_kline['Quote_asset_volume'])
            json_kline['Number_of_trades'] = json_kline.pop('n')
            json_kline['TBBA_volume'] = json_kline.pop('V')
            json_kline['TBBA_volume'] = float(json_kline['TBBA_volume'])
            json_kline['TBQA_volume'] = json_kline.pop('Q')
            json_kline['TBQA_volume'] = float(json_kline['TBQA_volume'])
            json_kline['Unused'] = json_kline.pop('B')
            col = get_col(token, db)
            col.insert_one(json_kline)
            print(json_kline)

async def handler():
    await asyncio.wait([handle_socket(uri) for uri in connections])

asyncio.get_event_loop().run_until_complete(handler())