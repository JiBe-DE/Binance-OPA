import math
import datetime as dt
import pandas as pd
import requests
from requests import get
import xml.etree.ElementTree as cxml
import zipfile
from io import BytesIO

# Modules SQL
import sqlite3, sqlalchemy
from sqlalchemy import create_engine, text


# DataFrame Columns 
klines_col=['Open_time', 'Open_price', 'High_price', 'Low_price', 'Close_price', 'Volume','Kline_Close_time', 'Quote_asset_volume', 'Number_of_trades', 'TBBA_volume','TBQA_volume', 'Unused']
trades_col=['Id', 'Price', 'Qty', 'QuoteQty', 'Time', 'IsBuyerMaker','IsBestMatch']
aggTrades_col=['Aggregate tradeId', 'Price', 'Quantity', 'First_tradeId', 'Last_tradeId', 'Timestamp', 'Buyer_is_maker?', 'Price_best_match?']

#
# Functions
#

def load_RESTAPI_xml(url):
    '''
        Downloads the xml content from the binance REST API response - prep for the xml parsing.
        root url='https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/spot/monthly/'.
    '''

    r=requests.get(url,allow_redirects=True)
    return r.content

def parseIt(data, token, interval=''):
    '''
        Parses the xml content and returns a list of the download links
    :param data : aggTrades, klines, trades.
    :param token : e.g. BTCUSDT
    :param interval : (only if data=klines) 1m, 3m, 15m, 1h, 1d, 1mo, ..etc
    :return: list of urls
    '''

    if data == 'klines':
        url = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/spot/monthly/'+data+'/'+token+'/'+interval+'/'
    elif (data == 'aggTrades' or data == 'trades') :
        url = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/spot/monthly/'+data+'/'+token+'/'
    else:
        print('The specified market data is not valid. Please choose between the following available data categories : aggTrades, klines, trades.')
        exit()
    links=[]
    eRoot=cxml.fromstring(load_RESTAPI_xml(url))
    print(f'Start parsing :')
    for eChild in eRoot :
        if(eChild.tag.split("}")[1] == 'Contents'):
            if '.zip' in eChild[0].text and 'CHECKSUM' not in eChild[0].text:
                link='https://data.binance.vision/'+eChild[0].text
                print(link)
                links.append(link)

    return links

def get_Dataframe(zip_file, col):
    '''
        Extracts the csv document from the zip file and opens it in a dataframe (returns a dataframe with the specified columns).
    '''

    with zipfile.ZipFile(BytesIO(zip_file)) as myzip:
        names = zipfile.ZipFile.namelist(myzip)
        with myzip.open(names[0]) as myfile:
            df_temp = pd.read_csv(myfile, sep=',', names=col)
    return df_temp

def create_Dataset(links, col):
    '''
        Concatenates downloaded data into a single dataframe.
    :param links: list of the parsed download urls from binance market data webpage
    :param col: list of specified columns for the downloaded data (klines_col, trades_col, aggTrades_col)
    :return: dataframe of the final historical dataset
    '''
    df = pd.DataFrame(columns=col)
    i=1
    print('Creation of the dataset :')
    for link in links:
        zip_file=get(link).content
        temp_df = get_Dataframe(zip_file, col)
        df = pd.concat([df, temp_df], axis=0)
        print('Dataframe ',i,'/',len(links),' integrated with success !')
        i+=1

    return df

def update_token_data(token):
    '''
        Updates token SQL database from the last kline entry
    :param token : selected token
    '''
    engine = create_engine('sqlite:///./historicDataBinance.db', echo=False)
    conn = engine.connect()
    stmt = text("SELECT Open_time FROM " + token.lower() + " ORDER BY Open_time DESC LIMIT 1;")
    result = conn.execute(stmt)
    startDate = result.fetchall()[0][0]/1000

    new_klines = []
    #nb_loop = math.floor((dt.datetime.now().timestamp() - dt.datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S.%f').timestamp())/(1000*60*5))+1
    nb_loop = math.floor((dt.datetime.now().timestamp() - startDate)/(1000*60*5))+1
   
    for i in range(nb_loop):
        # startTime = math.floor(dt.datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S.%f').timestamp()*1000+3600000) + i * (1000*5*60*1000)
        # UTC +1 : startTime = math.floor(startDate*1000+3600000) + i * (1000*5*60*1000)
        startTime = math.floor(startDate*1000) + i * (1000*5*60*1000)
        r = requests.get(
            url='https://api.binance.com/api/v3/klines?symbol={TOKEN}&interval=5m&startTime={startTime}&limit=1000'.format(TOKEN=token, startTime=startTime),
        )
        new_klines = new_klines + r.json()

    df = pd.DataFrame(new_klines)
    df.columns = klines_col
    #df['Kline_Close_time']= [dt.datetime.fromtimestamp(x/1000.0) for x in df.Kline_Close_time]
    #df['Open_time'] = [dt.datetime.fromtimestamp(x/1000.0) for x in df.Open_time]
    df = df.drop(df.index[len(df)-1])

    df.to_sql(token.lower(), con=engine, if_exists='append')
    conn.close()
