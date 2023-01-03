# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 23:50:57 2022

@author: barif
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import timedelta

def pre_process(df):
    """
    # Nommage des noms de colonnes 
    df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
                       'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']
    

    # Conversion des données 'Open Time' et 'Close Time' en format Date
    df['Open Time'] = pd.to_datetime(df['Open Time']/1000, unit='s')
    df['Close Time'] = pd.to_datetime(df['Close Time']/1000, unit='s')
    df['Close Time'] = df['Close Time'].dt.round('1s')

    # Conversion des autres varibales en format numérique 
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'TB Base Volume', 'TB Quote Volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)


    # Suppréssion des varaibles non pertinentes à l'analyse des données 
    colonnes = ['TB Quote Volume', 'Ignore']
    df.drop(colonnes, axis=1, inplace=True)

    # On réeordonne les données en format
    df = df[['Open Time', 'Close Time', 'Open', 'Close', 
                       'High', 'Low', 'Volume', 'Number of Trades', 
                       'Quote Asset Volume', 'TB Base Volume']]
    """

    df["Prediction"] = df["Close"].shift(-1)
    df = df[: -1]
    
    return df

def plot_data(df):
    for k in df.keys():
        if k not in ["Prediction"]:
            plt.title(f"Prediction vs {k}")
            plt.xlabel(f"{k}")
            plt.ylabel("Prediction")
            plt.plot(df[k], df["Prediction"], "o")
            plt.show()
            
def create_series(df):
    X = df.loc[:, ~df.columns.isin(["Open Time", "Close Time", "Volume", "Prediction"])].values
    y = df["Prediction"].values.reshape(-1, 1)
    
    return X, y
            
def normalize_data(X):
    features_scaler = MinMaxScaler(feature_range=(0, 1))      
    x_normalized = features_scaler.fit_transform(X)
    
    return x_normalized, features_scaler

def split_dataset(x_normalized, y):
    X_train, X_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

def create_model():
    regressor = LinearRegression()
    return regressor

def train_model(X_train, y_train, regressor):
    history = regressor.fit(X_train, y_train)
    
    return regressor

def predict(X_test, y_test, regressor):
    y_test_pred = regressor.predict(X_test)

    print('Mean absolute error: $ %.2f'
          % mean_absolute_error(y_test, y_test_pred))
    print('Mean absolute percentage error: %.2f'
          % mean_absolute_percentage_error(y_test, y_test_pred))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_test_pred))
    
    return y_test_pred
    
    
def prediction_price_tomorrow(df, regressor, features_scaler):
    last_row = df.iloc[-1, :]
    last_day_date = last_row["Close Time"]
    tomorrow_date = last_day_date + timedelta(days=1)
    tomorrow_date_str = tomorrow_date.isoformat().split("T")[0]

    x_predict = last_row.loc[ ~last_row.keys().isin(["Open Time", "Close Time", "Volume", "Prediction"])].values.reshape(-1,7)
    x_predict_scaled = features_scaler.transform(x_predict)
    tomorrow_price = regressor.predict(x_predict_scaled)
    
    return tomorrow_price

    
def prepareData(df_source):
    clean_data=pre_process(df_source)
    
    return clean_data