import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Modules ML
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

#
# Functions
#

def pre_process(df):
    df=df.drop(['Open_time','Kline_Close_time', 'Unused'], axis=1)
    df=df.fillna('bfill')
    df=df.astype(float)
    return df

def split_dataset(df, n, split_time):
    # n must be greater than split_time
    train_data=df[-n : -n + split_time]
    test_data=df[-n + split_time:]
    return train_data, test_data

def create_series(df, scaler, window, future, y_column):
    df_scaled = scaler.transform(df)
    X = []
    Y = []
    for i in range(window, len(df_scaled) - future + 1):
        X.append(df_scaled[i - window : i, 0 : df.shape[1]])
        Y.append(df_scaled[i + future - 1 : i + future, df.columns.get_loc(y_column)])
    return X, Y

def create_model(trainX, trainY, activation, optimizer, loss, dropout):
    model = Sequential()
    model.add(LSTM(64, activation=activation, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation=activation, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    return model

def plot_series(time, series, label, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

def prepareData(df_source):
    # Cleaning data
    clean_data=pre_process(df_source)
    # Train/Test data splitting
    #train_data, test_data=split_dataset(clean_data, subset_size, split_time)
    return clean_data #, test_data

def train_model(train_data, window, future, target, epochs):
    # Normalize data
    scaler = StandardScaler()
    scaler_train = scaler.fit(train_data)
    # Reformat input data
    trainX, trainY=create_series(train_data, scaler_train, window, future, target)
    # Convert to np arrays
    trainX, trainY = np.array(trainX), np.array(trainY)
    # Create and train model
    model = create_model(trainX, trainY, 'relu', 'adam', 'mse', 0.2 )
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=16, verbose=1)

    return model

def predict(data, model):
    #Normalize data
    scaler = StandardScaler()
    scaler_X = scaler.fit(data)
    df_scaled = scaler_X.transform(data)
    X = []
    X.append(df_scaled[:,0:data.shape[1]])
    X=np.array(X)
    Y=model.predict(X)
    prediction = np.repeat(Y, data.shape[1], axis=-1)
    Y = scaler_X.inverse_transform(prediction)[:,0]
    return Y

def test(test_data, model, sample_size, window, future, target):
    #Normalize data
    scaler = StandardScaler()
    scaler_test = scaler.fit(test_data)
    #Generate X and Y series
    testX, testY=create_series(test_data, scaler_test, window, future, target)
    #Convert to array
    testX, testY = np.array(testX), np.array(testY)
    #Predict
    y_pred=model.predict(testX[- sample_size :])
    #Rescale back predicted values
    prediction = np.repeat(y_pred, test_data.shape[1], axis=-1)
    y_pred = scaler_test.inverse_transform(prediction)[:,0]
    #Ground truth data
    real = testY[-sample_size : ]
    temp=np.repeat(real, test_data.shape[1], axis=-1)
    y_real = scaler_test.inverse_transform(temp)[:,0]
    #Mertics
    mae = tf.keras.metrics.mean_absolute_error(y_real, y_pred).numpy()
    print(f"mae: {mae:.2f} for forecast")
    #Plot
    plt.figure(figsize=(10, 6))
    plot_series(range(len(y_real)), y_real, label='True values')
    plot_series(range(len(y_pred)), y_pred, label='Predicted values')