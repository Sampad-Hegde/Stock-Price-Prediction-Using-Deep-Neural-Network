from sys import path as syspath
from os import path as osPath, getcwd, listdir
import math
from datetime import datetime, timedelta, time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# including the files from Web-Scraping folder
syspath.insert(1, (osPath.dirname(getcwd()).replace('\\', '\\\\')) + '\\\\Web_Scraping')

# noinspection PyUnresolvedReferences
from yahoo_search import getSymbolData
# noinspection PyUnresolvedReferences
from yahoofinscraper import YahooFinance as Yf

symbolData = None
numberOfPreviousRecord = 60


def getDataset(ticker):
    global symbolData
    symbolData = getSymbolData(ticker)

    return Yf(ticker=symbolData['symbol'], result_range='10y', interval='1d').result


def getRequiredTimeSeriesData(dataFrame, column='Close'):
    dataFrame = dataFrame.get(column)
    return dataFrame.values.reshape(-1, 1)


def splitAndFormDataset(dataset, train_size, inputLayerSize):
    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    trainDataSize = math.ceil(len(dataset) * train_size)
    for i in range(inputLayerSize, len(dataset)):
        if i < trainDataSize:
            X_train.append(dataset[i - inputLayerSize:i, 0])
            Y_train.append(dataset[i, 0])
        else:
            X_test.append(dataset[i - inputLayerSize:i, 0])
            Y_test.append(dataset[i, 0])
    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)


def buildModel(ipLayerSize):
    md = Sequential()
    md.add(LSTM(50, return_sequences=True, input_shape=(ipLayerSize, 1)))
    md.add(LSTM(50, return_sequences=50))
    md.add(LSTM(50))
    md.add(Dense(1))
    md.compile(loss='mean_squared_error', optimizer='adam')
    md.summary()
    return md


def predict(model, days, t_list):
    today = datetime.today()
    predicted_list = []
    with open('Output' + today.strftime("%d-%m-%Y") + '.txt', "w") as f:
        for i in range(1, days + 1):
            inPut = np.array(t_list[len(t_list) - numberOfPreviousRecord:]).reshape(-1, 1)
            inPut = np.array([inPut])
            yHat = model.predict(inPut)
            predicted_value = scalar.inverse_transform(yHat)[0][0]
            predicted_list.append(predicted_value)
            f.write(str(i) + "  :  " + (today + timedelta(days=i)).strftime('%d-%m-%Y') + '\t ' + str(predicted_value))
            f.write("\n")
            print(str(i) + "  :  ", (today + timedelta(days=i)).strftime('%d-%m-%Y') + '\t ', predicted_value)
            # print(i, " :  ", (today + timedelta(days=i)).strftime('%d-%m-%Y'), '\t ', scalar.inverse_transform(
            # yHat)[0][0])
            t_list = np.append(t_list, [yHat[0][0]])
    return predicted_list


def plotLossAccuracy(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig("Loss.png")
    plt.clf()

'''
def plotPriceGraph(Dataset, predictedList):
    plt.plot(np.append(Dataset, np.array(predictedList)), color='red', label='Real Google Stock Price')
    # plt.plot(np.array(predictedList), color='blue', label='Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.savefig("price_chart.png")
    '''


df = getDataset(ticker='tatamotor')
scalar = MinMaxScaler(feature_range=(0, 1))
marketCloseTime = time(16, 0, 0)

# noinspection PyUnresolvedReferences
if 'LSTM_Model_' + symbolData['shortname'] + '.h5' in listdir('./'):
    # noinspection PyUnresolvedReferences
    model = load_model('LSTM_Model_' + symbolData['shortname'] + '.h5')
    # noinspection PyUnresolvedReferences
    Dataset = getRequiredTimeSeriesData(df[-60:], 'Close')
    scaledData = scalar.fit_transform(Dataset)
    np.reshape(scaledData, scaledData.shape + (1,))
    predictedList = predict(model, 15, scaledData)
    # plotPriceGraph(Dataset, predictedList)

else:
    Dataset = getRequiredTimeSeriesData(df, 'Close')
    scaledData = scalar.fit_transform(Dataset)
    X_train, X_test, Y_train, Y_test = splitAndFormDataset(scaledData, 0.8, 60)

    # Reshaping Data to give input to Neural Network
    X_train = np.reshape(X_train, X_train.shape + (1,))
    X_test = np.reshape(X_test, X_test.shape + (1,))
    Y_train = np.reshape(Y_train, Y_train.shape + (1,))
    Y_test = np.reshape(Y_test, Y_test.shape + (1,))

    model = buildModel(numberOfPreviousRecord)

    hist = model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose=1, validation_data=(X_test, Y_test))

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    plotLossAccuracy(hist)

    train_predict = scalar.inverse_transform(train_predict)
    test_predict = scalar.inverse_transform(test_predict)

    X_input = scaledData[len(scaledData) - numberOfPreviousRecord:]
    temp_list = X_input.tolist()
    # noinspection PyUnresolvedReferences
    model.save('LSTM_Model_' + symbolData['shortname'] + '.h5')

    predictedList = predict(model, 15, temp_list)
    #plotPriceGraph(Dataset, predictedList)
