# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 22:47:04 2022

@author: Novin
"""

from sklearn.metrics import accuracy_score

import pandas as pd
from collections import deque
import numpy as np
import random
from matplotlib import pyplot as plt
from datetime import timedelta
from superclassingAPISoftmax import predictiveModel

FUTURE_PERIOD_PREDICT = 1
SEQ_LEN = 6
SEQ_LEN_news = 4
max_L = 15
SEQ_LEN = 6

marketDF = pd.read_excel('data/indicatorsBTCUSDTdaily.xlsx')
df = pd.read_excel('data/BTCUSDTNewsSA.xlsx')
marketDF['Date'] = pd.to_datetime(marketDF['Date'])
marketDF = marketDF.set_index('Date')
marketDF.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
marketDF.dropna(inplace=True)
marketDF['target'] = marketDF['Close'].shift(-FUTURE_PERIOD_PREDICT)
marketDF.fillna(method="ffill", inplace=True)

FUTURE_PERIOD_PREDICT = 1


def classify(current, future):
    res = []
    for c, f in zip(current, future):
        if float(c) > float(f):  # if the future price is higher than the current, that's a buy, or a 1
            res.append(0)
        else:  # otherwise... it's a 0!
            res.append(1)
    return res


# In[4]:

f_marketDF = marketDF
# only for removing unwanted features
def normalization(df):

    df = df.drop("High", axis=1)  # don't need this anymore.
    df = df.drop("Low", axis=1)  # don't need this anymore.
    df = df.drop("Open", axis=1)  # don't need this anymore.
    #
    drop_list = ['Close', 'EMA', 'bb_bbl', 'on_balance_volume', 'ATR', 'Volume','ADI','bb_bbh', 'bb_bbm', 'stochastic',
                 'RSI',  'momentum', 'MACD', 'wiliams', 'bb_bbh', 'bb_bbm']
    # drop_list = ['Volume', 'stochastic','RSI', 'ADI','momentum','MACD','bb_bbl', 'bb_bbm', 'ATR','wiliams']
    df.drop(drop_list[4:], axis=1, inplace=True)
    df.dropna(inplace=True)  # cleanup again
    return df


def modify(s, seperator):
    vec = s.replace('[', '')
    vec = vec.replace(']', '')
    vec = vec.strip()
    vec = vec.replace('  ', ' ')
    vec = vec.split(seperator)
    vector = []
    for w in vec:
        if w != '':
            vector.append(float(w))

    return vector


def getNewsVector(news, vectorName, seperator):
    newsEmbedding = np.asarray(modify(news[vectorName], seperator=seperator)).astype(int)
    return newsEmbedding


def getNews_embedding2(currentDate, df, vectorName, embedding_dim, seprator):
    news_date = df.index.values
    prevDate = currentDate - timedelta(days=SEQ_LEN_news)
    subDF = df.loc[prevDate:currentDate]
    if len(subDF) == 0:
        return (np.zeros((max_L, embedding_dim)))

    else:
        vector = np.zeros((max_L, embedding_dim))
        i = 0
        for d, row in subDF[:max_L].iterrows():
            c = row[vectorName]
            vector[i] = getNewsVector(row, vectorName, seprator)
            i = i + 1
        return vector


def preprocess_df(df ):
    df = normalization(df)
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)

    for d, row in df.iterrows():
        prev_days.append([n for n in row[:-1]])  # store all but the target

        if len(prev_days) == SEQ_LEN:  # make sure we have 7 sequences!
            vector = 'vector'
            embedding_dim = 210
            sequential_data.append([np.array(prev_days), row[-1],
                                    getNews_embedding2(d, newsDF, vector, embedding_dim, ','),
                                    d])  # i[-1] is the sequence target
            closeList = []

    random.shuffle(sequential_data)  # shuffle for good measure.

    X = []
    y = []
    newsX = []
    dates = []

    for seq, target, newsEmbedding, d in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)
        newsX.append(newsEmbedding)
        dates.append(d)

    return np.array(X), np.array(y), np.array(newsX), dates  # return X and y...and make X a numpy array!


dates = sorted(marketDF.index.values)  # get the dates
last_5pct = sorted(marketDF.index.values)[-int(0.2 * len(dates))]  # get the last 20% of the times

test_main_df = marketDF[(marketDF.index >= last_5pct)]  # make the validation data where the index is in the last 5%
main_df = marketDF[(marketDF.index < last_5pct)]  # now the main_df is all the data up to the last 5%

dates = sorted(main_df.index.values)  # get the dates
last_5pct = sorted(main_df.index.values)[-int(0.2 * len(dates))]  # get the last 20% of the times
validation_main_df = main_df[(main_df.index > last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

print(main_df.shape)
print(validation_main_df.shape)
print(test_main_df.shape)

print(main_df.head())
print(main_df.describe())

# dates = sorted(marketDF.index.values)  # get the dates
# last_5pct = sorted(marketDF.index.values)[-int(0.2*len(dates))]  # get the last 20% of the times

# validation_main_df = marketDF[(marketDF.index >= last_5pct)]  # make the validation data where the index is in the last 5%
# main_df = marketDF[(marketDF.index < last_5pct)]  # now the main_df is all the data up to the last 5%

# dates = sorted(main_df.index.values)  # get the dates
# last_5pct = sorted(main_df.index.values)[-int(0.2*len(dates))]  # get the last 20% of the times
# validation_main_df =  main_df[(main_df.index > last_5pct)]
# main_df = main_df[(main_df.index < last_5pct)]

print(main_df.shape)
print(validation_main_df.shape)
# print(test_main_df.shape)

print(main_df.head())
print(main_df.describe())

var = []
for i in range(len(df)):
    d = pd.to_datetime(df.loc[i, 'Date'])
    # h = str(df.loc[i,'time'] )
    var.append(d.strftime('%Y-%m-%d %H:%M:%S'))

df['timestamp'] = pd.to_datetime(var)
df.sort_values(by=['timestamp'], inplace=True)
print(df.head())

# concept cluster number
embedding_dim = 210
newsDF = df.set_index('timestamp')
newsStartDate = newsDF.index[0]
dailyMarketStart = marketDF.index[0]
maxDate = newsStartDate
if dailyMarketStart > maxDate:
    maxDate = dailyMarketStart


newsEndDate = newsDF.index[-1]
dailyMarketEnd = marketDF.index[-1]
minDate = newsStartDate
if dailyMarketEnd > minDate:
    minDate = dailyMarketEnd


marketDF = marketDF[maxDate:minDate]
newsDF = newsDF[maxDate:minDate]



train_x, train_y, train_news_x, train_dates = preprocess_df(main_df)
validation_x, validation_y, validation_news_x, validation_dates = preprocess_df(validation_main_df)
test_x, test_y, test_news_x, test_dates = preprocess_df(test_main_df)

dim = train_x.shape[1]  # 7 delay window


# saeede: I'm here!!!
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Percentage Error [Close]')
    plt.legend()
    plt.grid(True)


def plot_accuracy(history):
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


def classify(current, future):
    res = []
    for f, c in zip(current, future):
        if float(f) > float(c):  # if the future price is higher than the current, that's a buy, or a 1
            res.append(1)
        else:  # otherwise... it's a 0!
            res.append(0)
    return res


accuracy = lambda x, y: 1 if x == y else 0


def computeAccuracy(df):
    df.dropna(inplace=True)
    df = df.sort_index()
    label = classify(df['Close'], df['target'])
    pred = classify(df['predict'].shift(1), df['predict'])
    acc = accuracy_score(label, pred)
    print('acc:', acc)
    # return sum([accuracy(l,p) for l,p in zip(label,pred)])/len(pred)
    computeMetric(df['target'], df['predict'])


def mean_absolute_percentage_error(actual, pred):
    # 100 * abs((y_true - y_pred) / y_true)
    loss = 0
    len = 0
    for a, p in zip(actual, pred):
        if a != 0:
            loss += abs(a - p) / a
            len += 1
    return loss / len * 100


def mean_squared_error(actual, pred):
    # 100 * abs((y_true - y_pred) / y_true)
    loss = 0
    for a, p in zip(actual, pred):
        loss += np.square(a - p)
    return loss / len(actual)


def mean_absolute_error(actual, pred):
    # 100 * abs((y_true - y_pred) / y_true)
    loss = 0
    len = 0;
    for a, p in zip(actual, pred):
        if a != 0:
            loss += abs(a - p) / a
            len += 1
    return loss / len


def computeMetric(actual, prediction):
    actual = actual.sort_index()
    prediction = prediction.sort_index()
    mae = mean_absolute_error(actual, prediction)
    mape = mean_absolute_percentage_error(actual, prediction)
    mse = mean_squared_error(actual, prediction)

    print('mae:', mae)
    print('mape:', mape)
    print('mse:', mse)
    print('======================================')

    return {'mae': mae,
            'mape': mape,
            'mse': mse,
            }


def assess_profitability(predictions, prices, asset, buy_commission=0.001, sell_commission=0.001):
    # predictions: a list of predicted market values
    # prices: a list of actual market values
    assert len(predictions) == len(prices), "Length of predictions and prices should be the same"
    buy_price = None
    sell_state = True
    predictions = predictions.sort_index()
    prices = prices.sort_index()
    profits = []
    for i in range(1, len(predictions)):
        # if predictions.iloc[i] > predictions.iloc[i - 1] and sell_state and asset > 0:
        if predictions.iloc[i] > predictions.iloc[i - 1] and sell_state and asset > 0:
            # buy if the current prediction is higher than the previous prediction
            buy_price = prices.iloc[i] + (buy_commission * prices.iloc[i])
            if asset > buy_price:
                asset -= buy_price
                sell_state = False;

        if predictions.iloc[i] < predictions.iloc[i - 1] and buy_price:
            # sell if the current prediction is lower than the previous prediction and a buy has already been made
            sell_price = prices.iloc[i] - prices.iloc[i] * sell_commission
            profit = sell_price - buy_price
            asset += sell_price
            profits.append(profit)
            buy_price = None
            sell_state = True

    total_profit = sum(profits)
    average_profit = total_profit / len(profits) if profits else 0
    print(len(profits))
    print("total profit : {}".format(total_profit))
    print(average_profit)
    print("Asset + profit : {}".format(asset))

    print("=========================")

    return profits, total_profit, average_profit


def plot_prediction(daily_market , validation_dates, pred_valid, test_dates, pred_test, train_dates, pred_train):
    plt.figure(figsize=(30, 10))
    daily_market .loc[train_dates, 'predict'] = pred_train.reshape(pred_train.shape[0])
    daily_market .loc[validation_dates, 'predict'] = pred_valid.reshape(pred_valid.shape[0])
    daily_market .loc[test_dates, 'predict'] = pred_test.reshape(pred_test.shape[0])

    daily_market  = daily_market .sort_values('Date')
    print('training ACC {}'.format(computeAccuracy(daily_market .loc[train_dates])))
    print('Validation ACC {}'.format(computeAccuracy(daily_market .loc[validation_dates])))
    print('test ACC {}'.format(computeAccuracy(daily_market .loc[test_dates])))
    #profits, total_profit, average_profit = assess_profitability(daily_market .loc[test_dates, 'predict'],
    #                                                             daily_market .loc[test_dates, 'Open'], 100000, 0.001, 0.001)

    print("===================================")

    plt.figure(figsize=(30, 10))
    daily_market .loc[train_dates, 'target'].plot(color='blue', label='Close Price history')
    # marketDF.loc[validation_dates,'meanPrev'].plot( color ='green', label='Mean close Price history' )
    daily_market .loc[train_dates, 'predict'].plot(color='red', label='Predict Price history')
    # plt.savefig('news ta Validation set.png')
    plt.show()

    plt.figure(figsize=(30, 10))

    daily_market .loc[validation_dates, 'target'].plot(color='blue', label='Close Price history')
    # marketDF.loc[validation_dates,'meanPrev'].plot( color ='green', label='Mean close Price history' )
    daily_market .loc[validation_dates, 'predict'].plot(color='red', label='Predict Price history')
    # plt.savefig('news ta Validation set.png')
    plt.show()

    plt.figure(figsize=(30, 10))

    daily_market .loc[test_dates, 'target'].plot(color='blue', label='Target close price history', fontsize=14)
    # marketDF.loc[test_dates,'meanPrev'].plot( color ='green', label='Mean Close Price history' )
    daily_market .loc[test_dates, 'predict'].plot(color='red', label='ABM-BCSIM predicted price history', fontsize=14)
    # plt.savefig('news ta test set.png')
    plt.legend(fontsize=14)
    daily_market.to_excel('testStrategy1.xlsx')
    plt.show()



history, model = predictiveModel.baselineModelTrain(train_x, train_news_x,  train_y
                   , validation_x, validation_news_x,  validation_y,
                                            )
plot_loss(history)
pred_train = model.predict([train_x,train_news_x])

pred_valid = model.predict([validation_x,validation_news_x])

pred_test = model.predict([test_x,test_news_x])
print("Validation loss : {}".format(model.evaluate([validation_x,validation_news_x], np.array(validation_y))))
# print("Test loss : {}" .format(model.evaluate([test_x,test_news_x],np.array(test_y))))
plot_prediction(f_marketDF, validation_dates, pred_valid, test_dates, pred_test, train_dates, pred_train)
print("Test loss : {}".format(model.evaluate([test_x,test_news_x], np.array(test_y))))
