# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 22:47:04 2022

@author: Novin
"""

import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import timedelta
import random
from sklearn.metrics import accuracy_score
from superclassingAPISoftmax  import predictiveModel

FUTURE_PERIOD_PREDICT = 1

SEQ_LEN_news = 4
max_L = 15
SEQ_LEN_daily = 5
# def  getScore(marketDF,sentimentHere is the revised version of your text:


print("ABM-BCSIM BTCUSD")


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def logReturn(current, future):
    res = []
    for c, f in zip(current, future):
        res.append(math.pow(math.log(f / c), 2))
    return res


marketDF = pd.read_excel('data/BTCUSDTHourlyIndicators.xlsx')
df = pd.read_excel('data/BTCUSDTNewsSA.xlsx')
daily_marketDF = pd.read_excel('data/indicatorsBTCUSDTdaily.xlsx')

# marketDF['timestamp'] = marketDF['Date']  # +'T'+marketDF['time']
daily_marketDF['Date'] = pd.to_datetime(daily_marketDF['Date'])

marketDF['Date'] = pd.to_datetime(marketDF['Date'])
marketDF = marketDF.set_index('Date')
marketDF = marketDF.asfreq('H')
marketDF.fillna(method="bfill", inplace=True)  # if there are gaps in data, use previously known values

# marketDF['squearedLagReturn'] = logReturn( marketDF['Close'].shift(+1),marketDF['Close'])
daily_marketDF = daily_marketDF.set_index('Date')
daily_marketDF = daily_marketDF.asfreq('D')
daily_marketDF.fillna(method="bfill", inplace=True)  # if there are gaps in data, use previously known values

marketDF.fillna(method="bfill", inplace=True)  # if there are gaps in data, use previously known values
marketDF.dropna(inplace=True)

daily_marketDF.fillna(method="ffill", inplace=True)
daily_marketDF['target'] =list(map(classify, daily_marketDF['Close'],daily_marketDF['Close'].shift(-FUTURE_PERIOD_PREDICT)))

# add sentiment value to marketDF (BERT-BoEC + FinBERT-SIMF)

# marketDF.to_excel('TAScore.xlsx')

daily_marketDF.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values

# In[3]:

FUTURE_PERIOD_PREDICT = 1
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



def normalization(df):
    #
    df.drop(['High', 'Low', 'Open'], axis=1, inplace=True)
    drop_list = ['Close', 'EMA', 'bb_bbl', 'on_balance_volume', 'ATR','Volume','ADI', 'bb_bbh', 'bb_bbm',   'stochastic',
                 'RSI',  'momentum', 'MACD', 'wiliams', 'bb_bbh', 'bb_bbm']
    # drop_list = ['Volume', 'stochastic','RSI', 'ADI','momentum','MACD','bb_bbl', 'bb_bbm', 'ATR','wiliams']
    df.drop(drop_list[3:], axis=1, inplace=True)
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
    prevDate = currentDate - timedelta(hours=SEQ_LEN_news)
    subDF = df.loc[prevDate:currentDate]
    if len(subDF) == 0:
        return (np.zeros((max_L, embedding_dim)))

    else:
        vector = np.zeros((max_L, embedding_dim))
        i = 0
        for d, row in subDF[:max_L].iterrows():
            vector[i] = getNewsVector(row[1], vectorName, seprator)
            i = i + 1
        return vector


def prepaire_Data(today):
    # step 1 : selecting intra_day news and hourly market data as well as daily market

    prevDay = today - timedelta(days=1)
    partialNews = newsDF[prevDay:today]
    partialMarketData = marketDF[prevDay:today]
    lag_day = today - timedelta(days=SEQ_LEN_daily)
    dailyMarketData = daily_marketDF[lag_day:today]
    # drop targetfrom dailyMarketData
    dailyMarketData = dailyMarketData.drop('target', axis=1)

    newsVectors = np.zeros((embedding_dim, max_L * 6))
    dt_news_indices = [today + timedelta(hours=4 * i) for i in range(0, 6)]

    count = 0
    for i in range(0, 5):

        subDF = partialNews[dt_news_indices[i]:  dt_news_indices[i + 1]]
        sb_count = 0
        if len(subDF):
            for dt, row in subDF.iterrows():
                embed = getNewsVector(row, 'vector', ',')
                newsVectors[:, count] = embed
                count += 1
                sb_count += 1
        for j in range(sb_count, max_L):
            newsVectors[:, count] = np.zeros(embedding_dim)
            count += 1

    return np.array(partialMarketData).astype('float32'), newsVectors, np.array(dailyMarketData).astype('float32')


def preprocess_df(newsDF, marketDF, daily_marketDF, startDate, endDate):
    sequential_data = []
    startDate += timedelta(days=SEQ_LEN_daily)
    for dt in pd.date_range(start=startDate, end=endDate, freq='1D'):
        sequential_data.append([prepaire_Data(dt), daily_marketDF.loc[dt, 'target']])

    random.shuffle(sequential_data)  # shuffle for good measure.

    X = []
    newsX = []
    daily_x = []
    y = []

    for ele, target in sequential_data:  # going over our new sequential data
        market = ele[0]
        news = ele[1]
        dailymarket = ele[2]
        X.append(market)  # X is the sequences
        newsX.append(news)
        daily_x.append(dailymarket)
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), np.array(newsX), np.array(daily_x), np.array(
        y), pd.date_range(start=startDate, end=endDate, freq='1D')  # return X and y...and make X a numpy array!


## DATE UNIFICATION
newsStartDate = newsDF.index[0]
hourlyMarketStart = marketDF.index[0]
dailyMarketStart = daily_marketDF.index[0]
maxDate = newsStartDate
if hourlyMarketStart > maxDate:
    maxDate = hourlyMarketStart
elif dailyMarketStart > maxDate:
    maxDate = dailyMarketStart

newsEndDate = newsDF.index[-1]
hourlyMarketEnd = marketDF.index[-1]
dailyMarketEnd = daily_marketDF.index[-1]
minDate = newsStartDate
if hourlyMarketEnd > minDate:
    minDate = hourlyMarketEnd
elif dailyMarketEnd > minDate:
    minDate = dailyMarketEnd

marketDF = marketDF[maxDate:minDate]
newsDF = newsDF[maxDate:minDate]
daily_marketDF = daily_marketDF[maxDate:minDate]
f_daily_marketDF = daily_marketDF[maxDate:minDate]
# REMOVE EXTRA FEATURES
marketDF = normalization(marketDF)
daily_marketDF = normalization(daily_marketDF)

dates = sorted(daily_marketDF.index.values)  # get the dates
last_5pct = sorted(daily_marketDF.index.values)[-int(0.2 * len(dates))]  # get the last 20% of the times

test_main_df = daily_marketDF[
    (daily_marketDF.index >= last_5pct)]  # make the validation data where the index is in the last 5%
main_df = daily_marketDF[(daily_marketDF.index < last_5pct)]  # now the main_df is all the data up to the last 5%

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

train_x, train_news_x, train_daily, train_y, train_dates = preprocess_df(
    newsDF, marketDF, daily_marketDF, main_df.index[0], main_df.index[-1])

validation_x, validation_news_x, validation_daily, \
    validation_y, validation_dates = preprocess_df(newsDF, marketDF, daily_marketDF,
                                                   validation_main_df.index[0],
                                                   validation_main_df.index[-1])

test_x, test_news_x, test_daily, test_y, test_dates = preprocess_df(newsDF, marketDF, daily_marketDF,
                                                                    test_main_df.index[0],
                                                                    test_main_df.index[-1])


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
    profits, total_profit, average_profit = assess_profitability(daily_market .loc[test_dates, 'predict'],
                                                                 daily_market .loc[test_dates, 'Open'], 100000, 0.001, 0.001)

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


history, model = predictiveModel.modelTrain(train_x, train_news_x, train_daily, train_y
                                 , validation_x, validation_news_x, validation_daily, validation_y,
                                 problem='classifier')
# plot_loss(history)
pred_train = model.predict([train_x, train_news_x, train_daily])

pred_valid = model.predict([validation_x, validation_news_x,validation_daily])

pred_test = model.predict([test_x, test_news_x,test_daily])
print("Validation loss : {}".format(model.evaluate([validation_x, validation_news_x,validation_daily], np.array(validation_y))))
# print("Test loss : {}" .format(model.evaluate([test_x,test_news_x],np.array(test_y))))
plot_prediction(f_daily_marketDF, validation_dates, pred_valid, test_dates, pred_test, train_dates, pred_train)
print("Test loss : {}".format(model.evaluate([test_x, test_news_x,test_daily], np.array(test_y))))
