import time

import tensorflow as tf
from tensorflow.keras.losses import MeanAbsolutePercentageError,Huber,MeanAbsoluteError,MeanSquaredError,CosineSimilarity
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Input,Multiply, Lambda,Conv2D, LSTM, Dense, Flatten, concatenate , Reshape,Permute,Softmax
import random as rn
import os
import numpy as np
from tensorflow.keras import regularizers


sd=23
#sd=23 +130,000 gained
#sd=16
# Here sd means seed.
np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED']=str(sd)

from keras import backend as K
config =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(sd)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
K.set_session(sess)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 07:36:41 2022

@author: Novin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 22:47:04 2022

@author: Novin
"""


import pandas as pd
from collections import deque
import numpy as np
# import seaborn as sns

from matplotlib import pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# In[2]:

# tf.set_seed(1234)
# np.random.seed(1234)
FUTURE_PERIOD_PREDICT = 1


startTime = time.time()
marketDF = pd.read_excel('data/indicatorsBTCUSDTdaily.xlsx')
df = pd.read_excel('data/BTCUSDTNewsSA.xlsx')

marketDF['Date'] = pd.to_datetime(marketDF['Date'])
marketDF = marketDF.set_index('Date')

marketDF.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
marketDF.dropna(inplace=True)
marketDF['target'] = marketDF['Close'].shift(-FUTURE_PERIOD_PREDICT)

# add sentiment value to marketDF (BERT-BoEC + FinBERT-SIMF)
#marketDF = alignsentimentANDmarket(sentimentDF, marketDF, source='finBERT')
marketDF.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values

# In[3]:


FUTURE_PERIOD_PREDICT = 1


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


# In[4]:


# only for removing unwanted features
def normalization(df):
    #
    df.drop(['High', 'Low', 'Open'], axis=1, inplace=True)
    drop_list = ['Close', 'EMA', 'bb_bbl', 'on_balance_volume','ATR','Volume','ADI', 'bb_bbh', 'bb_bbm', 'stochastic',
                 'RSI', 'momentum', 'MACD', 'wiliams', 'bb_bbh', 'bb_bbm']
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
'''
# , shape=(1, 768), dtype=float32)
def modify(s, seperator):
    vec = s.replace("'[", '')
    vec = vec.replace(']', '')
    vec = vec.strip()
    vec = s.replace('tf.Tensor(\n', '')
    vec = vec.replace(", shape=(1, 768), dtype=float32)", '')
    vec = vec.replace('[[', '')
    vec = vec.replace(']]', '')
    vec = vec.strip()
    vec = vec.replace('  ', ' ')
    vec = vec.split(seperator)
    vector = []
    for w in vec:
        if w != '':
            vector.append(float(w))

    return vector
'''

SEQ_LEN_news = 4
max_L = 12
SEQ_LEN = 6


def getNewsVector(news, vectorName, seperator):
    newsEmbedding = np.asarray(modify(news[vectorName], seperator=seperator)).astype(int)
    # authorEmbedding =np.asarray( modify(news['authorEmbedding'],seperator= ' ')).astype(int)
    # posScore =float( news['pos'])
    # negScore =float( news['neg'])
    # neutScore =float( news['neut'])
    # return np.concatenate((newsEmbedding,[posScore+10,negScore+10],))#,,authorEmbedding,,neutScore+10
    # newsEmbedding = sentimentScore * newsEmbedding
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
            c = row[vectorName]
            vector[i] = getNewsVector(row, vectorName, seprator)
            i = i + 1
        return vector


def preprocess_df(df, newsDF):
    df = normalization(df)
    aligned_news_data = []
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)
    closeList = []
    for d, row in df.iterrows():
        prev_days.append([n for n in row[:-1]])  # store all but the target
        closeList.append(row['Close'])
        if len(prev_days) == SEQ_LEN:  # make sure we have 7 sequences!
            vector = 'vector'
            embedding_dim = 210
            sequential_data.append([np.array(prev_days), np.mean(closeList), row[-1],
                                    getNews_embedding2(d, newsDF, vector, embedding_dim, ','),
                                    d])  # i[-1] is the sequence target
            closeList = []

    # random.shuffle(sequential_data)  # shuffle for good measure.

    X = []
    y = []
    newsX = []
    dates = []
    meanprev = []

    for seq, mean, target, newsEmbedding, d in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)
        meanprev.append(mean)
        newsX.append(newsEmbedding)
        dates.append(d)

    return np.array(X), meanprev, np.array(y), np.array(newsX), dates  # return X and y...and make X a numpy array!


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
# clsBodyEmbed
# concept cluster number
embedding_dim = 210
newsDF = df.set_index('timestamp')

train_x, meanTrain, train_y, train_news_x, train_dates = preprocess_df(main_df, newsDF)
validation_x, meanValidation, validation_y, validation_news_x, validation_dates = preprocess_df(validation_main_df,
                                                                                                newsDF)
test_x, meanTest, test_y, test_news_x, test_dates = preprocess_df(test_main_df, newsDF)

print(train_news_x.shape)
print(validation_news_x.shape)
print(test_news_x.shape)

dim = train_x.shape[1]  # 7 delay window


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
    label = classify(df['Close'], df['target'])
    pred = classify(df['predict'], df['predict'].shift(1))
    return sum([accuracy(l, p) for l, p in zip(label, pred)]) / len(pred)



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
            buy_price = prices.iloc[i ] + (buy_commission * prices.iloc[i ])
            if asset > buy_price:
                asset -= buy_price
                sell_state = False;

        if predictions.iloc[i] < predictions.iloc[i - 1] and buy_price:
            # sell if the current prediction is lower than the previous prediction and a buy has already been made
            sell_price = prices.iloc[i ] - prices.iloc[i] * sell_commission
            profit = sell_price - buy_price
            asset += sell_price
            profits.append(profit)
            buy_price = None
            sell_state = True

    total_profit = sum(profits)
    average_profit = total_profit / len(profits) if profits else 0
    print(len(profits))
    print(total_profit)
    print(average_profit)
    print(asset)

    print("=========================")

    return profits, total_profit, average_profit


def plot_prediction(marketDF, validation_dates, pred_valid, test_dates, pred_test, train_dates, pred_train):
    plt.figure(figsize=(30, 10))
    marketDF.loc[train_dates, 'predict'] = pred_train.reshape(pred_train.shape[0])
    marketDF.loc[train_dates, 'meanPrev'] = meanTrain

    marketDF.loc[validation_dates, 'predict'] = pred_valid.reshape(pred_valid.shape[0])
    marketDF.loc[validation_dates, 'meanPrev'] = meanValidation

    marketDF.loc[test_dates, 'predict'] = pred_test.reshape(pred_test.shape[0])
    marketDF.loc[test_dates, 'meanPrev'] = meanTest
    marketDF.to_excel('AdaptivePredictNewsTA.xlsx')

    print('training ACC {}'.format(computeAccuracy(marketDF.loc[train_dates])))
    print('Validation ACC {}'.format(computeAccuracy(marketDF.loc[validation_dates])))
    print('test ACC {}'.format(computeAccuracy(marketDF.loc[test_dates])))
    #profits, total_profit, average_profit = assess_profitability(marketDF.loc[test_dates,'predict'],marketDF.loc[test_dates,'Open'], 100000, 0.001, 0.001)

    marketDF.loc[validation_dates, 'target'].plot(color='blue', label='Close Price history')
    # marketDF.loc[validation_dates,'meanPrev'].plot( color ='green', label='Mean close Price history' )
    marketDF.loc[validation_dates, 'predict'].plot(color='red', label='Predict Price history')
    # plt.savefig('news ta Validation set.png')
    plt.show()

    plt.figure(figsize=(30, 10))

    marketDF.loc[test_dates, 'target'].plot(color='blue', label='Close Price history')
    # marketDF.loc[test_dates,'meanPrev'].plot( color ='green', label='Mean Close Price history' )
    marketDF.loc[test_dates, 'predict'].plot(color='red', label='Predict Price history')
    # plt.savefig('news ta test set.png')
    marketDF.to_excel('AdaptivePredictNewsTA.xlsx')
    plt.show()

def build_model(train_x, train_news_x, train_y
                , validation_x, validation_news_x, validation_y,
                model_name):
    input_height_modality1= train_news_x.shape[1]
    input_width_modality1= train_news_x.shape[2]
    input_channels_modality1=1
    input_height_modality2 = train_x.shape[1]
    input_width_modality2= train_x.shape[2]
    input_channels_modality2 = 1

    input_shape_modality1 = (input_height_modality1, input_width_modality1)#, input_channels_modality1)
    input_shape_modality2 = (input_height_modality2, input_width_modality2)#, input_channels_modality2)

    # Input layers for each modality
    input_layer_modality1 = Input(shape=input_shape_modality1)
    input_layer_modality2 = Input(shape=input_shape_modality2)
    num_hidden_units = 64
    # Modality 1: Summarization network with SOTA method
    #input_matrix = Flatten(input_layer_modality1)
    hidden_layer = Dense(num_hidden_units, use_bias=False)(input_layer_modality1)
    hidden_layer = Dense(1, use_bias=False)(hidden_layer)

    # Perform self-attention mechanism
    attention_weights = Dense(1)(hidden_layer)
    attention_weights = Softmax(axis=1)(attention_weights)
    attention_weights = Permute((2, 1))(attention_weights)

    # Apply attention weights to hidden layer
    weighted_hidden_layer = Multiply()([attention_weights, hidden_layer])
    output_modality1 = Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_hidden_layer)


    # Modality 2: Multilayer dense network
    conv_layer_modality2 = Flatten()(input_layer_modality2)
    #flatten_layer_modality2 = Flatten()(conv_layer_modality2)
    dense_layer1_modality2 = Dense(units=128, activation="relu")(conv_layer_modality2)
    dense_layer2_modality2 = Dense(units=64, activation="relu")(dense_layer1_modality2)
    output_modality2 = Dense(units=1, activation="linear")(dense_layer2_modality2)

    # Concatenate the outputs of both modalities
    concatenated_outputs = concatenate([output_modality1, output_modality2])
    dense_regressed = Dense(units=1, activation="linear")(concatenated_outputs)

    # Create the combined model
    combined_model = tf.keras.Model(inputs=[input_layer_modality1, input_layer_modality2], outputs=dense_regressed)

    combined_model.compile(loss=MeanAbsolutePercentageError(), optimizer=Adam(lr=0.001, decay=1e-6),
                           metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()])
    combined_model.summary()
    #tf.keras.utils.plot_model(model, show_shapes=True)
    # callback = EarlyStopping(monitor='val_loss', patience=30 , min_delta=0.00001,)

    history = combined_model.fit([ train_news_x, train_x], train_y,
                        validation_data=([validation_news_x,validation_x ], validation_y),
                        epochs=20, batch_size=32, )  # callbacks=[callback])
    #combined_model.save(model_name)
    return history, combined_model

history, model = build_model(train_x, train_news_x, train_y
                , validation_x, validation_news_x, validation_y,
                 'BHAM')
endTime = time.time()
print("total time", endTime-startTime)

plot_loss(history)
pred_train = model.predict([ train_news_x,train_x])

pred_valid = model.predict([validation_news_x,validation_x])

pred_test = model.predict([test_news_x,test_x])
print("Validation loss : {}".format(model.evaluate([validation_news_x,validation_x], np.array(validation_y))))
# print("Test loss : {}" .format(model.evaluate([test_x,test_news_x],np.array(test_y))))
plot_prediction(marketDF, validation_dates, pred_valid, test_dates, pred_test, train_dates, pred_train)
print("Test loss : {}".format(model.evaluate([test_news_x,test_x], np.array(test_y))))
