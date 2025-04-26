# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 05:42:16 2020

@author: Novin
"""
import numpy as np

import pandas as pd
import ta

# Load datas
df = pd.read_excel('data/btcusdtdaily.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
# df =df.drop('Vol.', axis=1)
df.dropna(inplace=True)
# Clean NaN values
# df = ta.utils.dropna(df)
# df['Price'] =[ float(ele.replace(",","")) for ele in df['Price']]
# df['Low'] =[ float(ele.replace(",","")) for ele in df['Low']]
# df['High'] =[ float(ele.replace(",","")) for ele in df['High']]
# df['Vol.'] =[ float(ele.replace(",","")) for ele in df['Vol.']]

# Initialize Bollinger Bands Indicator
indicator_bb = ta.volatility.BollingerBands(close=df["Close"])
'''
set 1 :
Stochastic %K, Stochastic %D, Momentum, Rate of change,
William’s %R, Accumulation/Distribution (A/D)

set 2
EMA, MACD, RSI, On Balance Volume, Bollinger Bands

'''
df['stochastic'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'],
                                                    close=df['Close'],
                                                    fillna=False).stoch()

df['momentum'] = ta.momentum.AwesomeOscillatorIndicator(high=df['High'], low=df['Low'],
                                                        fillna=False)

df['wiliams'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'],
                                               close=df['Close'], lbp=14, fillna=False)

df['ADI'] = ta.volume.AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close']
                                            , volume=df['Volume'], fillna=False).acc_dist_index()

df['EMA'] = ta.trend.EMAIndicator(close=df['Close'], fillna=False).ema_indicator()

df['MACD'] = ta.trend.MACD(close=df['Close'], fillna=False).macd()

df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], fillna=False).rsi()

df['on_balance_volume'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'],
                                                             fillna=False).on_balance_volume()
df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], close=df['Close'], low=df['Low'])

df['bb_bbm'] = indicator_bb.bollinger_mavg()
df['bb_bbh'] = indicator_bb.bollinger_hband()
df['bb_bbl'] = indicator_bb.bollinger_lband()
df.to_excel('data/indicatorsBTCUSDTdaily.xlsx')
