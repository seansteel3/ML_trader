# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:10:30 2023

@author: q23853
"""


import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta, date
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import randrange
#import statsmodels.formula.api as smf
#import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import random
import math
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Concatenate, Bidirectional, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from imblearn.under_sampling import RandomUnderSampler
from scipy.signal import savgol_filter
import seaborn as sns
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LinearRegression
import pyearth
import json
from urllib.request import urlopen
import time             
from scipy.stats import norm
from sklearn.decomposition import PCA
from io import BytesIO
import logging
import requests
import tempfile


path1 = 'C:\\Users\\q23853\\Desktop\\random_trader\\processed_securities122023'
path2 = 'C:\\Users\\q23853\\Desktop\\random_trader\\processed_stocks2'
path3 = 'C:\\Users\\q23853\\Desktop\\random_trader\\'
os.chdir(path1)
pd.options.mode.chained_assignment = None
#%%
'''
Data collection and cleaning function
'''
#%%
def timeme(method):
    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))

        print(endTime - startTime,'ms')
        return result

    return wrapper

#define technical indeicator functions
def bbands(data, period, c_name = 'bb_center', u_name = 'bb_upp', l_name = 'bb_low'):
    ma = data.Close.rolling(window = period).mean()
    std = data.Close.rolling(window = period).std()
    data[c_name] = ma
    data[u_name] = ma + (2 * std)
    data[l_name] = ma - (2* std)
    return data

def minmax_ratio(data, period):
    return ((data['Close'] - data['Low'].rolling(period).min())/(data['High'].rolling(period).max() - data['Low'].rolling(period).min()))

def atr(high, low, close, period=14):
    tr = np.amax(np.vstack(((high - low).to_numpy(), (abs(high - close)).to_numpy(), (abs(low - close)).to_numpy())).T, axis=1)
    return pd.Series(tr).rolling(period).mean().to_numpy()

def rsi(close, period = 14):
    close_delta = close.diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com = period - 1, adjust=True, min_periods = period).mean()
    ma_down = down.ewm(com = period - 1, adjust=True, min_periods = period).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi/100

def vol(data):
    return np.sqrt((sum((data - data.mean())**2))/len(data))/data.mean()
#lam_vol = lambda x : (vol(x))

#define target and interesting ancillary functions
def cross_thresh(data, period, thresh, up):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=period)
    if up == True:
        datac = data.copy()
        datac['look_ahead'] = datac['High'].rolling(window = indexer, min_periods = 1).max()
        return_col  = (datac['look_ahead'] > datac['Close'] * thresh).astype(int)
    else:
        datac = data.copy()
        datac['look_ahead'] = datac['Low'].rolling(window = indexer, min_periods = 1).min()
        return_col  = (datac['look_ahead'] < datac['Close'] * thresh).astype(int)
    return return_col


#total ML trader data preparation function
def prepare_MLdata(data, period, pred):
    if len(data) > period + 1:
        #apply BBands, MinMax, ATR, RSI, and VOL for 21, 84, and 252 trading days
        data = bbands(data, 21, c_name = 'bb_center21', u_name = 'bb_upp21', l_name = 'bb_low21')
        data = bbands(data, 84, c_name = 'bb_center84', u_name = 'bb_upp84', l_name = 'bb_low84')
        data = bbands(data, 252, c_name = 'bb_center252', u_name = 'bb_upp252', l_name = 'bb_low252')
        
        data['minmax21'] = minmax_ratio(data, 21)
        data['minmax84'] = minmax_ratio(data, 84)
        data['minmax252'] = minmax_ratio(data, 252)
        
        data['atr_21'] = atr(data['High'], data['Low'], data['Close'], 21)
        data['atr_84'] = atr(data['High'], data['Low'], data['Close'], 84)
        data['atr_252'] = atr(data['High'], data['Low'], data['Close'], 252)
        
        data['rsi_21'] = rsi(data['Close'], 21)
        data['rsi_84'] = rsi(data['Close'], 84)
        data['rsi_252'] = rsi(data['Close'], 252)
        
        data['vol_21'] = data['Close'].rolling(21).apply(lambda x : (vol(x)))
        data['vol_84'] = data['Close'].rolling(84).apply(lambda x : (vol(x)))
        data['vol_252'] = data['Close'].rolling(252).apply(lambda x : (vol(x)))
        
        #create proper date columns and day index 
        data['Date'] = data.Date.dt.date
        
        if pred == False: #get labels with data instead of predictions
            #apply crossing 5,10,15% thresholds
            data['gain_5'] = cross_thresh(data, period, 1.05, up = True)
            data['gain_10'] = cross_thresh(data, period, 1.1, up = True)
            data['gain_15'] = cross_thresh(data, period, 1.15, up = True)
            
            data['lose_5'] = cross_thresh(data, period, 0.95, up = False)
            data['lose_10'] = cross_thresh(data, period, 0.9, up = False)
            data['lose_15'] = cross_thresh(data, period, 0.85, up = False)
            #apply end 5,10,15% thresholds
            data['end_p5'] = (data['Close'].shift(-period) > data['Close'] * 1.05).astype(int)
            data['end_p10'] = (data['Close'].shift(-period) > data['Close'] * 1.1).astype(int)
            data['end_p15'] = (data['Close'].shift(-period) > data['Close'] * 1.15).astype(int)
            
            data['end_n5'] = (data['Close'].shift(-period) < data['Close'] * 0.95).astype(int)
            data['end_n10'] = (data['Close'].shift(-period) < data['Close'] * 0.9).astype(int)
            data['end_n15'] = (data['Close'].shift(-period) < data['Close'] * 0.85).astype(int)
        
        return data
    else:
        return np.nan

#additional back test function: find the price in 6mo and date
def price_date(data, period = 127, date_name = 'date_6mo', price_name = 'price_6mo'):
    end_dates = []
    end_prices = []
    try:
        #find the price and date 126 trading days ahead
        for i in range(len(data) - int(period)):
            end_dates.append(data['Date'].iloc[i + int(period)])
            end_prices.append(data['Close'].iloc[i + int(period)])
        #create a mask for the end of the dataframe where no data is
        masks = [-1] * int(period)
        end_dates = end_dates + masks
        end_prices = end_prices + masks
        data[date_name] = end_dates
        data[price_name] = end_prices
    except:
        print("exception in date!")
        masks = [-1] * len(data)
        data[date_name] = masks
        data[price_name] = masks
    return data
    
#additional back test function: find the price at 10% crossed within 6mo and date
def cross_value_price_date(data, period = 127, value = 0.1, date_name = 'date_10per', price_name = 'price_10per'):
    end_prices = []
    end_dates = []
    for i in range(len(data) - int(period)):
        threshold = data['Close'].iloc[i] * (1 + value)
        #subset the data to next 126 trading days
        sub_data = data.iloc[i: i + int(period)]
        sub_data.reset_index(inplace = True, drop = True)
        #add the day and price where its over 10% within next 126 trading days if it exists
        if sub_data['High'].gt(threshold).sum() > 0:
            cross_index = sub_data['High'].gt(threshold).idxmax()
            end_dates.append(sub_data['Date'].iloc[cross_index])
            end_prices.append(sub_data['High'].iloc[cross_index])
        else: #add masking -1 if it doesn't exist
            end_dates.append(-1)
            end_prices.append(-1)
    #create mask for the end of the dataframe where we don't have 6mo of data 
    masks = [-1] * int(period)
    end_dates = end_dates + masks
    end_prices = end_prices + masks
    data[date_name] = end_dates
    data[price_name] = end_prices
    return data

def cross_value_price_date2(data, period = 127, value = 0.1, date_name = 'date_10per', price_name = 'price_10per'):
    end_dates = []
    end_prices_high = []
    end_prices_low = []
    end_prices_close = []
    end_prices_thresh = []
    try:
        for i in range(len(data) - int(period)):
            threshold = data['Close'].iloc[i] * (1 + value)
            #subset the data to next 126 trading days
            sub_data = data.iloc[i: i + int(period)]
            sub_data.reset_index(inplace = True, drop = True)
            #add the day and price where its over 10% within next 126 trading days if it exists
            if sub_data['High'].gt(threshold).sum() > 0:
                cross_index = sub_data['High'].gt(threshold).idxmax()
                end_dates.append(sub_data['Date'].iloc[cross_index])
                end_prices_high.append(sub_data['High'].iloc[cross_index])
                end_prices_low.append(sub_data['Low'].iloc[cross_index])
                end_prices_close.append(sub_data['Close'].iloc[cross_index])
                end_prices_thresh.append(threshold)
            else: #add masking -1 if it doesn't exist
                end_dates.append(-1)
                end_prices_high.append(-1)
                end_prices_low.append(-1)
                end_prices_close.append(-1)
                end_prices_thresh.append(-1)
        #create mask for the end of the dataframe where we don't have 6mo of data 
        masks = [-1] * int(period)
        end_dates = end_dates + masks
        end_prices_high = end_prices_high + masks
        end_prices_low = end_prices_low + masks
        end_prices_close = end_prices_close + masks
        end_prices_thresh = end_prices_thresh + masks
        data[date_name] = end_dates
        data[price_name + '_high'] = end_prices_high
        data[price_name + '_low'] = end_prices_low
        data[price_name + '_close'] = end_prices_close
        data[price_name + '_thresh'] = end_prices_thresh
    except:
        print("Exception!")
        masks = [-1] * len(data)
        data[date_name] = masks
        data[price_name + '_high'] = masks
        data[price_name + '_low'] = masks
        data[price_name + '_close'] = masks
        data[price_name + '_thresh'] = masks
    return data


#put the two backtest date/price functions together to update data
def update_selldata(data, period, value, period_date_name, period_price_name, up_date_name, up_price_name):
    tickers = list(set(data['Ticker']))
    datas = []
    for i in tqdm(range(len(tickers))):
        dat = data[data['Ticker'] == tickers[i]]
        dat = price_date(dat, period, period_date_name, period_price_name)
        dat = cross_value_price_date2(dat, period, value, up_date_name, up_price_name)
        datas.append(dat)
    if len(tickers) > 1:
        return pd.concat(datas)
    else:
        return dat

### loop through all tickers and apply fuctions ###
#improve data extraction: Idea 1, use yfinacne tickers in a list. 
#@timeme
def get_data_yf(tickers, start, end, disk, name = None, pred = False, period = None):
    if period == None:
        period = 127
    data = yf.download(tickers, start, end,  progress=False, auto_adjust= False)
    datas = []
    for i in tqdm(range(len(tickers))):
        ticker = tickers[i]
        try:
            temp = data.xs(ticker, axis = 1, level = 1)
            temp.dropna(inplace = True)
            if len(temp) > 0:
                temp.reset_index(inplace = True)
                temp['Ticker'] = [ticker] * len(temp)
                temp = temp[['Ticker','Date','High','Low','Close', 'Volume']]
                temp = prepare_MLdata(period = period, data = temp, pred = pred)
                if isinstance(temp, pd.DataFrame):
                    datas.append(temp)
        except:
            continue
    data = pd.concat(datas)
    data.reset_index(inplace = True, drop = True)
    if disk == True:
        data.to_csv(name)
    else:
        return data
    
#%%
'''
Get data
'''
#%%
#read in saved ticker lists 
#download data
key = 'nJVBKg9okmibd0ojIvPJQo2CCzYn9piM'
tickers = urlopen(F"https://financialmodelingprep.com/api/v3/available-traded/list?apikey={key}")
tickers = tickers.read().decode("utf-8")
tickers = pd.DataFrame(json.loads(tickers))
#get them into both a list and a moreinfo dataframe
tickers =  tickers[tickers['exchangeShortName'].isin(['NASDAQ', 'ASX', 'AMEX', 'ETF', 'CBOE'])]
tickers = tickers[tickers['type'] != 'trust'] #no need to figure out how taxes and trading works right now
all_tickers = list(tickers['symbol'])
#clear out weird symbols (no need to figure these out right now)
all_tickers = [tick for tick in all_tickers if '/' not in tick]
all_tickers = [tick for tick in all_tickers if ' ' not in tick]
all_tickers = [tick for tick in all_tickers if '^' not in tick]
all_tickers = [tick for tick in all_tickers if '.' not in tick]
all_tickers = [tick for tick in all_tickers if '-' not in tick]

#tickers = tickers[tickers['symbol'].isin(all_tickers)]
#download a refresh of all data
data = get_data_yf(all_tickers, start = '2012-01-01', end = '2023-12-07', disk = True, name = 'securities_12072023_4mo.csv', period = 84)

#save the tickers successfully extracted for later
data = pd.read_csv('securities_12072023_6mo.csv')
modeled_tickers = list(set(data['Ticker']))


#%%% Idea 2 MUCH more data: https://site.financialmodelingprep.com/developer/docs#charts
##Probelm: paid for more than 250/calls per day & unclear if bulk calls are feasible
##Pros: way more data, faster per call and can get an active stock list from any exchange and way more other data types
def get_singlestock_data(key, ticker, start, end, time):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start}&to={end}&apikey={key}" 
    res = urlopen(url)
    data = res.read().decode("utf-8")
    data =  pd.DataFrame(json.loads(data)['historical'])
    data['ticker'] = [ticker] * len(data)
    return data#[['ticker','date', 'open', 'low', 'high', 'close', 'volume']]

#get data from multiple stocks,  2-3x faster than yfinace on individual stocks
## Can be bulked, but only 4 at a time and for 1 year, worth reinvestigating after paying
def bulk_url_builder(tickers, start, end, key):
    base_url = "https://financialmodelingprep.com/api/v3/historical-price-full/"
    for i in range(len(tickers)):
        base_url = base_url + f"{tickers[i]}," #add symbols
    base_url = base_url[:-1] #remove last comma 
    url_end = f"?from={start}&to={end}&apikey={key}"#add end url
    url = base_url + url_end
    return url
#%%
'''
Model construction and Assessment: Functions
'''
#%%
#data adjustment functions
def remove_outlier(df, thresh=1.5, cols=None):
    #drop columns with nan
    df = df.dropna(axis = 1)
    df = df._get_numeric_data()#drop(['Date', 'end_date', 'end_price'], axis = 1)
    df.apply(pd.to_numeric)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    #upper and lower bounds with a boolian mask
    lower_bound = Q1[cols] - thresh * IQR[cols]
    upper_bound = Q3[cols] + thresh * IQR[cols]
    outlier_mask = (df[cols] < lower_bound) | (df[cols] > upper_bound)
    #remove rows with outliers
    df = df[~outlier_mask.any(axis=1)]
    return df

def index_resample_tab(datax, datay):
    rus = RandomUnderSampler()
    rus.fit_resample(datax, datay)
    random.shuffle(rus.sample_indices_)
    datax = datax[rus.sample_indices_] 
    datay = datay[rus.sample_indices_] 
    return datax, datay

def remove_arr_outlier3(y_datas, x_datas, limit = 7):
    #get the means of each feature, and save values 7std away from the mean
    caps_upper = []
    caps_lower = []
    for i in range(x_datas.shape[-1]):
        mn = np.mean(x_datas[:,i])
        std = np.std(x_datas[:,i])
        caps_upper.append(mn + (limit * std))
        caps_lower.append(mn - (limit * std))
    #remove arrays that have values outside of 7 STD of the mean
    for i in tqdm(range(x_datas.shape[-1])):
        x_datas_clean = []
        y_datas_clean = []
        for j in range(x_datas.shape[0]):
            check = (x_datas[j,i] > caps_upper[i]) or (x_datas[j,i] < caps_lower[i])
            if check == False:
                x_datas_clean.append(x_datas[j,:])
                y_datas_clean.append(y_datas[j])
        x_datas = np.array(x_datas_clean)
        y_datas = np.array(y_datas_clean)
    return x_datas, y_datas

#model buidling functions
def build_dense(x2_shape1, dense_ct, dropout, lr, dense_act):
    x_inputs2 = Input(shape = (x2_shape1))
    x2 = Dense(dense_ct, activation = dense_act)(x_inputs2)
    x2 = Dense(int(dense_ct/2), activation = dense_act)(x2)
    x2 = Dense(5, activation = dense_act)(x2)
    x2 = Dropout(dropout)(x2)
    x_out = Dense(1, activation = 'sigmoid')(x2)
    dense_mod = Model(inputs = x_inputs2, outputs = x_out)
    #set the optimizer
    opt = Nadam(learning_rate = lr)
    dense_mod.compile(optimizer = opt, loss = 'binary_crossentropy',
                     metrics = ['acc'])
    return dense_mod

def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    
    Stolen straight from: https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p

def preprocessing_pipe(data, price_norm_colz, x_columns, y_columns, price_bounds,
                       date_cuts, date_props, max_data_points, outlier_limit, test_size):
    '''
    Parameters
    ----------
    data : df containing all data needed for at least ML
    price_norm_colz : column names in a list to be normalized to close price
    x_columns : x column names in a list 
    y_columns : y column names in a list 
    price_bounds : upper and lower price bounds in a list
    date_cuts : date cut-offs for each section to be sampled from differently in a list
    date_props : proportion of datapoints to sample from each date section in a list of the same order of date_cuts
    max_data_points : max data points to sample before balancing
    outlier_limit : number of standard deviations away from mean to be considered an outlier to remove
    test_size : number of data points in the test data before rebalancing
    '''
    data = data.dropna()
    data = data[((data['Close'] > price_bounds[0]) & (data['Close'] < price_bounds[1]))]
    #normalize columns tied to stock price
    for i in range(len(price_norm_colz)):
        col = price_norm_colz[i]
        data[col] = data[col]/data['Close']
    #split data into 4 temporal quarters
    cuts = []
    for i in range(len(date_cuts)):
        cuts.append(pd.to_datetime(date_cuts[i]).date())
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    subsets = []
    subsets.append(data[data['Date'] <= cuts[0]])
    for i in range(1, len(cuts)):
        subsets.append(data[((data['Date'] > cuts[i - 1]) & (data['Date'] <= cuts[i]))])
    #sample datapoints for training
    train_data = []
    for i in range(len(date_props)):
        train_data.append(subsets[i].sample(int(max_data_points * date_props[i])))
    train_data = pd.concat(train_data)
    #convert to arrays
    x_train = np.array(train_data[x_columns])
    y_train = np.array(train_data[y_columns])
    #remove extreme outliers
    x_train, y_train = remove_arr_outlier3(y_train, x_train, limit = outlier_limit)
    #rebalance the data
    x_train, y_train = index_resample_tab(x_train, y_train) 
    #standardize
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    #Test data
    data_te = data[pd.to_datetime(data['Date']).dt.date > cuts[-1]]
    if len(data_te) <= test_size:
        test_size = len(data_te) - 1
    data_te = data_te.sample(int(test_size))
    x_test = np.array(data_te[x_columns])
    y_test = np.array(data_te[y_columns])
    x_test, y_test = index_resample_tab(x_test, y_test) 
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaler

def simul_preprocessing_pipe(data, price_norm_colz, x_columns, scaler):
    data = data.dropna()
    data = data.reset_index(drop = True)
    #normalize columns tied to stock price
    for i in range(len(price_norm_colz)):
        col = price_norm_colz[i]
        data[col] = data[col]/data['Close']
    #Extract data needed for ML
    x_data = np.array(data[x_columns])
    #standardize
    x_data = scaler.transform(x_data)
    return data, x_data

#%%
'''
Data finalization
'''
#%%

data_6mo = pd.read_csv('securities_12072023_6mo.csv')
data_4mo = pd.read_csv('securities_12072023_4mo.csv')

x_columns = ['Volume','bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 
          'bb_upp84', 'bb_low84', 'bb_center252', 'bb_upp252', 'bb_low252', 
          'minmax21', 'minmax84','minmax252', 'atr_21', 'atr_84', 'atr_252',
          'rsi_21', 'rsi_84', 'rsi_252', 'vol_21', 'vol_84', 'vol_252']

y_columns = ['gain_10']

price_norm_colz = ['bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84', 
                  'bb_low84', 'bb_center252',  'bb_upp252', 'bb_low252', 'atr_21',
                  'atr_84', 'atr_252']
price_bounds = [2, 100]
date_cuts = ['2016-01-08', '2018-01-13', '2020-01-16', '2021-10-01']
date_props = [0.2, 0.2, 0.25, 0.35]
max_data_points = 5.0e5
outlier_limit = 4.5
test_size = 1e5

### 6 month models ###

#preprocess data for model building
preprocess_6mo = preprocessing_pipe(data_6mo, price_norm_colz, x_columns, 
                                    y_columns, price_bounds, date_cuts, 
                                    date_props, max_data_points, outlier_limit, 
                                    test_size)
x_train6, x_test6, y_train6, y_test6, scaler6 = preprocess_6mo
#build models
models_6mo = build_all_models(x_train6, y_train6.reshape(-1,))
models_6mo = list(models_6mo) + [scaler6, 1]
models_6mo = [models_6mo[3], models_6mo[0], models_6mo[1], models_6mo[2], models_6mo[4], models_6mo[5], models_6mo[6]]
#create predictions on train and test data
pred_tr = predict_purchases(x_train6, models_6mo, conf = 0.5)
pred_te = predict_purchases(x_test6, models_6mo, conf = 0.5)
#evaluate preformace on train and test data
heatmap(y_train6, pred_tr['pred'], 'train')
heatmap(y_test6, pred_te['pred'], 'test')

#drop na and prepare data for simulation testing
sim_6mo = data_6mo[data_6mo['Date'] >= '2021-12-01']
sim_6mo, sim_6mo_xdata = simul_preprocessing_pipe(sim_6mo, price_norm_colz, 
                                                  x_columns, scaler6)
#make predictions
sim_6mo_preds = predict_purchases(sim_6mo_xdata, models_6mo, conf = 0.5)
#return predictions to dataframe and subset to price limits
sim_6mo['consensus'] = sim_6mo_preds['consensus']
sim_6mo['pred'] = sim_6mo_preds['pred']
sim_6mo = sim_6mo[((sim_6mo['Close'] > price_bounds[0]) & (sim_6mo['Close'] < price_bounds[1]))]

#evaluate prediction quality when unblanced
sim_6mo['gain_10'].sum()/len(sim_6mo)
heatmap(sim_6mo['gain_10'], sim_6mo['pred'], 'simulation 6mo unbalanced')
#evaulate when balanced
sim_6mo_pred_bal, sim_6mo_true_bal = index_resample_tab(sim_6mo['pred'].values.reshape(-1,1),
                                                        sim_6mo['gain_10'].values.reshape(-1,1))
sim_6mo_true_bal.mean()
heatmap(sim_6mo_true_bal, sim_6mo_pred_bal, 'simulation 6mo balanced')

sim_6mo.to_csv('sim_6mo.csv', index = False)

### 4 month models ###
preprocess_4mo = preprocessing_pipe(data_4mo, price_norm_colz, x_columns, 
                                    y_columns, price_bounds, date_cuts, 
                                    date_props, max_data_points, outlier_limit, 
                                    test_size)
x_train4, x_test4, y_train4, y_test4, scaler4 = preprocess_4mo
#build models
models_4mo = build_all_models(x_train4, y_train4.reshape(-1,))
models_4mo = list(models_4mo) + [scaler4, 1]
models_4mo = [models_4mo[3], models_4mo[0], models_4mo[1], models_4mo[2], models_4mo[4], models_4mo[5], models_4mo[6]]
#create predictions on train and test data
pred_tr4 = predict_purchases(x_train4, models_4mo, conf = 0.5)
pred_te4 = predict_purchases(x_test4, models_4mo, conf = 0.5)
#evaluate preformace on train and test data
heatmap(y_train4, pred_tr4['pred'], 'train 4')
heatmap(y_test4, pred_te4['pred'], 'test 4')

#drop na and prepare data for simulation testing
sim_4mo = data_4mo[data_4mo['Date'] >= '2021-12-01']
sim_4mo, sim_4mo_xdata = simul_preprocessing_pipe(sim_4mo, price_norm_colz, 
                                                  x_columns, scaler4)
#make predictions
sim_4mo_preds = predict_purchases(sim_4mo_xdata, models_4mo, conf = 0.5)
#return predictions to dataframe and subset to price limits
sim_4mo['consensus'] = sim_4mo_preds['consensus']
sim_4mo['pred'] = sim_4mo_preds['pred']
sim_4mo = sim_4mo[((sim_4mo['Close'] > price_bounds[0]) & (sim_4mo['Close'] < price_bounds[1]))]
#evaluate prediction quality when unblanced
sim_4mo['gain_10'].sum()/len(sim_4mo)
heatmap(sim_4mo['gain_10'], sim_4mo['pred'], 'simulation 4mo unbalanced')
#evaulate when balanced
sim_4mo_pred_bal, sim_4mo_true_bal = index_resample_tab(sim_4mo['pred'].values.reshape(-1,1),
                                                        sim_4mo['gain_10'].values.reshape(-1,1))
sim_4mo_true_bal.mean()
heatmap(sim_4mo_true_bal, sim_4mo_pred_bal, 'simulation 4mo balanced')

sim_4mo.to_csv('sim_4mo.csv', index = False)



#original code to be deleted?

data_orig = data.copy()
data = data_orig.copy()
#remove na
data = data_6mo.dropna()
data = data[((data['Close'] > 2) & (data['Close'] < 100))]
#normalize columns tied to stock price
price_norm_colz = ['bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84', 
                  'bb_low84', 'bb_center252',  'bb_upp252', 'bb_low252', 'atr_21',
                  'atr_84', 'atr_252']
for i in range(len(price_norm_colz)):
    col = price_norm_colz[i]
    data[col] = data[col]/data['Close']
#data.columns
#subet the data to whats needed to build the models


#split data into 4 temporal quarters
cut1 = pd.to_datetime('2016-01-08').date()
cut2 = pd.to_datetime('2018-01-13').date()
cut3 = pd.to_datetime('2020-01-16').date()
cut4 = pd.to_datetime('2021-10-01').date() #change this for temporal test data
#convert to datetime for speed
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.date
#subset
datas_tr1 = data[data['Date'] <= cut1]
datas_tr2 = data[((data['Date'] > cut1) & (data['Date'] <= cut2))]
datas_tr3 = data[((data['Date'] > cut2) & (data['Date'] <= cut3))]
datas_tr4 = data[((data['Date'] > cut3) & (data['Date'] <= cut4))]

#sample the data
max_data_points = 1.0e6
datas_tr1 = datas_tr1.sample(int(max_data_points * 0.2))
datas_tr2 = datas_tr2.sample(int(max_data_points * 0.2))
datas_tr3 = datas_tr3.sample(int(max_data_points * 0.25))
datas_tr4 = datas_tr4.sample(int(max_data_points * 0.35))

#combine data into one dataframe
datas_tr = pd.concat([datas_tr1, datas_tr2, datas_tr3, datas_tr4])

#convert to arrays
x_train = np.array(datas_tr[x_columns])
y_train = np.array(datas_tr[y_columns])
#remove extreme outliers
x_train, y_train = remove_arr_outlier3(y_train, x_train, limit = 4.5)

#rebalance the data
x_train, y_train = index_resample_tab(x_train, y_train) #1.423Mil datapoints remain
#standardize
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

#Test data: temporal
data_te = data[pd.to_datetime(data['Date']).dt.date > cut4]
data_te = data_te[((data_te['Close'] < 100) & (data_te['Close'] > 2))]

data_te = data_te.sample(100000)
x_test = np.array(data_te[x_columns])
y_test = np.array(data_te[y_columns])
x_test, y_test = index_resample_tab(x_test, y_test) 
x_test = scaler.transform(x_test)


scaler6 = scaler
x_train6 = x_train.copy()
y_train6 = y_train.copy()
x_test6 = x_test.copy()
y_test6 = y_test.copy()



#%%
'''
Model Building
'''
#%%
#%%%Build models from scratch
#random forest
rfc = RandomForestClassifier(max_depth = 7)
rfc.fit(x_train, y_train)
#xgboost
xgbc = xgb.XGBClassifier(tree_method="hist")
xgbc.fit(x_train, y_train)
#MARS
mars = pyearth.Earth()
mars.fit(x_train, y_train)
#logistic regression
lrc = LogisticRegression()
lrc.fit(x_train, y_train)
#Dense ANN
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)

dense = build_dense(x_train.shape[1], dense_ct = 14, dropout = 0.5,
                    lr = lr_schedule, dense_act = 'tanh')

dense.fit(x_train, y_train, epochs = 9, batch_size = 128)


def build_all_models(x_train, y_train):
    rfc = RandomForestClassifier(max_depth = 7)
    rfc.fit(x_train, y_train)
    #xgboost
    xgbc = xgb.XGBClassifier(tree_method="hist")
    xgbc.fit(x_train, y_train)
    #MARS
    mars = pyearth.Earth()
    mars.fit(x_train, y_train)
    #logistic regression
    lrc = LogisticRegression()
    lrc.fit(x_train, y_train)
    #Dense ANN
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)

    dense = build_dense(x_train.shape[1], dense_ct = 14, dropout = 0.5,
                        lr = lr_schedule, dense_act = 'tanh')

    dense.fit(x_train, y_train, epochs = 9, batch_size = 128)
    return rfc, xgbc, mars, lrc, dense


models_6mo = build_all_models(x_train6, y_train6.reshape(-1,))
models_6mo = [models_6mo[3], models_6mo[0], models_6mo[1], models_6mo[2], models_6mo[4], models_6mo[5], models_6mo[6]]
models_4mo = build_all_models(x_train4, y_train4.reshape(-1,))



#%%%retrive old models
def get_models(lrc_path, rf_path, xgb_path, mars_path, scaler_path, dense_path, tickers_path):
    print('Downloading models...')
    lrc = joblib.load(BytesIO(requests.get(lrc_path).content))
    rfc = joblib.load(BytesIO(requests.get(rf_path).content))
    xgbc = joblib.load(BytesIO(requests.get(xgb_path).content))
    mars = joblib.load(BytesIO(requests.get(mars_path).content))
    tab_scaler = joblib.load(BytesIO(requests.get(scaler_path).content))
    dense_r = requests.get(dense_path)    
    open('dense.h5', 'wb').write(dense_r.content)
    dense = tf.keras.models.load_model('dense.h5', compile = False)
    
    #get the valid tickers
    valid_tickers = pd.read_csv(BytesIO(requests.get(tickers_path).content))
    valid_tickers = list(valid_tickers.iloc[:,1])
    return lrc, rfc, xgbc, mars, tab_scaler, dense, valid_tickers

temp_dir = tempfile.mkdtemp() #make a temp directory to house the .h5 file
os.chdir(temp_dir)

#supress unecessary tensorflow warnings    
logging.getLogger('tensorflow').setLevel(50)

#set the urls
lrc_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/lrc.joblib'
rf_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/rf.joblib'
xgb_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/xgbc.joblib'
mars_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/mars.joblib'
scaler_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/tabular_scaler.joblib'
dense_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/dense.h5'
tickers_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/v0.0.1/valid_tickers.csv'

lrc, rfc, xgbc, mars, tab_scaler, dense, valid_tickers = get_models(lrc_path, rf_path, xgb_path, mars_path, scaler_path, dense_path, tickers_path)

#%%%save the models - if needed
MODEL_PATH = 'C:\\Users\\q23853\\Desktop\\random_trader\\production_models2\\dense.h5'
MODEL_PATH = 'C:\\Users\\q23853\Desktop\\random_trader\\processed_securities122023\\dense.h5'
dense.save(MODEL_PATH, include_optimizer = False, save_format='h5')


os.chdir('C:\\Users\\q23853\\Desktop\\random_trader\\production_models2')
joblib.dump(rfc, 'rf.joblib')
joblib.dump(xgbc, 'xgbc.joblib')
joblib.dump(mars, 'mars.joblib')
joblib.dump(lrc, 'lrc.joblib')
joblib.dump(scaler, 'tabular_scaler.joblib')
#%%
'''
Model assessment: functions
'''
#%%
def pca_graph(x_pca, model):
    if model == None:
        plt.scatter(x_pca[:,0], x_pca[:,1], c = x_pca[:,2], alpha = 0.1)
        plt.title('PCA with true labels')
        plt.xlabel('1st Componet')
        plt.ylabel('2nd Componet')
        plt.show()
    else:
        plt.scatter(x_pca[:,0], x_pca[:,1], c = x_pca[:,model], alpha = 0.1)
        if model == 3:
            model_name = 'RF'
        elif model == 4:
           model_name = 'XG Boost' 
        elif model == 5:
           model_name = 'MARS' 
        elif model == 6:
           model_name = 'LR' 
        else:
            model_name = 'FF-ANN'
        plt.title(f'PCA with {model_name} predicted labels')
        plt.xlabel('1st Componet')
        plt.ylabel('2nd Componet')
        plt.show()

def heatmap(ys, preds, dataset_type):
    acc = accuracy_score(ys, preds)
    prec = precision_score(ys, preds)
    conf = confusion_matrix(ys, preds)
    ax= plt.subplot()
    sns.heatmap(conf, annot=True, fmt='g', ax=ax); 
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(f'{dataset_type} Accuracy: ' + str(np.round(acc * 100, 2)) + '%' 
                 + f'\n {dataset_type} Precision: '+ str(np.round(prec * 100, 2)) + '%'); 
    ax.xaxis.set_ticklabels(['Do Not Buy', 'Buy']); ax.yaxis.set_ticklabels(['Do Not Buy', 'Buy']);
    plt.show()
    
def plot_roc(fpr, tpr, auc, multi, name = None):
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    if multi != True:
        plt.title(str(name) + ' AUC: ' + str(np.round(auc, 3)))
        plt.show()
    else:
        plt.title("ROC-AUC Curves")

def multi_roc(y_true, y_probs):
    for i in range(len(y_probs)):
        fpr , tpr , thresholds = roc_curve (y_true , y_probs[i])
        auc = roc_auc_score(y_true, y_probs[i])
        plot_roc(fpr, tpr, auc, multi = True)
        
#%%
'''
Model assessments
'''
#%%

#model train logits
rfc_proba = rfc.predict_proba(x_train)[:,1]
xgb_proba = xgbc.predict_proba(x_train)[:,1]
lr_proba = lrc.predict_proba(x_train)[:,1]
mars_train_proba = mars.predict(x_train)
dense_train_proba = dense.predict(x_train)

#model train predictions
rfc_train_pred = rfc.predict(x_train)
xgb_train_pred = xgbc.predict(x_train)
mars_train_pred = np.where(mars_train_proba >= 0.5, 1, 0)
lr_train_pred = lrc.predict(x_train)
dense_train_pred = np.where(dense_train_proba >= 0.5, 1, 0)

#test logit preds
rfc_proba = rfc.predict_proba(x_test)[:,1]
xgb_proba = xgbc.predict_proba(x_test)[:,1]
lr_proba = lrc.predict_proba(x_test)[:,1]
mars_proba = mars.predict(x_test)
dense_proba = dense.predict(x_test)

#test preds
rfc_test_pred = rfc.predict(x_test)
xgb_test_pred = xgbc.predict(x_test)
mars_test_pred = np.where(mars_proba >= 0.5, 1, 0)
lr_test_pred = lrc.predict(x_test)
dense_test_pred = np.where(dense_proba >= 0.5, 1, 0)

print(accuracy_score(rfc_train_pred, y_train))
print(confusion_matrix(dense_train_pred, y_train))

#model outputs (Mars and LR)
print(mars.summary())

lrc_coef = lrc.coef_.reshape(-1,1)
lrc_p = logit_pvalue(lrc, x_train)
lrc_p = lrc_p[1:].reshape(-1,1)
lrc_logodds = np.exp(lrc_coef)
lrc_summary = np.concatenate([lrc_coef, lrc_logodds, lrc_p], axis = 1)
lrc_summary = pd.DataFrame(lrc_summary, columns = ['coef', 'log_odds', 'p-val'])
lrc_summary['var'] = x_columns

#%%% Individual model assement on test data

#visualize the decision boundaries
#PCA
pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x_train)
x_pca = np.concatenate([x_pca, y_train.reshape(len(y_train), 1)], axis = 1)
x_pca = np.concatenate([x_pca, rfc_train_pred.reshape(len(rfc_train_pred), 1)], axis = 1)
x_pca = np.concatenate([x_pca, xgb_train_pred.reshape(len(xgb_train_pred), 1)], axis = 1)
x_pca = np.concatenate([x_pca, mars_train_pred.reshape(len(mars_train_pred), 1)], axis = 1)
x_pca = np.concatenate([x_pca, lr_train_pred.reshape(len(lr_train_pred), 1)], axis = 1)
x_pca = np.concatenate([x_pca, dense_train_pred.reshape(len(dense_train_pred), 1)], axis = 1)

pca_graph(x_pca, model = 6)

#visualize the confusion matrix, accuracy, precision
#evaluate
#training heatmaps
heatmap(y_train, rfc_train_pred, 'RF Training')
heatmap(y_train, xgb_train_pred, 'XGB Training')
heatmap(y_train, mars_train_pred, 'MARS Training')
heatmap(y_train, dense_train_pred, 'FF-ANN Training')
heatmap(y_train, lr_train_pred, 'LR Training')

#testing heatmaps
heatmap(y_test, rfc_test_pred, 'RF Test')
heatmap(y_test, xgb_test_pred, 'XGB Test')
heatmap(y_test, mars_test_pred, 'MARS Test')
heatmap(y_test, dense_test_pred, 'FF-ANN Test')
heatmap(y_test, lr_test_pred, 'LR Test')

#ROC curves for individual models
#all ROCs
multi_roc(y_test, [rfc_proba, xgb_proba, mars_proba, lr_proba, dense_proba])

#indiivudal ROC
proba = lr_proba
fpr , tpr , thresholds = roc_curve (y_test , proba)
auc = roc_auc_score(y_test, proba)
plot_roc(fpr, tpr, auc, multi = False, name = 'lrc')

#%%% Ensemble Model Assessment on Test data
#set up df for enseble calculations
pred_df = pd.DataFrame([rfc_proba, xgb_proba, lr_proba, mars_proba, dense_proba]).transpose()
pred_df.columns = ['rfc', 'xgb', 'lr', 'mars', 'dense']
pred_df['dense'] = pred_df['dense'].apply(lambda x: x[0]) #remove the list output of keras

#set up ensembles
pred_df['mean'] = pred_df[['rfc', 'xgb', 'lr', 'mars', 'dense']].mean(axis = 1)
pred_df['vote50'] = np.where(pred_df['mean'] > 0.5, 1,0)
pred_df['vote75'] = np.where(pred_df['mean'] > 0.75, 1,0)
pred_df['all50'] = np.where((pred_df[['rfc', 'xgb', 'mars', 'lr', 'dense']]>= 0.5).all(axis=1), 1,0)
pred_df['all75'] = np.where((pred_df[['rfc', 'xgb', 'mars', 'lr', 'dense']]>= 0.75).all(axis=1), 1,0)
pred_df['v75a50'] = np.where(((pred_df['vote75'] == 1) & (pred_df['all50'] == 1)), 1, 0)

#assess
#heatmaps
heatmap(y_test, pred_df['vote50'], 'Vote 50')
heatmap(y_test, pred_df['vote75'], 'Vote 75') #best balance for precision and restrictivity
heatmap(y_test, pred_df['all50'], 'All 50')
heatmap(y_test, pred_df['all75'], 'All 75')
heatmap(y_test, pred_df['v75a50'], 'All 50 and Vote 75')

#roc-auc 
proba = pred_df['mean']
fpr , tpr , thresholds = roc_curve (y_test , proba)
auc = roc_auc_score(y_test, proba)
plot_roc(fpr, tpr, auc, multi = False, name = 'Ensemble ROC')

### Make prediction frames for 6 vs 4 month models
models_6mo = list(models_6mo) + [scaler6, 1]
models_6mo = [models_6mo[3], models_6mo[0], models_6mo[1], models_6mo[2], models_6mo[4], models_6mo[5], models_6mo[6]]

models_4mo = list(models_4mo) + [scaler4, 1]
models_4mo = [models_4mo[3], models_4mo[0], models_4mo[1], models_4mo[2], models_4mo[4], models_4mo[5], models_4mo[6]]


simdat_6mo = data_6mo.dropna()
simdat_6mo = simdat_6mo[simdat_6mo['Date'] >= '2020-01-01']

pred_dat_6mo = simdat_6mo[x_columns]
pred_dat_6mo = scaler6.transform(pred_dat_6mo)
ydat_6mo = simdat_6mo[y_columns].values

simdat_4mo = data_4mo.dropna()
simdat_4mo = simdat_4mo[simdat_4mo['Date'] >= '2020-01-01']

pred_dat_4mo = simdat_4mo[x_columns]
pred_dat_4mo = scaler4.transform(pred_dat_4mo)
pred_dat_4mo_for_6mo = scaler6.transform(pred_dat_4mo)
ydat_4mo = simdat_4mo[y_columns].values

simdat_6mo.reset_index(inplace = True, drop = True)
simdat_4mo.reset_index(inplace = True, drop = True)

##############
pred_6mo = predict_purchases(pred_dat_6mo, models_6mo, conf = 0.75)
pred_6mo['pred_75'] = pred_6mo['pred']
pred_6mo['pred'] = np.where(pred_6mo['consensus'] >= 0.5, 1, 0)

pred_4mo = predict_purchases(pred_dat_4mo, models_4mo, conf = 0.75)
pred_4mo['pred_75'] = pred_4mo['pred']
pred_4mo['pred'] = np.where(pred_4mo['consensus'] >= 0.5, 1, 0)

pred_6on4 = predict_purchases(pred_dat_4mo_for_6mo, models_6mo, conf = 0.75)
pred_6on4['pred_75'] = pred_6on4['pred']
pred_6on4['pred'] = np.where(pred_6on4['consensus'] >= 0.5, 1, 0)

#visualize confusion matrices
heatmap(ydat_6mo, pred_6mo['pred_75'], '6mo 75')
heatmap(ydat_4mo, pred_4mo['pred_75'], '4mo 75')
heatmap(ydat_4mo, pred_6on4['pred_75'], '6 on 4mo 75')

heatmap(ydat_6mo, pred_6mo['pred'], '6mo 50')
heatmap(ydat_4mo, pred_4mo['pred'], '4mo 50')
heatmap(ydat_4mo, pred_6on4['pred'], '6 on 4mo 50')


simdat_4mo['pred'] = pred_4mo['pred']
simdat_4mo['pred_75'] = pred_4mo['pred_75']

simdat_6mo['pred'] = pred_6mo['pred']
simdat_6mo['pred_75'] = pred_6mo['pred_75']


simdat_6mo = simdat_6mo.dropna()
heatmap(simdat_6mo['gain_10'] , simdat_6mo['pred'] , '6 on data')


x_test4, y_test4 = index_resample_tab(np.array(pred_6on4), ydat_4mo.reshape(-1,1)) 

heatmap(y_test2,list(x_test2[:, -1]), '6mo')
heatmap(y_test3, list(x_test3[:, -1]), '4mo')
heatmap(y_test4, list(x_test4[:, -1]), '4mo')


simdat_4mo.to_csv('simdat_4mo.csv', index = False)
simdat_6mo.to_csv('simdat_6mo.csv', index = False)


tickers = list(set(simdat_6mo['Ticker']))
simdat_6mo2 = apply_sell_over(simdat_6mo, tickers, 127, 'date_6mo', 
                       'price_6mo', 'date_10per', 
                       'price_10per', value = 0.1)

tickers = list(set(simdat_4mo['Ticker']))
simdat_4mo2 = apply_sell_over(simdat_4mo, tickers, 85, 'date_4mo', 
                       'price_4mo', 'date_10per', 
                       'price_10per', value = 0.1)

simdat_4mo2.to_csv('simdat_4mo.csv', index = False)
simdat_6mo2.to_csv('simdat_6mo.csv', index = False)


heatmap(simdat_4mo['gain_10'], simdat_4mo['pred'], '4mo 50 from data')

sum(simdat_6mo2['price_10per'])
######## explore options for bounding mars - hard bound best ########
def logistic(x, k=0.1, mid = 0.5):
    s = 1 / (1 + np.exp(-k * (x - mid)))
    return s


preds['mars_soft.1'] = preds['mars'].apply(lambda x: logistic(x, 0.1))
preds['mars_soft.25'] = preds['mars'].apply(lambda x: logistic(x,0.25))
preds['mars_soft.5'] = preds['mars'].apply(lambda x: logistic(x, 0.5))
preds['mars_soft1'] = preds['mars'].apply(lambda x: math.erf(x))

z = preds.sample(20000)

preds['mean_soft.1'] = preds[['rf', 'xgb', 'lrc', 'mars_soft.1', 'dense']].mean(axis = 1)
preds['mean_soft.25'] = preds[['rf', 'xgb', 'lrc', 'mars_soft.25', 'dense']].mean(axis = 1)
preds['mean_soft.5'] = preds[['rf', 'xgb', 'lrc', 'mars_soft.5', 'dense']].mean(axis = 1)
preds['mean_soft1'] = preds[['rf', 'xgb', 'lrc', 'mars_soft1', 'dense']].mean(axis = 1)

preds['pred_soft.1'] = np.where(preds['mean_soft.1'] > 0.75, 1, 0)
preds['pred_soft.25'] = np.where(preds['mean_soft.25'] > 0.75, 1, 0)
preds['pred_soft.5'] = np.where(preds['mean_soft.5'] > 0.75, 1, 0)
preds['pred_soft1'] = np.where(preds['mean_soft1'] > 0.75, 1, 0)

heatmap(y_test, preds['pred_soft.1'], 'pred_soft.1')
heatmap(y_test, preds['pred_soft.25'], 'soft.25')
heatmap(y_test, preds['pred_soft.5'], 'soft.5')
heatmap(y_test, preds['pred_soft1'], 'soft1')


z = preds[((preds['mars'] < 2) & (preds['mars'] > -1))]

plt.scatter(z['mars_soft.5'], z['mars'])

z = preds.sample(50000)

z = z[['mars', 'mars_soft.1', 'mars_soft.25', 'mars_soft.5']]



#%%
'''
Model Monitoring: Make predictions to check later
'''
#%%
#load all models
def get_models(lrc_path, rf_path, xgb_path, mars_path, scaler_path, dense_path, tickers_path):
    print('Downloading models...')
    lrc = joblib.load(BytesIO(requests.get(lrc_path).content))
    rfc = joblib.load(BytesIO(requests.get(rf_path).content))
    xgbc = joblib.load(BytesIO(requests.get(xgb_path).content))
    mars = joblib.load(BytesIO(requests.get(mars_path).content))
    tab_scaler = joblib.load(BytesIO(requests.get(scaler_path).content))
    dense_r = requests.get(dense_path)    
    open('dense.h5', 'wb').write(dense_r.content)
    dense = tf.keras.models.load_model('dense.h5', compile = False)
    
    #get the valid tickers
    valid_tickers = pd.read_csv(BytesIO(requests.get(tickers_path).content))
    valid_tickers = list(valid_tickers.iloc[:,1])
    return lrc, rfc, xgbc, mars, dense, tab_scaler, valid_tickers

def prep_4pred(data, price_norm_colz, x_columns, scaler):
    data = data.copy()
    data.dropna(inplace = True)
    for i in range(len(price_norm_colz)):
        col = price_norm_colz[i]
        data[col] = data[col]/data['Close']
    x_data = data[x_columns]
    x_data = scaler.transform(x_data)
    ticker_col = data['Ticker']
    dates = data['Date']
    return x_data, ticker_col, dates
    
def predict_purchases(data, models, conf = 0.75):
    #unpack models
    lrc, rfc, xgbc, mars, dense, tab_scaler, valid_tickers = models
    #make predictions
    print('\nMaking predictions... ')
    #make predictions
    rfc_prd = rfc.predict_proba(data)[:,1]
    xgbc_prd = xgbc.predict_proba(data)[:,1]
    mars_prd = mars.predict(data)
    lrc_prd = lrc.predict_proba(data)[:,1]
    dense_prd = dense.predict(data)
    #set a prediction frame for ensemble strategy
    pred_frame = pd.DataFrame([rfc_prd, xgbc_prd, mars_prd, lrc_prd, dense_prd]).transpose()
    pred_frame.columns = ['rf', 'xgb', 'mars', 'lrc', 'dense']
    pred_frame['dense'] = pred_frame['dense'].apply(lambda x: x[0])
    #bound mars (sig/log/erf all seemed worse at curosry glance in terms of precision-resriction trade off)
    pred_frame['mars'] = np.where(pred_frame['mars'] <= 0, 0, np.where(pred_frame['mars'] >= 1, 1, pred_frame['mars']))
    #make final prediction
    consensus = pred_frame.mean(axis = 1)
    buy = np.where(consensus>= conf, 1, 0)
    pred_frame['consensus'] = consensus
    pred_frame['pred'] = buy
    return pred_frame
    
def monitor_preds(models, tickers, current_file_path, start = '2022-06-01', 
                  end = '2023-12-17', cut_off = '2023-12-01',
                  price_norm_colz = None, x_columns = None):
    current_monitor = pd.read_csv(current_file_path)
    print('Downloading data...')
    data = get_data_yf(tickers, start, end, disk = False, name = None, pred = True)
    xdata, ticker_names, dates = prep_4pred(data = data, price_norm_colz = price_norm_colz,
                      x_columns = x_columns, scaler = models[5])
    pred_df = predict_purchases(xdata, models)
    pred_df['Date'] = dates.values
    pred_df['Ticker'] = ticker_names.values
    closes = pd.merge(pred_df, data, how = 'left', on = ['Date', 'Ticker'])
    pred_df['Close'] = closes['Close']
    pred_df = pred_df[pred_df['Date'].astype(str) >= cut_off]
    pred_df = pd.concat([current_monitor, pred_df])
    pred_df.drop_duplicates(['Ticker', 'Date'])
    pred_df.to_csv(current_file_path, index = False)
    return pred_df
    

start = '2022-01-01'
end = (date.today()).strftime('%Y-%m-%d') 
cut_off = '2023-12-01'

lrc_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/lrc.joblib'
rf_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/rf.joblib'
xgb_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/xgbc.joblib'
mars_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/mars.joblib'
scaler_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/tabular_scaler.joblib'
dense_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/dense.h5'
tickers_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/v0.0.1/valid_tickers.csv'


price_norm_colz = ['bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84', 
                  'bb_low84', 'bb_center252',  'bb_upp252', 'bb_low252', 'atr_21',
                  'atr_84', 'atr_252']

x_columns = ['Volume','bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 
          'bb_upp84', 'bb_low84', 'bb_center252', 'bb_upp252', 'bb_low252', 
          'minmax21', 'minmax84','minmax252', 'atr_21', 'atr_84', 'atr_252',
          'rsi_21', 'rsi_84', 'rsi_252', 'vol_21', 'vol_84', 'vol_252']

models = get_models(lrc_path, rf_path, xgb_path, mars_path, scaler_path, dense_path, tickers_path)
scaler = models[5]
#

current_file_path ='C:\\Users\\q23853\\Desktop\\random_trader\\monitor\\monitor.csv'

tickers = models[6]
tickers = random.sample(tickers, 750)

pred = monitor_preds(models, tickers, current_file_path, cut_off = cut_off,
                     price_norm_colz = price_norm_colz, x_columns = x_columns,
                     end = end)

max_dt = pd.to_datetime(pred['Date']).dt.date.max()

pred = pred[((pred['Date'] == max_dt) & (pred['Close'] > 2) & (pred['Close'] < 100))]

#%%
'''
Trading Simulations: functions
'''
#%%
def hold_cond(row, i):
    row = row.iloc[i]
    cond1 = ((row['price_10per'] == -1) & (row['date_6mo'] >= end_date))
    cond2 = ((row['price_10per'] != -1) & (row['date_10per'] >= end_date))
    if  cond1 | cond2:
        return 1
    else:
        return 0

def sale_replace(df, sale_col, date_col, bank, data, trading_dates, max_date_ind):
    bank += df[sale_col] * df['share_num']
    #print(bank)
    sell_date = df[date_col]
    new_buy, bank = build_portfolio(bank = bank, number = 1, 
                                      trading_dates = trading_dates, data = data, 
                                      max_date_ind = max_date_ind, 
                                      buy_date = sell_date)
    return new_buy, bank

def get_data_random_date(data, max_date_ind, trading_dates):
    bdate_ind = random.randint(0,max_date_ind)
    bdate = trading_dates[bdate_ind]
    bdate = bdate.replace('-', '')
    temp_data = data.query(f"Datetime == {bdate}")
    return temp_data


def build_portfolio(bank, number, trading_dates, data, max_date_ind,
                    buy_date = None):
    #pick a buying date
    if buy_date == None:
        counter = 0
        temp_data = get_data_random_date(data, max_date_ind, trading_dates)
        while ((counter < 25) & (len(temp_data) < number)): #keep trying dates until a day has enough data
            temp_data = get_data_random_date(data, max_date_ind, trading_dates)
            counter += 1            
    else:
        bdate = buy_date
        bdate = bdate.replace('-', '')
        #subset the data to that date
        temp_data = data.query(f"Datetime == {bdate}")
    #sample to the desired number of stocks
    if len(temp_data) < number:
        number = len(temp_data)
    if len(temp_data) == 0:
        print('guy')
    temp_data = temp_data.sample(number)
    #evenly allocate funds from the bank to each purchased stock
    allocation = bank/number
    temp_data['share_num'] = np.floor(allocation/temp_data['Close'])
    temp_data['cost'] = temp_data['share_num'] * temp_data['Close']
    #update the bank
    total_cost = temp_data['cost'].sum()
    bank -= total_cost
    return temp_data, bank

def sell_over_value(data, period, value, period_date_name, 
                       period_price_name, up_date_name = 'date_10per', 
                       up_price_name = 'price_10per'): #slow as heck, gotta be a way to vectorize?
    thresh_end_prices = []
    thresh_end_dates = []
    end_dates = []
    end_prices = []
    thresh_end_dates = []
    thresh_end_prices = []
    if len(data) > int(period) + 1:
        for i in (range(len(data) - int(period))):
            #get the end date/price regaurdless of if it increases in price
            end_dates.append(data['Date'].iloc[i + int(period)])
            end_prices.append(data['Close'].iloc[i + int(period)])
            #get the end date/price if it increases in price
            threshold = data['Close'].iloc[i] * (1 + value)
            #subset the data to next 126 trading days
            sub_data = data.iloc[i: i + int(period)]
            sub_data.reset_index(inplace = True, drop = True)
            #add the day and price where its over 10% within next 126 trading days if it exists
            if sub_data['High'].gt(threshold).sum() > 0:
                cross_index = sub_data['High'].gt(threshold).idxmax()
                thresh_end_dates.append(sub_data['Date'].iloc[cross_index])
                thresh_end_prices.append(sub_data['Close'].iloc[cross_index]) #use close since we will never reliably catch the highs
            else: #add masking -1 if it doesn't exist
                thresh_end_dates.append(-1)
                thresh_end_prices.append(-1)
    #create mask for the end of the dataframe where we don't have 6mo of data 
        masks = [-1] * int(period)
        data[up_date_name] = thresh_end_dates + masks
        data[up_price_name] = thresh_end_prices + masks
        data[period_date_name] = end_dates + masks
        data[period_price_name] = end_prices + masks
    else: #if data too small, just mark all as -1
        masks = [-1] * len(data)
        data[up_date_name] = masks
        data[up_price_name] = masks
        data[period_date_name] = masks
        data[period_price_name] = masks
    return data

#get the dates/prices 6mo in advance and at 10% thresh
def apply_sell_over(data, tickers, period, period_date_name, 
                       period_price_name, up_date_name, 
                       up_price_name, value = 0.1):
    datas = []
    for i in tqdm(range(len(tickers))):
        temp = data[data['Ticker'] == tickers[i]]
        datas.append(sell_over_value(temp, period, value, period_date_name, 
                               period_price_name, up_date_name, up_price_name))
    return pd.concat(datas)

def apply_master_cond(act_df, end_date, up_sell_col, up_time_col, end_sell_col, end_time_col):
    cond1 = act_df[up_sell_col] == -1
    try: #if the vector is just -1, cond2 is false
        cond2 = act_df[up_time_col] < end_date
    except:
        cond2 = False
    cond3 = act_df[end_time_col] < end_date
    master_cond = np.where(((cond1 == True) & (cond3 == False)), 0,
            np.where(((cond1 == False) & (cond2 == True)), 1,
            np.where(((cond1 == True) & (cond3 == True)), 2,
            np.where(((cond1 == False) & (cond2 == False)), 0, 0))))
     
    act_df['cond'] = master_cond
    return act_df

def find_last_valid_day(df, data, end_datequery):
    #data must have datetime and tickername multiindex
    #loop through dataframe fixing those who are nan on sale day
    for i in range(len(df)):
        if np.isnan(df['end_price'].iloc[i]):
            sec = df['Ticker'].iloc[i]
            lookup_df = data.query(f"(Datetime <= {end_datequery}) & (Ticker == '{sec}')") #get the data up to closest to end end date
            df['end_price'].iloc[i] = lookup_df['Close'].iloc[-1]
    return df

def init_trader(bank, number, trading_dates, buy_data, max_date_ind,
                up_sell_col, end_time):
    portfolio, bank = build_portfolio(bank = bank, number = number, 
                                      trading_dates = trading_dates, data = buy_data, 
                                      max_date_ind = max_date_ind)
    #get final date
    start_date = portfolio['Date'].iloc[0]
    ## Old Version ##
    end_date = trading_dates[trading_dates.index(start_date) + end_time]
    ## New Version ##
    #start_dt = datetime.strptime(start_date, '%Y-%m-%d')   
    #end_dt = start_dt + timedelta(days = end_time)
    #find the closest date to 1 year after start
    #end_date = datetime.strftime(min(all_trading_dates, key = lambda d: abs(d - end_dt)), '%Y-%m-%d')
    
    #subset all -1s to one dataframe
    neg_df = portfolio[portfolio[up_sell_col] == -1]
    #subset all else to activedf
    act_df = portfolio[portfolio[up_sell_col] != -1]
    return act_df, neg_df, bank, end_date, start_date

def trade_neg_df(neg_df, act_df, bank, end_date, data, up_sell_col = 'price_10per', 
                 end_time_col = 'date_6mo', end_sell_col = 'price_6mo'):
    cond1 = (len(neg_df) > 0)
    try: 
        cond2 = ((neg_df[end_time_col].iloc[0] < end_date)) 
    except: 
        cond2 = False
    while  (cond1 & cond2):
        bank += sum(neg_df[end_sell_col] * neg_df['share_num'])
        sell_date = neg_df[end_time_col].iloc[0]
        neg_df, bank = build_portfolio(bank = bank, number = len(neg_df), 
                                          trading_dates = trading_dates, data = data, 
                                          max_date_ind = max_date_ind, 
                                          buy_date = sell_date)
        #send new non -1's to active df, keep the rest
        act_df = pd.concat([act_df, neg_df[neg_df[up_sell_col] != -1]])
        neg_df = neg_df[neg_df[up_sell_col] == -1]
        cond1 = (len(neg_df) > 0)
        try: 
            cond2 = ((neg_df[end_time_col].iloc[0] < end_date)) 
        except: 
            cond2 = False
    neg_df['cond'] = [0] * len(neg_df)
    return act_df, neg_df, bank
    

def active_trader(act_df, hold_df, bank, buy_data, trading_dates, max_date_ind, end_date,
                  up_sell_col, up_time_col,
                  end_sell_col, end_time_col):
    def lam_sale_replace(row, data): #encapsulate the lambda bank operation
        nonlocal bank
        #print(bank)
        if row['cond'] == 1:
            sale_col = up_sell_col
            date_col = up_time_col
            new_row, bank = sale_replace(row, sale_col, date_col, bank, buy_data, 
                                         trading_dates, max_date_ind)
        elif row['cond'] == 2:
            sale_col = end_sell_col
            date_col = end_time_col
            new_row, bank = sale_replace(row, sale_col, date_col, bank, buy_data,
                                         trading_dates, max_date_ind)
        return new_row.iloc[0]
    bank = bank
    while len(act_df) > 0:
        act_df = act_df.apply(lambda x: lam_sale_replace(x, buy_data), axis = 1)
        act_df = apply_master_cond(act_df, end_date, up_sell_col, up_time_col, end_sell_col, end_time_col)
        new_hold = act_df[act_df['cond'] == 0]#.iloc[:,-1]
        if isinstance(new_hold, pd.DataFrame):
            hold_df = pd.concat([hold_df, new_hold])
        act_df = act_df[act_df['cond'] != 0]
    return hold_df, bank
     
def final_sale(bank, hold_df, end_data, end_date):
     #get the price on end date, then sell on end date
     end_datequery = end_date.replace('-', '')
     sell_frame = end_data.query(f"Datetime == {end_datequery}")
     
     hold_df['end_price'] = pd.merge(hold_df, sell_frame, how = 'left', on = 'Ticker')['Close_y'].values
     
     sell_bank = sum(hold_df['end_price'] * hold_df['share_num'])
     if np.isnan(sell_bank):
         hold_df = find_last_valid_day(hold_df, end_data, end_datequery)    
         bank += sum(hold_df['end_price'] * hold_df['share_num'])
     else:
         bank += sell_bank
     
     return bank, hold_df
 
def find_last_valid_day(df, data, end_datequery):
    #data must have datetime and tickername multiindex
    #loop through dataframe fixing those who are nan on sale day
    for i in range(len(df)):
        if np.isnan(df['end_price'].iloc[i]):
            sec = df['Ticker'].iloc[i]
            try:
                #get the data up to closest to end end date of that ticker
                lookup_df = data.query(f"(Datetime <= {end_datequery})")
                lookup_df = lookup_df[lookup_df['Ticker'] == sec] #get the data up to closest to end end date
                df['end_price'].iloc[i] = lookup_df['Close'].iloc[-1]
            except:
                df['end_price'].iloc[i] = lookup_df['Close'].iloc[-1]
    return df


def trade_simulation(bank_size, number, trading_dates, buy_data, end_data, end_time, 
                     max_date_ind, up_sell_col, up_time_col, end_sell_col, end_time_col,
                     iterations):
    banks = []
    final_ports = []
    end_dates = []
    start_dates = []
    for i in tqdm(range(iterations)):
        try: #  REMOVE THIS AND BE SURE THE DATA IS CORRECT INSTEAD
            bank = bank_size  
            #set up the trader with an active df and non-increase in value df
            act_df, neg_df, bank, end_date, start_date = init_trader(bank, number, trading_dates,
                                         buy_data, max_date_ind,
                                         up_sell_col, end_time)
            #trade the stocks which never go up to the end
            act_df, hold_df, bank = trade_neg_df(neg_df, act_df, bank, end_date, 
                                                 buy_data, up_sell_col, end_time_col,
                                                 end_sell_col)
            #set the conditions for the active trades
            act_df = apply_master_cond(act_df, end_date, up_sell_col, up_time_col, end_sell_col, end_time_col)
            new_hold = act_df[act_df['cond'] == 0]
            hold_df = pd.concat([hold_df, new_hold])
            act_df = act_df[act_df['cond'] != 0]
            #trade the active dataframe to the end #NEED TO EDIT LAMBDA GLOBAL
            hold_df, bank = active_trader(act_df, hold_df, bank, buy_data, trading_dates, 
                                          max_date_ind, end_date, up_sell_col, up_time_col,
                                          end_sell_col, end_time_col)
            final_bank, end_portfolio = final_sale(bank, hold_df, end_data, end_date)
            banks.append(final_bank)
            final_ports.append(end_portfolio)
            end_dates.append(end_date)
            start_dates.append(start_date)
        except:
            continue
    return banks, final_ports, end_dates, start_dates

def datetimeticker_index(df):
    df['Datetime'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Datetime', 'Ticker'])
    return df

def datetime_index(df):
    df['Datetime'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Datetime'])
    return df

#%%
'''
Trading simulations
'''

#%%
### Use old 6mo model data
col3 = ['Ticker', 'Date', 'High', 'Low', 'Close', 'date_10per', 'price_10per', 'date_6mo', 'price_6mo', 'consensus', 'pred']
bank_size_init = 10000

data_new = pd.read_csv('data_with_preds1.csv')

#data1.to_csv('data1_updt.csv', index = False)

valid_tickers = []
datas = []
with os.scandir(path2) as it:
    for entry in it:
        if entry.name.endswith(".csv"):
            #get the valid ticker list
            name = entry.name.split('.')[0]
            valid_tickers.append(name)
            temp_data = pd.read_csv(entry)
            temp_data['Ticker'] = [name] * len(temp_data)
            datas.append(temp_data)
data_old = pd.concat(datas)
del datas


data_new = data_new[data_new['Date'] >= '2021-01-01']
data_old = data_old[data_old['Date'] >= '2021-01-01']


data1 = data1[data1['Date'] >= '2021-01-01']

data1 = data_old[col3]
#sets sells for lows, highs and close
data1 = update_selldata(data = data1, period = 127, 
                value = 0.1, 
                period_date_name = 'date_6mo', 
                period_price_name = 'price_6mo', 
                up_date_name = 'date_10per',
                up_price_name = 'price_10per')


data1 = datetime_index(data1)
data1_all = data1.copy()

data1_buy = data1_all[data1_all['consensus'] > 0.00001]
#data1_buy = data1_all[data1_all['pred'] == 1]
data1_buy = data1_buy[data1_buy['price_6mo'] != -1]
data1_buy = data1_buy[((data1_buy['Close'] > 2) & (data1_buy['Close'] < 100))]

#data1_buy = data1_buy[data1_buy['price_10per'] != -1]

trading_dates = list(set(data1_buy['Date']))
trading_dates.sort()
max_date_ind = trading_dates.index('2022-10-07')

all_trading_dates = list(set(data1_all['Date']))
all_trading_dates.sort()
all_trading_dates = [datetime.strptime(dt, '%Y-%m-%d') for dt in all_trading_dates]

data1_buy.columns


banks_old, final_ports, end_dates, start_dates = trade_simulation(bank_size = 10000, 
                                        number = 15, trading_dates = trading_dates, 
                                        buy_data = data1_buy, end_data = data1_all, 
                                        end_time = 127, max_date_ind = max_date_ind,
                                        up_sell_col = 'price_10per', up_time_col = 'date_10per',
                                        end_sell_col = 'price_6mo', end_time_col = 'date_6mo',
                                        iterations = 750)


banks = banks_old
pd.DataFrame(banks).hist(bins = 50)
plt.show()
banks = [x for x in banks if x < 22000]
returns4 = [((x - bank_size_init)/bank_size_init) for x in banks]
neg4 = [x for x in banks if x < bank_size_init]
print('Mean: ' + str(round(np.mean(returns4) * 100, 3)) + '%')
print('Median: ' + str(round(np.median(returns4) * 100, 3)) + '%')
print('Neg Chance: ' + str(round((len(neg4)/len(returns4))* 100, 3)) + '%')
pd.DataFrame(returns4).hist(bins = 50)
plt.show()

z = pd.DataFrame([start_dates, banks]).transpose()
z.columns = ['dt', 'rt']
zz = z.groupby([pd.to_datetime(z['dt']).dt.year, pd.to_datetime(z['dt']).dt.month]).rt.count()

zx = zz.index
zxx = []
for i in range(len(zx)):
    q = str(zx[i][0]) + '/' + str(zx[i][1])
    
    zxx.append(q)

plt.plot(zz.values)
plt.xticks(zxx)

heatmap(np.where(data1_buy['price_10per'] == -1, 0, 1), np.where(data1_buy['consensus'] > 0.5, 1, 0), 'q')




df = final_ports[0]

df.columns

sum(df.share_num * df.end_price)


#get the data
all_4mo = pd.read_csv('securities_12072023_4mo.csv')
all_4mo = all_4mo[all_4mo['Date'] > '2021-01-01']
buy_4mo = pd.read_csv('sim_4mo.csv')
all_6mo = pd.read_csv('securities_12072023_6mo.csv')
all_6mo = all_6mo[all_6mo['Date'] > '2021-01-01']
buy_6mo = pd.read_csv('sim_6mo.csv')


data1 = 0
data1.columns
data1 = datetimeticker_index(data1)

data1 = data1[data1['Date'] >= '2021-01-01']
data1_buy = data1[data1['pred'] == 1]
data1_all = data1[col3]
data1_buy = data1[col3]

#add datetime column as index for faster querying
all_4mo = datetimeticker_index(all_4mo)
buy_4mo = datetimeticker_index(buy_4mo)
all_6mo = datetimeticker_index(all_6mo)
buy_6mo = datetimeticker_index(buy_6mo)


#get valid trading dates
trading_dates = list(set(list(set(all_4mo['Date'])) + list(set(all_6mo['Date']))))
trading_dates.sort()
max_date_ind = trading_dates.index('2022-11-30') 

#subset data to just needed columns
col1 = ['Date', 'High', 'Low', 'Close', 'date_10per', 'price_10per', 'date_6mo', 'price_6mo']
col2 = ['Date', 'High', 'Low', 'Close', 'date_10per', 'price_10per', 'date_4mo', 'price_4mo']
col3 = ['Date', 'High', 'Low', 'Close', 'date_10per', 'price_10per', 'date_6mo', 'price_6mo', 'consensus', 'pred']
col4 = ['Date', 'High', 'Low', 'Close', 'date_10per', 'price_10per', 'date_4mo', 'price_4mo', 'consensus', 'pred']
all_4mo = all_4mo[col2]
buy_4mo = buy_4mo[col4]
all_6mo = all_6mo[col1]
buy_6mo = buy_6mo[col3]

###run simulations ###

#prepare date range and buy df
buy_4mo_pred = buy_4mo[buy_4mo['pred'] == 1]
buy_4mo_pred = buy_4mo_pred[buy_4mo_pred['price_4mo'] != -1]

trading_dates = list(set(buy_4mo_pred['Date']))
trading_dates.sort()
max_date_ind = trading_dates.index('2022-08-10')


all_trading_dates = list(set(all_4mo['Date']))
all_trading_dates.sort()
all_trading_dates = [datetime.strptime(dt, '%Y-%m-%d') for dt in all_trading_dates]

bank_size_init = 10000

banks4, final_ports4 = trade_simulation(bank_size = bank_size_init, number = 15, 
                                      trading_dates = trading_dates, 
                                      buy_data = buy_4mo_pred, end_data = all_4mo, 
                                      end_time = 365, max_date_ind = max_date_ind,
                                      up_sell_col = 'price_10per', up_time_col = 'date_10per',
                                      end_sell_col = 'price_4mo', end_time_col = 'date_4mo',
                                      iterations = 1250)



banks6, final_ports6 = trade_simulation(bank_size = bank_size_init, number = 15, 
                                      trading_dates = trading_dates, 
                                      buy_data = buy_6mo, end_data = all_6mo, 
                                      end_time = 365, max_date_ind = max_date_ind,
                                      up_sell_col = 'price_10per', up_time_col = 'date_10per',
                                      end_sell_col = 'price_6mo', end_time_col = 'date_6mo',
                                      iterations = 10)

### Assess ###
pd.DataFrame(banks4).hist(bins = 50)
plt.show()
banks4_r = [x for x in banks4 if x < 30000]
returns4 = [((x - bank_size_init)/bank_size_init) for x in banks4_r]
neg4 = [x for x in banks4_r if x < bank_size_init]
print('Mean: ' + str(round(np.mean(returns4) * 100, 3)) + '%')
print('Median: ' + str(round(np.median(returns4) * 100, 3)) + '%')
print('Neg Chance: ' + str(round((len(neg4)/len(returns4))* 100, 3)) + '%')
pd.DataFrame(returns4).hist(bins = 50)
plt.show()


pd.DataFrame(banks_orig).hist(bins = 50)
plt.show()
banks_orig_r = [x for x in banks_orig if x < 30000]
returns_orig = [((x - bank_size_init)/bank_size_init) for x in banks_orig_r]
neg_orig = [x for x in banks_orig_r if x < bank_size_init]
print('Mean: ' + str(round(np.mean(returns_orig) * 100, 3)) + '%')
print('Median: ' + str(round(np.median(returns_orig) * 100, 3)) + '%')
print('Neg Chance: ' + str(round((len(neg_orig)/len(returns_orig))* 100, 3)) + '%')
pd.DataFrame(returns_orig).hist(bins = 50)
plt.show()





#old code to be deleted?

#get predictions
data.dropna(inplace = True)
x_data, tickers, dates = prep_4pred(data, price_norm_colz, x_columns, scaler)
preds = predict_purchases(x_data, models, conf = 0.75)

preds['Date'] = dates.values
preds['Ticker'] = tickers.values
data = pd.merge(data, preds, how = 'left', on = ['Date', 'Ticker'])
data.reset_index(inplace = True, drop = True)
#data.dropna(inplace = True)



tickers = list(set(data['Ticker']))
datas = apply_sell_over(data, tickers, 0.1)

os.chdir(path1)
datas.to_csv('data_with_preds1.csv')

data = pd.read_csv('data_with_preds1.csv')

#data = data[((data['Close'] > 2) * (data['Close'] < 100))]

#get the valid trading period (6mo before end at least)
trading_dates = list(set(list(data['Date'])))
trading_dates.sort()
max_date_ind = trading_dates.index('2022-06-01') 

#add datetime column for faster quering 
data['Datetime'] = pd.to_datetime(data['Date'])
data_full = data.set_index(['Datetime', 'Ticker'])

#subset data to just needed columns
sim_cols = ['Ticker', 'Date', 'High', 'Low', 'Close','pred', 'consensus', 'date_10per', 'price_10per', 'date_6mo',
'price_6mo', 'Datetime']
dat1 = data[((data['Close'] < 100) & (data['Close'] > 2))]
dat1 = dat1[sim_cols]
dat1 = dat1.set_index('Datetime')
dat1 = dat1[dat1['date_6mo'] != -1]
buy_data = dat1

del data
del dat1


drop_cols = ['Volume', 'bb_center21',
       'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84', 'bb_low84',
       'bb_center252', 'bb_upp252', 'bb_low252', 'minmax21', 'minmax84',
       'minmax252', 'atr_21', 'atr_84', 'atr_252', 'rsi_21', 'rsi_84',
       'rsi_252', 'vol_21', 'vol_84', 'vol_252', 'gain_5','lose_5', 'lose_10', 'lose_15', 'end_p5', 'end_p10',
       'end_p15', 'end_n5', 'end_n10', 'end_n15']



sim4mo = pd.read_csv('simdat_4mo.csv')

sim4mo = simdat_4mo.copy()
sim4mo = sim4mo.drop(columns = drop_cols)
buy4mo = sim4mo[sim4mo['pred'] == 1] #buy data for standard 4mo model
buy4mo6 = sim4mo[sim4mo['pred_y'] == 1] #buy data for 6mo model on 4mo data

sim6mo = pd.read_csv('simdat_6mo.csv')
sim6mo = sim6mo.drop(columns = drop_cols)
buy6mo = sim6mo[sim6mo['pred'] == 1] #buy data for standard 6mo model


sim4mo['pred_x'].sum()


z = sim4mo.sample(10000)

buy_data['pred'] = np.where(buy_data['consensus'] > 0.5, 1,0)
buy_data = buy_data[buy_data['pred'] == 1]




ypred, ytrue = index_resample_tab(datax = sim4mo['pred'].values.reshape(-1,1), 
                                  datay = sim4mo['gain_10'].values.reshape(-1,1))

confusion_matrix(ytrue, ypred)
accuracy_score(ytrue, ypred)
precision_score(ytrue, ypred)



banks2, final_ports = trade_simulation(bank_size = 10000, number = 15, 
                                      trading_dates = trading_dates, 
                                      buy_data = buy_data, end_data = data_full, 
                                      end_time = 126, max_date_ind = max_date_ind,
                                      up_sell_col = 'price_10per', up_time_col = 'date_10per',
                                      end_sell_col = 'price_6mo', end_time_col = 'date_6mo',
                                      iterations = 300)

banks = banks + banks2


banks = [x for x in banks if x < 22500]
np.mean(banks)

pd.DataFrame(banks).hist(bins = 50)

up_sell_col = 'price_10per'
up_time_col = 'date_10per'
end_sell_col = 'price_6mo'
end_time_col = 'date_6mo'

banks = []
ends = []
for i in tqdm(range(250)):

    bank = 10000
    number = 15
    
    #set up the trader with an active df and non-increase in value df
    act_df, neg_df, bank, end_date = init_trader(bank, number, trading_dates = trading_dates,
                                 buy_data = buy_data, max_date_ind = max_date_ind,
                                 up_sell_col = 'price_10per', end_time = 126)
    #trade the stocks which never go up to the end
    act_df, hold_df, bank = trade_neg_df(neg_df, act_df, bank, end_date, buy_data, up_sell_col = 'price_10per', 
                     end_time_col = 'date_6mo', end_sell_col = 'price_6mo')
    #set the conditions for the active trades
    act_df = apply_master_cond(act_df, end_date, up_sell_col, up_time_col, end_sell_col, end_time_col)
    new_hold = act_df[act_df['cond'] == 0]
    hold_df = pd.concat([hold_df, new_hold])
    act_df = act_df[act_df['cond'] != 0]
    #trade the active dataframe to the end #NEED TO EDIT LAMBDA GLOBAL
    hold_df, bank = active_trader(act_df, hold_df, bank, buy_data, end_date = end_date,
                                  trading_dates = trading_dates, max_date_ind = max_date_ind)
    final_bank, end_portfolio = final_sale(bank, hold_df, end_data = data_full, end_date = end_date)
    banks.append(final_bank)
    ends.append(end_portfolio)

############


#%% SAND BOXING

#of those that go up by 10%, whats the average time it takes to increase in value
data1 = pd.read_csv('data_with_preds1.csv')

data = data[data['Ticker'].isna() == False]

data['Open'] = data['Open'].astype(float)
data['High'] = data['High'].astype(float)
data['Low'] = data['Low'].astype(float)

tiks = list(set(list(data['Ticker'])))

data['Close'].iloc[10] * (1 + value)


z = data.sample(1000)

data = update_selldata(data, period = 85, value = 0.1, period_date_name = 'date_4mo', 
                       period_price_name = 'price_4mo', up_date_name = 'date_10per', 
                       up_price_name = 'price_10per')

data.to_csv('securities_12072023_4mo.csv', index = False)


data = data[data['pred'] == 1]
dat = data[data['gain_10'] == 1]
dat = data[data['date_10per'] != -1]
dat = dat[dat['gain_10'] == 1]



time_to_increase = pd.to_datetime(dat['Date']) - pd.to_datetime(dat['date_10per'])

days_increase = time_to_increase.dt.days

days_increase = days_increase[days_increase <= -1]

days_increase.hist(bins = 100, density = True)

days_increase.median() #51 day average, 34 day median, for those that are predicted its mean of 23 days and median of 9 days

#days increase overtime 
days_increase = time_to_increase.dt.days

dat['days_increase'] = days_increase
dat['day'] = pd.to_datetime(dat['Date']).dt.day
dat['month'] = pd.to_datetime(dat['Date']).dt.month
dat['year'] = pd.to_datetime(dat['Date']).dt.year

dat2 = dat[dat['days_increase'] <= -1]


times = dat2.groupby(['year', 'month'])['days_increase'].mean()

plt.plot(times.values) #not at all stable, but is CLOSE to stationary




### Of those that increase by 10% in 6mo, what percentage also decrease by 5,10,15%
data = pd.read_csv('data_with_preds1.csv')
data = data2.copy()

data_up = data[data['gain_10'] == 1]
data_down = data[data['lose_5'] == 1]
data_updown = data_up[data_up['lose_5'] == 1]
data_updown5 = data_up[((data_up['lose_5'] == 1) & (data_up['lose_10'] == 0))]
data_updown10 = data_up[((data_up['lose_10'] == 1) & (data_up['lose_15'] == 0))]
data_updown15 = data_up[(data_up['lose_15'] == 1)]

print('go down ' + str(np.round(len(data_updown)/len(data), 4)))
print('go up, then down ' + str(np.round(len(data_updown)/len(data_up), 4)))
print('go up, then down5 ' + str(np.round(len(data_updown5)/len(data_up), 4)))
print('go up, then down10 ' + str(np.round(len(data_updown10)/len(data_up), 4)))
print('go up, then down15+ ' + str(np.round(len(data_updown15)/len(data_up), 4)))


len(data_updown5)/len(data_up)


### Of those that drop by 10% what is the average time to recovery (back to 0) within 6mo?





z = data_up.sample(1000)


tiks = list(set(data['Ticker']))
period = 127
datas = []
for i in tqdm(range(len(tiks))):
    dat = data[data['Ticker'] == tiks[i]]
    
    dat['end_n5'] = (dat['Close'].shift(-period) < dat['Close'] * 0.95).astype(int)
    dat['end_n10'] = (dat['Close'].shift(-period) < dat['Close'] * 0.9).astype(int)
    dat['end_n15'] = (dat['Close'].shift(-period) < dat['Close'] * 0.85).astype(int)
    
    dat['lose_5'] = cross_thresh(dat, period, 0.95, up = False)
    dat['lose_10'] = cross_thresh(dat, period, 0.9, up = False)
    dat['lose_15'] = cross_thresh(dat, period, 0.85, up = False)
    datas.append(dat)
data2 = pd.concat(datas)

data2.to_csv('data_with_preds1.csv', index = False)
