# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 20:05:45 2023

@author: q23853
"""

import requests
import joblib
from io import BytesIO
import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
import tempfile
import os
import random
from datetime import date, datetime, timedelta
from tqdm import tqdm
import logging



temp_dir = tempfile.mkdtemp() #make a temp directory to house the .h5 file
os.chdir(temp_dir)

#supress unecessary tensorflow warnings    
logging.getLogger('tensorflow').setLevel(30)

#set the urls
lrc_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/lrc.joblib'
rf_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/rf.joblib'
xgb_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/xgbc.joblib'
mars_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/mars.joblib'
scaler_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/tabular_scaler.joblib'
dense_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/main/production_models/dense.h5'
tickers_path = 'https://raw.githubusercontent.com/seansteel3/ML_trader/v0.0.1/valid_tickers.csv'

paths = [lrc_path, rf_path, xgb_path, mars_path, scaler_path, dense_path, tickers_path]

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
    return lrc, rfc, xgbc, mars, tab_scaler, dense, valid_tickers

#define the technical indeicator functions

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
lam_vol = lambda x : (vol(x))

def prepare_MLdata(ticker, start, end):
    #get the data, if it exists
    data = yf.download(ticker, start= start, end= end,  progress=False, auto_adjust= False)
    if len(data) > 127:
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
        
        data['vol_21'] = data['Close'].rolling(21).apply(lam_vol)
        data['vol_84'] = data['Close'].rolling(84).apply(lam_vol)
        data['vol_252'] = data['Close'].rolling(252).apply(lam_vol)
        return data
    else:
        return np.nan


def sample_tickers(N_tickers, today_date, valid_tickers, tab_scaler):
    print("Collecting Historical Data... ")
    ticker_inds = random.sample(range(len(valid_tickers)), N_tickers)
    tickers = [valid_tickers[ind] for ind in ticker_inds]
    #download their data from yfinance (last 60 days for wide MOE [I'm not calculating when/where calendar shifts may be])
    today = today_date + timedelta(days = -1)
    start = today + timedelta(days = -600)
    start = datetime.strftime(start, '%Y-%m-%d')
    end = datetime.strftime(today, '%Y-%m-%d')
    datas = []
    saved_tickers = []
    for i in tqdm(range(len(tickers))):
        ticker = tickers[i]
        data = prepare_MLdata(ticker, start, end)
        #check the last day is within constraints
        try: 
            constrains = ((data['Close'].iloc[-1] > 2) & (data['Close'].iloc[-1] < 75) & (len(data) > 252))
            if constrains:    
                datas.append(pd.DataFrame(data.iloc[-1]).transpose()) #save just the last day
                saved_tickers.append(ticker)
        except:
            continue
    #convert the data to arrays, normalize, and standardize
    colz_norm_name = ['bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84', 
                      'bb_low84', 'bb_center252',  'bb_upp252', 'bb_low252', 'atr_21',
                      'atr_84', 'atr_252'] 
    colz_ml =  [ 'bb_center21', 'bb_upp21','bb_low21','bb_center84','bb_upp84',
                'bb_low84','bb_center252','bb_upp252','bb_low252','minmax21',
                'minmax84','minmax252','atr_21','atr_84','atr_252','rsi_21','rsi_84',
                 'rsi_252','vol_21','vol_84','vol_252']
    datas = pd.concat(datas)
    #normalize to price
    for i in range(len(colz_norm_name)):
        col = colz_norm_name[i]
        datas[col] = datas[col]/datas['Close']
    datas = np.array(datas[colz_ml])
    datas = tab_scaler.transform(datas)
    return datas, saved_tickers

def predict_purchases(datas, saved_tickers, conf_thresh, models):
    #unpack models
    lrc, rfc, xgbc, mars, tab_scaler, dense, valid_tickers = models
    #make predictions
    print('\nMaking predictions... ')
    #make predictions
    rfc_prd = rfc.predict_proba(datas)[:,1]
    xgbc_prd = xgbc.predict_proba(datas)[:,1]
    mars_prd = mars.predict(datas)
    lrc_prd = lrc.predict_proba(datas)[:,1]
    dense_prd = dense.predict(datas)
    #set a prediction frame for ensemble strategy
    pred_frame = pd.DataFrame([rfc_prd, xgbc_prd, mars_prd, lrc_prd, dense_prd]).transpose()
    pred_frame.columns = ['rf', 'xgb', 'mars', 'lrc', 'dense']
    pred_frame['dense'] = pred_frame['dense'].apply(lambda x: x[0])
    #make final prediction
    all_80 = np.where((pred_frame[['rf', 'xgb', 'mars', 'lrc', 'dense']]>= conf_thresh).all(axis=1), 1, 0)
    consensus = pred_frame.mean(axis = 1)
    #return a final output of a top suggestion, and the remaining options
    out_df = pd.DataFrame([saved_tickers, consensus, all_80]).transpose()
    out_df.columns = ['Ticker', 'Consensus', 'Prediction']
    out_df = out_df.sort_values('Consensus', ascending = False)
    out_df.reset_index(inplace = True, drop = True)
    return out_df 

def run_ML_trader(paths, N_tickers, conf_thresh, counter = 0, return_df = True):
    #unpack paths
    lrc_path, rf_path, xgb_path, mars_path, scaler_path, dense_path, tickers_path = paths
    #get models
    models = get_models(lrc_path, rf_path, xgb_path, mars_path, scaler_path, 
                        dense_path, tickers_path)
    #get data
    today_date = date.today() + timedelta(days = -5)
    datas, saved_tickers = sample_tickers(N_tickers, today_date, valid_tickers = models[-1], tab_scaler = models[4])
    #make predictions
    out_df = predict_purchases(datas, saved_tickers, conf_thresh, models)
    if (out_df['Prediction'].sum() > 0):
        print("Models predict: [" + str(out_df['Ticker'].iloc[0]) + "] is a buy with " + 
              str(np.round(out_df['Consensus'].iloc[0] * 100, 1)) + '% confidence!')
        return out_df
    elif  ((out_df['Prediction'].sum() == 0) & (counter == 0)):
        print('No stocks met criteria, expanding search and trying again...')
        run_ML_trader(paths, N_tickers * 3, conf_thresh, counter = 1)
    else:
        print("No stocks had high enough confidence! \nRerun, lower the confidence, or wait for \nanother time with better market conditions.")
        if return_df == True:
            return out_df
    

out_df = run_ML_trader(paths = paths, N_tickers = 25, conf_thresh = 0.8, counter = 0)


