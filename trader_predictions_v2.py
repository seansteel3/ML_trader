#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:01:57 2024

@author: seansteele
"""



import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA,FastICA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from statsmodels.regression.rolling import RollingOLS
import requests
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import ParameterGrid
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from datetime import datetime, timedelta
from random import sample 
import joblib
from io import BytesIO
from tensorflow import keras
import tempfile

pd.options.mode.chained_assignment = None
fmp_key = 'nJVBKg9okmibd0ojIvPJQo2CCzYn9piM'

#%%FUNCTIONS
def download_models(url):
    temp_dir = tempfile.mkdtemp() #make a temp directory to house the .h5 file
    os.chdir(temp_dir)
    response = requests.get(url)
    files = response.json()
    models = []
    for model_info in files:
        name = model_info['name']
        mod_path = model_info['download_url']
        mod_content = requests.get(mod_path).content
        if name.split('.', 1)[1] == 'h5':
            open(name, 'wb').write(mod_content)
            model = keras.models.load_model(name, compile = False)
            models.append(model)
        elif name.split('.', 1)[0][:2] == 'qt':
            transformer = joblib.load(BytesIO(mod_content))
        else:
            model = joblib.load(BytesIO(mod_content))
            models.append(model)
    return models, transformer


def normalize_cols(data, norm_cols):
    for i in range(len(norm_cols)):
        col = norm_cols[i]
        data[col] = data[col]/data['close']
    return data

def get_stock_picks(gain_url, neg_url, fmp_url, fmp_key, sample_size, final_cols, keeps, 
                    norm_cols, gain_feats, neg_feats, attempts = 0):
    
    #Step0: Download models and transformers
    print('Downloading Models...')
    gain_models, gain_transformer = download_models(gain_url)
    neg_models, neg_transformer = download_models(neg_url)
    #Step1: Get all tradable tickers and sample down
    all_tickers = get_tickers(url = fmp_url, key = fmp_key, excludes = ['/', ' ', '^', '_', '.'])
    tickers = sample(all_tickers, sample_size)
    start = (datetime.today() - timedelta(450)).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')
    #Step3: get data
    print('Downloading Data...')
    data = get_all_data(tickers, start, end, fmp_key, final_cols)[0]
    data = normalize_cols(data, norm_cols)
    data = data[keeps]
    data.dropna(inplace = True)
    data.reset_index(inplace = True, drop = True)
    #Step4: segment out data and apply standardizers as required for each model
    gain_data = data[gain_feats]
    gainx = gain_transformer.transform(gain_data)
    neg_data = data[neg30_feats]
    negx = neg_transformer.transform(neg_data)
    #Step5: Pass data into  models
    print('Predicting Stock Picks...')
    gain_pred = predict_purchases(gainx, conf_thresh = 0.5, models = gain_models)
    gain_pred.columns = ['gain_consensus', 'gain_prediction']
    neg_pred = predict_purchases(negx, conf_thresh = 0.5, models = neg_models)
    neg_pred.columns = ['neg_consensus', 'neg_prediction']
    #Step6: Merge results and clean
    pred_df = pd.DataFrame(np.concatenate([np.array(gain_pred), np.array(neg_pred)], axis = 1))
    pred_df.columns = ['gain_consensus', 'gain_prediction', 'neg_consensus', 'neg_prediction']
    pred_df['ticker'] = data['ticker']
    pred_df['date'] = data['date']
    #step7: Return results
    dates = list(set(pred_df['date']))
    dates.sort()
    buy_df = pred_df[((pred_df['gain_prediction'] == 1) & (pred_df['neg_prediction'] == 0))]
    buy_df = buy_df[buy_df['date'] == dates[-1]]
    if (len(buy_df) < 1) & (attempts == 0):
        print('None in sample predicted to buy, retrying with larger sample...')
        buy_df, pred_df = get_stock_picks(gain_url, neg_url, fmp_url, fmp_key, 
                                          sample_size * 2, final_cols, keeps, 
                                          norm_cols, gain_feats, neg_feats,
                                          attempts = 1)
    elif (len(buy_df) < 1) & (attempts != 0):
        print('None in sample predicted to buy. Retry with larger sample or try another time. :(')
    else:
        print("Complete!")
    return buy_df, pred_df

def heatmap(ys, preds, dataset_type):
    acc = accuracy_score(ys, preds)
    prec = precision_score(ys, preds)
    conf = confusion_matrix(ys, preds)
    ax= plt.subplot()
    sns.heatmap(conf, annot=True, fmt='g', ax=ax); 
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(f'{dataset_type} Accuracy: ' + str(np.round(acc * 100, 2)) + '%' 
                 + f'\n {dataset_type} Precision: '+ str(np.round(prec * 100, 2)) + '%'); 
    ax.xaxis.set_ticklabels(['Pred: 0', 'Pred: 1']); ax.yaxis.set_ticklabels(['Pred: 0', 'Pred: 1']);
    plt.show()

def predict_purchases(data, conf_thresh, models):
    #make predictions
    print('\nMaking predictions... ')
    #make predictions
    preds = []
    for i in range(len(models)):
        try:
            pred = models[i].predict_proba(data)[:,1]
            preds.append(list(pred))
        except:
            pred = list(models[i].predict(data))
            pred = [prd[0] for prd in pred]
            preds.append(pred)
    #set a prediction frame for ensemble strategy
    pred_frame = pd.DataFrame(preds).T
    #pred_frame['dense'] = pred_frame['dense'].apply(lambda x: x[0])
    #make final prediction
    all_tresh = np.where((pred_frame >= conf_thresh).all(axis=1), 1, 0)
    consensus = pred_frame.mean(axis = 1)
    #return a final output of a top suggestion, and the remaining options
    out_df = pd.DataFrame([consensus, all_tresh]).transpose()
    out_df.columns = ['Consensus', 'Prediction']
    out_df.reset_index(inplace = True, drop = True)
    return out_df 


def get_financial_data(data, ticker, final_cols):
    metric_url = 'https://financialmodelingprep.com/api/v3/key-metrics/'
    full_metric = metric_url + ticker + '?period=quarter&apikey=' + fmp_key
    ratios_url = 'https://financialmodelingprep.com/api/v3/ratios/'
    full_ratios = ratios_url + ticker + '?period=quarter&apikey=' + fmp_key
    met = requests.get(full_metric).json()
    met = pd.DataFrame.from_dict(met)
    rat = requests.get(full_ratios).json()
    rat = pd.DataFrame.from_dict(rat)

    rat_cols = list(rat.columns.difference(met.columns)) + ['symbol', 'date']
    fund_dat = pd.merge(met, rat[rat_cols], how = 'left', on = ['symbol', 'date'])
    financial_cols = list(set(list(fund_dat.columns)) - set(['symbol', 'date', 'calendarYear', 'period']))
    return fund_dat, financial_cols

def macd(data, period):
    me1 = data.close.ewm(span = int(period/2)).mean()
    me2 = data.close.ewm(span = int(period)).mean()
    return me1 - me2

def aroon(data, period):
    au = data.high.rolling(period).apply(lambda x: x.argmax()) / (period)
    ad = data.low.rolling(period).apply(lambda x: x.argmin()) / (period)
    return au - ad

def adx(data):
    dmh = data.high - data.high.shift(1)
    dml = data.high - data.high.shift(1)
    dp = (data.bb_center21 + dmh)/data.atr_21
    dn = (data.bb_center21 - dml)/data.atr_21
    dx = (abs(dp - dn)/abs(dp + dn))
    return (dx.shift(1) + dx)/21

def stoc_osc(data, period):
    lper = data.low.rolling(window = period).min()
    hper = data.high.rolling(window = period).max()
    osc = (data.close - lper)/(hper - lper) 
    return osc

def bbands(data, period, c_name = 'bb_center', u_name = 'bb_upp', l_name = 'bb_low'):
    ma = data.close.ewm(span = period).mean()
    std = data.close.ewm(span = period).std()
    data[c_name] = ma
    data[u_name] = ma + (2 * std)
    data[l_name] = ma - (2* std)
    return data

def minmax_ratio(data, period):
    return ((data['close'] - data['low'].rolling(period).min())/(data['high'].rolling(period).max() - data['low'].rolling(period).min()))

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

def cross_thresh(data, period, thresh, up):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=period)
    if up == True:
        datac = data.copy()
        datac['look_ahead'] = datac['high'].shift(-1).rolling(window = indexer, min_periods = 1).max()
        return_col  = (datac['look_ahead'] > datac['close'] * thresh).astype(int)
    else:
        datac = data.copy()
        datac['look_ahead'] = datac['low'].shift(-1).rolling(window = indexer, min_periods = 1).min()
        return_col  = (datac['look_ahead'] < datac['close'] * thresh).astype(int)
    return return_col


#total ML trader data preparation function
def prepare_MLdata(data, period):
    if len(data) > period + 1:
        #apply BBands, MinMax, ATR, RSI, and VOL for 21, 84, and 252 trading days
        data = bbands(data, 21, c_name = 'bb_center21', u_name = 'bb_upp21', l_name = 'bb_low21')
        data = bbands(data, 84, c_name = 'bb_center84', u_name = 'bb_upp84', l_name = 'bb_low84')
        data = bbands(data, 252, c_name = 'bb_center252', u_name = 'bb_upp252', l_name = 'bb_low252')
        
        data['minmax21'] = minmax_ratio(data, 21)
        data['minmax84'] = minmax_ratio(data, 84)
        data['minmax252'] = minmax_ratio(data, 252)
        
        data['atr_21'] = atr(data['high'], data['low'], data['close'], 21)
        data['atr_84'] = atr(data['high'], data['low'], data['close'], 84)
        data['atr_252'] = atr(data['high'], data['low'], data['close'], 252)
        
        data['rsi_21'] = rsi(data['close'], 21)
        data['rsi_84'] = rsi(data['close'], 84)
        data['rsi_252'] = rsi(data['close'], 252)
        
        data['vol_21'] = data['close'].rolling(21).apply(lambda x : (vol(x)))
        data['vol_84'] = data['close'].rolling(84).apply(lambda x : (vol(x)))
        data['vol_252'] = data['close'].rolling(252).apply(lambda x : (vol(x)))
        
        data['stoc_osc21'] = stoc_osc(data, 21)
        data['stoc_osc84'] = stoc_osc(data, 84)
        data['stoc_osc252'] = stoc_osc(data, 252)
        
        data['aroon_21'] = aroon(data, 21)
        data['aroon_84'] = aroon(data, 84)
        data['aroon_252'] = aroon(data, 252)
        
        data['macd_21'] = macd(data, 21)
        data['macd_84'] = macd(data, 84)
        data['macd_252'] = macd(data, 252)

        data['adx'] = adx(data) #only 1 period for now, doesn't seem like long periods helpful anyway
            
        #apply crossing 5,10,15% thresholds
        data['gain_5'] = cross_thresh(data, period, 1.05, up = True)
        data['gain_10'] = cross_thresh(data, period, 1.1, up = True)
        data['gain_15'] = cross_thresh(data, period, 1.15, up = True)
        
        data['lose_5'] = cross_thresh(data, period, 0.95, up = False)
        data['lose_10'] = cross_thresh(data, period, 0.9, up = False)
        data['lose_15'] = cross_thresh(data, period, 0.85, up = False)
        #apply end 5,10,15% thresholds
        data['end_p5'] = (data['close'].shift(-period) > data['close'] * 1.05).astype(int)
        data['end_p10'] = (data['close'].shift(-period) > data['close'] * 1.1).astype(int)
        data['end_p30'] = (data['close'].shift(-period) > data['close'] * 1.30).astype(int)
        
        data['end_n10'] = (data['close'].shift(-period) < data['close'] * 0.9).astype(int)
        data['end_n30'] = (data['close'].shift(-period) < data['close'] * 0.7).astype(int)
        data['end_n50'] = (data['close'].shift(-period) < data['close'] * 0.5).astype(int)
        
        #get the price and date of 6mo ahead
        data = price_date(data, period, date_name = 'date_6mo', price_name = 'price_6mo')
        #get the price and date of crossing threshold of 10%
        data = cross_value_price_date2(data)
        return data
    else:
        return np.nan

#additional back test function: find the price in 6mo and date
def price_date(data, period, date_name = 'date_6mo', price_name = 'price_6mo'):
    """
    BACKTESTING FUNCTION BUT ADDS MINIMUAL OVERHEAD GIVEN THE API CALL TIME AND RATE LIMIT;
    FOR SIMPLICITY ITS KEPT
    """
    end_dates = []
    end_prices = []
    try:
        #find the price and date N trading days ahead if there is enough data
        for i in range(len(data) - int(period)):
            end_dates.append(data['date'].iloc[i + int(period)])
            end_prices.append(data['close'].iloc[i + int(period)])
        #create a mask for the end of the dataframe where no data is
        masks = [-1] * int(period)
        end_dates = end_dates + masks
        end_prices = end_prices + masks
        data[date_name] = end_dates
        data[price_name] = end_prices
    except:
        #print("exception in date!")
        masks = [-1] * len(data)
        data[date_name] = masks
        data[price_name] = masks
    return data
    
#additional back test function: find the price at 10% crossed within 6mo and date
def cross_value_price_date2(data, period = 127, value = 0.1, date_name = 'date_10per', price_name = 'price_10per'):
    """
    BACKTESTING FUNCTION BUT ADDS MINIMUAL OVERHEAD GIVEN THE API CALL TIME AND RATE LIMIT;
    FOR SIMPLICITY ITS KEPT
    """
    end_dates = []
    end_prices_high = []
    end_prices_low = []
    end_prices_close = []
    end_prices_thresh = []
    try:
        for i in range(len(data) - int(period)):
            threshold = data['close'].iloc[i] * (1 + value)
            #subset the data to next 126 trading days
            sub_data = data.iloc[i: i + int(period)]
            sub_data.reset_index(inplace = True, drop = True)
            #add the day and price where its over 10% within next 126 trading days if it exists
            if sub_data['high'].gt(threshold).sum() > 0:
                cross_index = sub_data['high'].gt(threshold).idxmax()
                end_dates.append(sub_data['date'].iloc[cross_index])
                end_prices_high.append(sub_data['high'].iloc[cross_index])
                end_prices_low.append(sub_data['low'].iloc[cross_index])
                end_prices_close.append(sub_data['close'].iloc[cross_index])
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


#get available tickers
def get_tickers(url, key, excludes):
    tickers = requests.get(f"{url}apikey={key}").json()
    tickers = pd.DataFrame(tickers)
    #get them into both a list and a moreinfo dataframe
    try:
        tickers =  tickers[tickers['exchangeShortName'].isin(['NASDAQ', 'ASX', 'AMEX', 'ETF', 'CBOE'])]
        tickers = tickers[tickers['type'] != 'trust'] #no need to figure out how taxes and trading works right now
    except:
        tickers =  tickers[tickers['exchange'].isin(['NASDAQ', 'ASX', 'AMEX', 'ETF', 'CBOE'])]
    all_tickers = list(tickers['symbol'])
    #clear out weird symbols (no need to figure these out right now)
    for i in range(len(excludes)):
        all_tickers = [tick for tick in all_tickers if excludes[i] not in tick]
    return all_tickers

def get_all_data(tickers, start, end, fmp_key, final_cols):
    datas = []
    failed_fins = []
    for i in tqdm(range(len(tickers))):
        try:
            tik = tickers[i]
            historical_url = 'https://financialmodelingprep.com/api/v3/historical-price-full/'
            url = historical_url + tik + '?from=' + start + '&to=' + end + '&apikey=' + fmp_key
            data = requests.get(url).json()
            data = pd.DataFrame.from_dict(data['historical'])
            data = data.sort_values(by = ['date'])
            data = data.reset_index(drop = True)
            if len(data) <= 252: #check minimum length required
                continue
            else: 
                act_days = (pd.to_datetime(data['date'].iloc[0]) - pd.to_datetime(data['date'].iloc[-1]))/np.timedelta64(1, 'D')
                act_years = act_days/365
                act_trade = len(data)/252
                calendar_diff = act_years - act_trade
            if calendar_diff > 1: #check continous data
                continue
            else: 
                try:
                    #get financial data if its available 
                    fund_dat, financial_cols = get_financial_data(data, tik, final_cols)
                    fund_dat = fund_dat.sort_values(by = ['date'])
                    fund_dat['datetime'] = pd.to_datetime(fund_dat['date'])
                    #get technical data
                    data['ticker'] = [tik] * len(data)
                    data = prepare_MLdata(data, period = 127)
                    data['datetime'] = pd.to_datetime(data['date'])
                    #merge and save
                    data = pd.merge_asof(data, fund_dat, on='datetime', direction='backward')
                    data['date'] = data['date_x']
                    data = data[final_cols]
                    datas.append(data)
                except:
                    failed_fins.append(tik)
                    continue 
        except:
            print('catastrophic failure!') #only occurs on delisted tickers or when API rate hit
            continue
    datas = pd.concat(datas)
    datas = datas.reset_index(drop = True)
    return datas, failed_fins


#%% ADDITIONALS: Phase III will need these to be dynamically fetched, hardcoded fine for now


norm_cols = ['bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84', 
                  'bb_low84', 'bb_center252',  'bb_upp252', 'bb_low252', 'atr_21',
                  'atr_84', 'atr_252', 'macd_21', 'macd_84', 'macd_252']

neg30_feats = ['atr_252','atr_84','atr_21','returnOnTangibleAssets', 'bb_upp252',
                'returnOnAssets','operatingProfitMargin','vol_252','netProfitMargin',
                'priceEarningsRatio','netIncomePerShare','bb_upp84','ebitPerRevenue','pretaxProfitMargin',
                'bb_low252','priceCashFlowRatio','operatingCashFlowPerShare','enterpriseValueMultiple',
                'operatingCashFlowSalesRatio','freeCashFlowPerShare','earningsYield',
                'pfcfRatio','vol_84','vol_21','freeCashFlowYield','macd_252',
                'bb_upp21','returnOnCapitalEmployed','roic']

gain_feats = ['atr_21','vol_252','atr_84','vol_84','atr_252','vol_21','bb_upp252','bb_upp21',
              'bb_low252','bb_low84','bb_upp84','vwap','bb_low21','bb_center252','adx', 
              'bb_center84','dividendYield','payoutRatio','researchAndDdevelopementToRevenue',
              'grahamNetNet','marketCap','bb_center21','bookValuePerShare','rsi_252']

ml_cols = list(set(neg30_feats + gain_feats))

keeps = ['date', 'ticker', 'high', 'low', 'open', 'close', 'volume', 'gain_5', 
         'gain_10', 'gain_15', 'lose_5', 'lose_10', 'lose_15', 'end_p5', 'end_p10', 
         'end_p30', 'end_n10', 'end_n30', 'end_n50', 'date_6mo', 'price_6mo', 
         'date_10per', 'price_10per_high', 'price_10per_low', 
         'price_10per_close', 'price_10per_thresh'] + ml_cols

final_cols = ['date', 'ticker',
 'open',
 'high',
 'low',
 'close',
 'adjClose',
 'volume',
 'changePercent',
 'vwap',
 'bb_center21',
 'bb_upp21',
 'bb_low21',
 'bb_center84',
 'bb_upp84',
 'bb_low84',
 'bb_center252',
 'bb_upp252',
 'bb_low252',
 'minmax21',
 'minmax84',
 'minmax252',
 'atr_21',
 'atr_84',
 'atr_252',
 'rsi_21',
 'rsi_84',
 'rsi_252',
 'vol_21',
 'vol_84',
 'vol_252',
 'stoc_osc21',
 'stoc_osc84',
 'stoc_osc252',
 'aroon_21',
 'aroon_84',
 'aroon_252',
 'macd_21',
 'macd_84',
 'macd_252',
 'adx',
 'revenuePerShare',
 'netIncomePerShare',
 'operatingCashFlowPerShare',
 'freeCashFlowPerShare',
 'cashPerShare',
 'bookValuePerShare',
 'tangibleBookValuePerShare',
 'shareholdersEquityPerShare',
 'interestDebtPerShare',
 'marketCap',
 'enterpriseValue',
 'peRatio',
 'priceToSalesRatio',
 'pocfratio',
 'pfcfRatio',
 'pbRatio',
 'ptbRatio',
 'evToSales',
 'enterpriseValueOverEBITDA',
 'evToOperatingCashFlow',
 'evToFreeCashFlow',
 'earningsYield',
 'freeCashFlowYield',
 'debtToEquity',
 'debtToAssets',
 'netDebtToEBITDA',
 'currentRatio',
 'interestCoverage',
 'incomeQuality',
 'dividendYield',
 'payoutRatio',
 'salesGeneralAndAdministrativeToRevenue',
 'researchAndDdevelopementToRevenue',
 'intangiblesToTotalAssets',
 'capexToOperatingCashFlow',
 'capexToRevenue',
 'capexToDepreciation',
 'stockBasedCompensationToRevenue',
 'grahamNumber',
 'roic',
 'returnOnTangibleAssets',
 'grahamNetNet',
 'workingCapital',
 'tangibleAssetValue',
 'netCurrentAssetValue',
 'investedCapital',
 'averageReceivables',
 'averagePayables',
 'averageInventory',
 'daysSalesOutstanding',
 'daysPayablesOutstanding',
 'daysOfInventoryOnHand',
 'receivablesTurnover',
 'payablesTurnover',
 'inventoryTurnover',
 'roe',
 'capexPerShare',
 'assetTurnover',
 'capitalExpenditureCoverageRatio',
 'cashConversionCycle',
 'cashFlowCoverageRatios',
 'cashFlowToDebtRatio',
 'cashRatio',
 'companyEquityMultiplier',
 'daysOfInventoryOutstanding',
 'daysOfPayablesOutstanding',
 'daysOfSalesOutstanding',
 'debtEquityRatio',
 'debtRatio',
 'dividendPaidAndCapexCoverageRatio',
 'dividendPayoutRatio',
 'ebitPerRevenue',
 'ebtPerEbit',
 'effectiveTaxRate',
 'enterpriseValueMultiple',
 'fixedAssetTurnover',
 'freeCashFlowOperatingCashFlowRatio',
 'grossProfitMargin',
 'longTermDebtToCapitalization',
 'netIncomePerEBT',
 'netProfitMargin',
 'operatingCashFlowSalesRatio',
 'operatingCycle',
 'operatingProfitMargin',
 'pretaxProfitMargin',
 'priceBookValueRatio',
 'priceCashFlowRatio',
 'priceEarningsRatio',
 'priceEarningsToGrowthRatio',
 'priceFairValue',
 'priceSalesRatio',
 'priceToBookRatio',
 'priceToFreeCashFlowsRatio',
 'priceToOperatingCashFlowsRatio',
 'quickRatio',
 'returnOnAssets',
 'returnOnCapitalEmployed',
 'returnOnEquity',
 'shortTermCoverageRatios',
 'totalDebtToCapitalization', 'gain_5',
 'gain_10',
 'gain_15',
 'lose_5',
 'lose_10',
 'lose_15',
 'end_p5',
 'end_p10',
 'end_p30',
 'end_n10',
 'end_n30',
 'end_n50',
 'date_6mo',
 'price_6mo',
 'date_10per',
 'price_10per_high',
 'price_10per_low',
 'price_10per_close',
 'price_10per_thresh'
 ]

#%% Data pulls

gain_url = "https://api.github.com/repos/seansteel3/ML_trader/contents/production_models/modelsg10/"
neg30_url = "https://api.github.com/repos/seansteel3/ML_trader/contents/production_models/modelsn30/"
fmp_url = 'https://financialmodelingprep.com/api/v3/available-traded/list?'

buy_df, pred_df = get_stock_picks(gain_url = gain_url, neg_url = neg30_url, 
                                  fmp_url = fmp_url, fmp_key = fmp_key, 
                                  sample_size = 100, final_cols = final_cols, 
                                  keeps = keeps, norm_cols = norm_cols, 
                                  gain_feats = gain_feats, neg_feats = neg30_feats)

