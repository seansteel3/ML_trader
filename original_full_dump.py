# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:58:06 2023

1) EXPLORE MARS MODEL

2) EXPLORE MODEL SEPARATE TRAINING AND COMBINING
    -Not very effective compared to a scratch dual model Out-of-Box

3) EXPLORE INTENSE REGULARIZTION AND LONG TRAINING FOR DUAL RNN
    -Might be the "right way" but not worth the headache!

4) STRATGEY CONSIDERATIONS: Maximize accuracy of Predicted 1's
    One of these two should be the final Descion model
        -All agree at > 70%, 75% or 80%
    Stacked aren't terribly good
    
5) VALIDATION ON COVID TIMEFRAMES
    -RUN BEST MODEL IN A CROSS VAL OVER THIS
    -Call this proxy done: None of the models outpreformed the others
    in a timeframe setting by all that much (compared to average preformace)

6) FINAL EXPLORATION: HOW LONG BACK SHOULD TRAINING DATA LOOK?
    -SAME TEST RANGE, KEEP EXTENDING BACK TRAINING
    -REPEAT AT LEAST 2X MORE ON DIFFERENT TEST RANGES
    
    -Seems like more data is marginally better, but it varies a bit.
    Conclusion: TRAIN IN QAURTERS OF TIME:
        -20% q1
        -20% q2
        -25% q3
        -35% q4

1. DONE!
2. DONE!
3. DONE!
4. DONE!
5. DONE!
6. DONE!


FUTURE: EXPLORE MULTI-OBJECTIVE
    -DUAL RNN WITH 2 CLASSES
    -2 RF
    -USING THOSE OUTPUTS IN A STRATGEGY 
        -ONLY BUY IF BOTH ALIGN
        -USE 2 SINLGE CLASS MODELS
            -BOTH RF
            -BOTH DUAL RNN
            -ENSEMBLE OF THEM
            -BUY ONLY IF BOTH ALIGN

@author: q23853
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta
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


path1 = 'C:\\Users\\q23853\\Desktop\\random_trader'
path2 = 'C:\\Users\\q23853\\Desktop\\random_trader\\processed_stocks2'
os.chdir(path2)
#%%
#get all exchange tickers
amex_tickers = pd.read_csv('amex.csv', sep =',')
amex_tickers = list(amex_tickers['Symbol'])
nyse_tickers = pd.read_csv('nyse.csv', sep =',')
nyse_tickers = list(nyse_tickers['Symbol'])
nasdaq_tickers = pd.read_csv('nasdaq.csv', sep =',')
nasdaq_tickers = list(nasdaq_tickers['Symbol'])

#clean up non-alpha characters
all_tickers = amex_tickers + nyse_tickers + nasdaq_tickers
all_tickers = [tick for tick in all_tickers if str(tick) != 'nan']
strings = set([string for ticker in all_tickers for string in ticker])
all_tickers = [tick for tick in all_tickers if '/' not in tick]
all_tickers = [tick for tick in all_tickers if ' ' not in tick]
all_tickers = [tick for tick in all_tickers if '^' not in tick]



def lookback_window(final_date, lookback):     
    end = datetime.strptime(final_date, '%Y-%m-%d')
    start = end - timedelta(days = lookback)
    start = start.strftime('%Y-%m-%d')
    end = end.strftime('%Y-%m-%d')
    return start, end

def lookforward_window(start_date, lookforward):     
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = start + timedelta(days = lookforward)
    start = start.strftime('%Y-%m-%d')
    end = end.strftime('%Y-%m-%d')
    return start, end
    
stock = all_tickers[23 + 100]
current_date = '2023-05-30'
lookback = 120

start, end = lookback_window(current_date, lookback)
 
data = yf.download(stock, start= start, end= end,  progress=False)
 
 
#define and calulate volatility over a timeperiod
def vol(data):
    return np.sqrt((sum((data - data.mean())**2))/len(data))/data.mean()

lam_vol = lambda x : (vol(x))
zz = data['High'].rolling(15).apply(lam_vol)
plt.plot(zz)

vol(data['Close'])

pdiff = data['High'] - data['Low']
vol(pdiff)


#%%
'''
While a correlation exists in the past between historical volality and an increase 
in value by 10% or more in the next 6mo(P= 0.000), this correlation does not hold 
within the last 6 months as a test set (50% accuracy of Logistic Regression)
'''
#%%
def vol_thresh_cross(ticker, buy_date, lookback, lookforward):
    start_sellby, end_sellby = lookforward_window(buy_date, lookforward)
    start_look, end_look = lookback_window(buy_date, lookback)
    #get data
    hist_data = yf.download(ticker, start = start_look, end = end_look,  progress=False)
    trade_data = yf.download(ticker, start = start_sellby, end = end_sellby,  progress=False)
    #see if it crosses threshold at any timeframe
    thresh = trade_data['High'].iloc[0] * 1.1
    if any(trade_data['High'] > thresh):
        label = 1
    else:
        label = 0
    #calculate the pdiff vol and overall hist vol
    all_vol = vol(hist_data['Close'])
    pdiff = hist_data['High'] - hist_data['Low']
    pdiff_vol = vol(pdiff)
    return label, all_vol, pdiff_vol

#create a list of dates that can be sampled for training/exploring
valid_end = '2023-02-05'
valid_end_dt = datetime.strptime(valid_end, "%Y-%m-%d")
valid_start = '2023-01-01'
valid_start_dt = datetime.strptime(valid_start, "%Y-%m-%d")

diff = valid_end_dt - valid_start_dt

valid_date_list = [valid_end_dt - timedelta(days=x) for x in range(2922)]
valid_date_list = [date.strftime('%Y-%m-%d') for date in valid_date_list]

#randomly sample 500 stocks and dates getting vol_thresh_cross info
labels = []
all_vols = []
pdiff_vols = []
num_points = 500
for i in tqdm(range(num_points)):
    try:
        ticker_num = randrange(len(all_tickers))
        trade_date_num =  randrange(len(valid_date_list))
        ticker = all_tickers[ticker_num]
        buy_date = valid_date_list[trade_date_num]
        label, all_vol, pdiff_vol = vol_thresh_cross(ticker, buy_date, 
                                                     lookback, lookforward)
        labels.append(label)
        all_vols.append(all_vol)
        pdiff_vols.append(pdiff_vol)
    except:
        continue


vol_data = pd.DataFrame([labels, all_vols, pdiff_vols]).transpose()
vol_data.columns = ['labels', 'all_vols', 'pdiff_vols']
vol_data['labels'].mean()

#under sample data
# class count
count_class_0, count_class_1 = vol_data.labels.value_counts()
# separate according to `label`
df_class_0 = vol_data[vol_data['labels'] == 0]
df_class_1 = vol_data[vol_data['labels'] == 1]
# sample only from class 0 quantity of rows of class 1
df_class_1_under = df_class_1.sample(count_class_1)
vol_resam = pd.concat([df_class_0, df_class_1_under], axis=0)
vol_resam = vol_resam.dropna()

log_reg = smf.logit("labels ~ all_vols + pdiff_vols", data = vol_resam).fit()

log_reg.summary()

np.exp(log_reg.params[1])

threshy = 0.5
preds = np.where(log_reg.predict(vol_resam) > threshy, 1, 0)

confusion_matrix(vol_resam['labels'], preds)
accuracy_score(vol_resam['labels'], preds)
#%%
'''
Prepare a few common techinical indicators
'''
#%%

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
    return rsi

def vol(data):
    return np.sqrt((sum((data - data.mean())**2))/len(data))/data.mean()
lam_vol = lambda x : (vol(x))

def cross_thresh(data, index_num, thresh_amt, period = 126):
    '''
    Does the data ever cross some threshold in the next 126 days
    '''
    current_index = data.iloc[index_num].name
    end_index = data.index.get_loc(current_index) + period
    forward_data = data.loc[current_index:data.index[end_index]]
    thresh = data['Close'].loc[current_index] * thresh_amt
    if any(forward_data['High'] > thresh):
        label = 1
    else:
        label = 0
    return label


def df_upcross_thresh(data, thresh_amt, period = 126):
    '''
    run cross_tresh for every eligible day in the dataframe 
    '''
    stop = len(data) - (period + 1)
    new_col = []
    for i in range(stop):
        new_col.append(cross_thresh(data, i, thresh_amt, period))
    #fill rest of the list
    new_col = new_col + [np.nan] * (len(data) - len(new_col))
    return new_col
        
def close_down(row, thresh_amt, period = 126):
    try:
        current_index = row.name
        end_index = data.index.get_loc(current_index) + period
        forward_data = data.loc[current_index:data.index[end_index]]
        thresh = data['Close'].loc[current_index] * thresh_amt
        if forward_data['Close'].iloc[-1] < thresh:
            return 1
        else:
            return 0
    except IndexError:
            return np.nan


def prepare_data(ticker, start, end):
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
        data['ris_252'] = rsi(data['Close'], 252)
        
        data['vol_21'] = data['Close'].rolling(21).apply(lam_vol)
        data['vol_84'] = data['Close'].rolling(84).apply(lam_vol)
        data['vol_252'] = data['Close'].rolling(252).apply(lam_vol)
        #look 126 trading days to the future and see if any highs or closes exceed 5%/10%/15%
        data['gain_5'] = df_upcross_thresh(data, 1.05)
        data['gain_10'] = df_upcross_thresh(data, 1.1)
        data['gain_15'] = df_upcross_thresh(data, 1.15)
        
        # data['lose_5'] = data.apply(lambda row: close_down(row, 0.95), axis = 1)
        # data['lose_10'] = data.apply(lambda row: close_down(row, 0.9), axis = 1)
        # data['lose_15'] = data.apply(lambda row: close_down(row, 0.85), axis = 1)
        return data
    else:
        return np.nan

start = '2014-01-01'
end = '2023-08-01'

datas = []
names = []

start_ind = int(len(all_tickers)/10)*2
end_ind = int(len(all_tickers)/10)*4
for i in tqdm(range(start_ind, end_ind)):
    ticker = all_tickers[i]
    data = prepare_data(ticker, start, end)
    if isinstance(data, pd.DataFrame):
        data['lose_5'] = data.apply(lambda row: close_down(row, 0.95), axis = 1)
        data['lose_10'] = data.apply(lambda row: close_down(row, 0.9), axis = 1)
        data['lose_15'] = data.apply(lambda row: close_down(row, 0.85), axis = 1)
        datas.append(data)
        names.append(ticker)
    else:
        continue
    

for i in range(len(names)):
    name = names[i]
    file_name = name + '.csv'
    df = datas[i]
    df.to_csv(file_name, sep = ',', index = False, encoding = 'utf-8')
    
###################

#get data
dat = prepare_data('DAL', start, end)

#segment into sections
yr_y = dat.Close[-252:].values
yr_x = np.array(list(range(252)))
mo3_y = dat.Close[-84:].values
mo3_x = np.array(list(range(84)))
mo1_y = dat.Close[-21:].values
mo1_x = np.array(list(range(21)))

#get coef's
np.polyfit(yr_x, yr_y,1)[0]
np.polyfit(mo3_x, mo3_y,1)[0]
np.polyfit(mo1_x, mo1_y,1)[0]

plt.plot(dat.Close[-252:])

def linear_regress(data, period):
    dat_y = data[-period:].values
    dat_x = np.array(list(range(period)))
    return np.polyfit(dat_x, dat_y, 1)[0]

dat['beta_21'] = dat.Close.rolling(window = 21).apply(lambda x: linear_regress(x, 21))

### Write the trend data to the source ML ready data ###
os.chdir(path2)
#read in names of each stock
with os.scandir(path2) as it:
    for entry in it:
        if entry.name.endswith(".csv"):
            #get the valid ticker list
            name = entry.name.split('.')[0]
            save_name = str(name + '.csv')
            temp_data = pd.read_csv(entry)
            #add the regression trend
            temp_data['beta_21'] = temp_data.Close.rolling(window = 21).apply(lambda x: linear_regress(x, 21))
            temp_data['beta_84'] = temp_data.Close.rolling(window = 84).apply(lambda x: linear_regress(x, 84))
            temp_data['beta_252'] = temp_data.Close.rolling(window = 252).apply(lambda x: linear_regress(x, 252))
            #write back to file
            temp_data.to_csv(save_name, index = False)
            
            
temp_data = pd.read_csv('A.csv')
#add the regression trend
temp_data['beta_21'] = temp_data.Close.rolling(window = 21).apply(lambda x: linear_regress(x, 21))
temp_data['beta_84'] = temp_data.Close.rolling(window = 84).apply(lambda x: linear_regress(x, 84))
temp_data['beta_252'] = temp_data.Close.rolling(window = 252).apply(lambda x: linear_regress(x, 252))
#write back to file
temp_data.to_csv('A.csv', index = False)


#%%
'''
Randomly select 5% of the days in each stock into 1 dataframe
Randomly select 40% train, 20% validation, 10% test
Rebalance each by sampling minority class from the remaining 30%
Remainder of the remaining 30% becomes its own imbalanced test set
'''
#%%
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

def df_rebalance(df_2balance, df_2donate, drop_rate, col_2_bal, drop_thresh = 0.5005):
    drops = list(range(drop_rate))
    while df_2balance[col_2_bal].mean() > drop_thresh and len(df_2donate) > drop_rate:
        new_rows = pd.DataFrame(df_2donate.iloc[:drop_rate])
        df_2balance = pd.concat([df_2balance, new_rows])
        df_2donate.reset_index(inplace = True, drop = True)
        df_2donate = df_2donate.drop(drops)
    return df_2balance, df_2donate


def get_ml_data(valid_tickers, min_dt, max_dt, colz_norm, colz_outliar,
                data_prop = 0.02, min_pr = 2, max_pr = 75, rebal_col = 'gain_10',
                drop_rate = 50):
    #get the data
    datas = []
    for i in tqdm(range(len(valid_tickers))):
        data = pd.read_csv(valid_tickers[i] + '.csv')
        data = data[((data['Date'] >= min_dt) & ((data['Date'] <= max_dt)))]
        data.dropna(inplace = True)
        per_df = int(len(data) * data_prop) 
        data = data.sample(per_df)
        datas.append(data)
    main_df = pd.concat(datas)
    #shuffle data
    main_df = main_df.sample(len(main_df))
    #drop data where close >$2 and <$85
    main_df = main_df[(main_df['Close'] > min_pr) & (main_df['Close'] < max_pr)]
    #normalize BB, MA and ATR data to closeing price
    # colz_norm = list(range(7,16)) + list(range(19,22)) + list(range(34, 37))
    for i in range(len(colz_norm)):
        num = colz_norm[i]
        main_df.iloc[:, num] = main_df.iloc[:, num]/main_df['Close']

    #clear out extreme outliars (7.5 instead of 1.5 as the usual thresh)
    #Only adjust the meterics (ignore close, volume, etc)
    # num_outliar = list(range(7,28)) + list(range(34,37))
    #remove outliars
    main_nout = remove_outlier(main_df, thresh = 7.5, cols = colz_outliar)
    main_nout = main_nout.sample(len(main_nout))
    #split data into Live/Extra based on level of imbalance
    extra_thresh = int((main_nout[rebal_col].mean() + 0.01) * len(main_nout))
    extra_df = main_nout.iloc[:extra_thresh]
    live_df = main_nout.iloc[extra_thresh:]

    #prepare rebalancing datasets
    extra_0_df = extra_df[extra_df[rebal_col] == 0]
    extra_0_df.reset_index(inplace = True, drop = True)
    extra_1_df = extra_df[extra_df[rebal_col] == 1]
    extra_1_df.reset_index(inplace = True, drop = True)

    #rebalance each minority class gain_10 from the extra dataframe
    if live_df[rebal_col].mean() > 0.51:
        live_df, extra_0_df = df_rebalance(live_df, extra_0_df, drop_rate, rebal_col)
    else:
        live_df, extra_1_df = df_rebalance(live_df, extra_1_df, drop_rate, rebal_col)
    #test_df, extra_0_df = df_rebalance(test_df, extra_0_df, drop_rate, 'gain_10')
    print('Data Balance: ' + str(live_df[rebal_col].mean()))
    return live_df

#get all valid tickers
os.chdir(path2)

#read in names of each stock
valid_tickers = []
# updated_data = []
with os.scandir(path2) as it:
    for entry in it:
        if entry.name.endswith(".csv"):
            #get the valid ticker list
            name = entry.name.split('.')[0]
            valid_tickers.append(name)
            

#load 1 dataset to see columns as reminder
data1 = pd.read_csv('AAPL.csv')

#set the columns for normalization
data_columns = data1.columns
colz_norm = list(range(7,16)) + list(range(19,22)) + list(range(34, 37))


#set columns for outliar removal
num_outliar = list(range(7,28)) + list(range(34,37))
colz_outliar = []
for i in range(len(num_outliar)):
    num = num_outliar[i]
    colz_outliar.append(data_columns[num])

#get the data
train_df = get_ml_data(valid_tickers, '2013-01-01', '2017-01-01', colz_norm, colz_outliar,
                    data_prop = 0.05, min_pr = 2, max_pr = 75, rebal_col = 'gain_10',
                    drop_rate = 50)

train_df = train_df.sample(len(train_df))
val_split = int(len(train_df) * 0.65)
val_df = train_df.iloc[val_split:]
train_df = train_df.iloc[:val_split]

test_df = get_ml_data(valid_tickers, '2019-01-01', '2020-01-01', colz_norm, colz_outliar,
                    data_prop = 0.01, min_pr = 2, max_pr = 75, rebal_col = 'gain_10',
                    drop_rate = 50)



#%%
'''
Prepare RF classifer for gain_10

66.5% accuracy? This feels too high, but no clear lookahead bias I don't think?
Will need to backtest in simulation to confirm

drop_5 classifier feels more real, but still unknown
58% accuracy.


'''
#%%

num_train = list(range(6,28))
#num_train = list(range(6,19)) + list(range(22,25))+ list(range(34,37))
colz_train = []
for i in range(len(num_train)):
    num = num_train[i]
    colz_train.append(data_columns[num])


#split data and labels off
x_train = train_df[colz_train]
x_val = val_df[colz_train]
x_test = test_df[colz_train]

y_train = train_df['gain_10']
y_val = val_df['gain_10']
y_test = test_df['gain_10']

#run classifier
rfc = RandomForestClassifier(max_depth = 7)
rfc.fit(x_train, y_train)


train_pred = rfc.predict(x_train)
val_pred = rfc.predict(x_val)
test_pred = rfc.predict(x_test)

#evaluate
train_acc = accuracy_score(y_train, train_pred)
train_conf = confusion_matrix(y_train, train_pred)

val_acc = accuracy_score(y_val, val_pred)
val_conf = confusion_matrix(y_val, val_pred)

test_acc = accuracy_score(y_test, test_pred)
test_conf = confusion_matrix(y_test, test_pred)

#use heatmaps to display values of interest
ax= plt.subplot()
sns.heatmap(train_conf, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Training Set Accuracy: ' + str(np.round(train_acc * 100, 2)) + '%'); 
ax.xaxis.set_ticklabels(['Do Not Buy', 'Buy']); ax.yaxis.set_ticklabels(['Do Not Buy', 'Buy']);
plt.show()

ax= plt.subplot()
sns.heatmap(val_conf, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Validation Set Accuracy: ' + str(np.round(val_acc * 100, 2)) + '%'); 
ax.xaxis.set_ticklabels(['Do Not Buy', 'Buy']); ax.yaxis.set_ticklabels(['Do Not Buy', 'Buy']);
plt.show()

ax= plt.subplot()
sns.heatmap(test_conf, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Test Set Accuracy: ' + str(np.round(test_acc * 100, 2)) + '%'); 
ax.xaxis.set_ticklabels(['Do Not Buy', 'Buy']); ax.yaxis.set_ticklabels(['Do Not Buy', 'Buy']);
plt.show()

# rfc.feature_importances_

plt.bar(colz_train, rfc.feature_importances_)
plt.xticks(rotation = 90)
plt.title('Random Forest Feature Importances')
plt.show()

#save model
os.chdir('C:\\Users\\q23853\\Desktop\\random_trader')
joblib.dump(rfc, 'inital_rt_rf.joblib')

#load model
rfc = joblib.load('inital_rt_rf.joblib')


#%%
'''
Prepare data for a RNN
'''

#%%
#get all valid tickers
os.chdir(path2)
#read in names of each stock
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
datas = pd.concat(datas)
#remove useless/redundant columns and na
datas.drop(['end_date', 'end_price', 'ma_21', 'ma_84', 'ma_252'], axis = 1, inplace = True)
datas.dropna(inplace = True)
#normalize relevant columns to close price
colz_datas_norm = list(range(7,16)) + list(range(19,22))
for i in range(len(colz_datas_norm)):
    num = colz_datas_norm[i]
    datas.iloc[:, num] = datas.iloc[:, num]/datas['Close']
#convert RSI to 0-1 scale
datas['rsi_21'] = datas['rsi_21']/100
datas['rsi_84'] = datas['rsi_84']/100
datas['rsi_252'] = datas['ris_252']/100
datas.drop(['ris_252'], inplace = True, axis = 1) #remove the typo
datas.reset_index(inplace = True, drop = True)
datas_tr = datas[datas['Date'] < '2016-01-01']
datas_val = datas[(datas['Date'] > '2016-06-01') & (datas['Date'] < '2018-01-01')]
datas_te = datas[(datas['Date'] > '2018-01-01') & (datas['Date'] < '2020-01-01')]

### sample the data ###
    
def get_rnn_data(datas, valid_tickers, x_columns1, y_columns, num_trd_days, 
                 num_dates, prop_tickers):
    #pick random set of tickers
    chsn_tkrs = random.sample(valid_tickers, int(prop_tickers * len(valid_tickers)))
    #for each ticker...
    x_datas = []
    y_datas = []
    dt_ranges_all = []
    for i in tqdm(range(len(chsn_tkrs))):
        #subset the data to the ticker
        tik = chsn_tkrs[i]
        temp_data = datas[datas['Ticker'] == tik]
        #if the data is large enough and the last day is < $75 and > $2
        try:
            price_check = ((temp_data['Close'].iloc[-1] < 75) & (temp_data['Close'].iloc[-1] > 2))
        except:
            price_check = False
        if ((len(temp_data) > num_trd_days + num_dates + 1) & (price_check)):
            #grab M random dates at least N trading days after the start of the ticker sample
            vld_dts = list(temp_data['Date'])
            end_dts = vld_dts[num_trd_days:]
            chsn_dts = random.sample(end_dts, num_dates)
            #get the date ranges by index
            dt_ranges = []
            for j in range(len(chsn_dts)):
                dt_end_ind = vld_dts.index(chsn_dts[j])
                dt_start_ind = dt_end_ind - num_trd_days
                dt_ranges.append((vld_dts[dt_start_ind], vld_dts[dt_end_ind]))
                dt_ranges_all.append((vld_dts[dt_start_ind], vld_dts[dt_end_ind]))
            #for each date range...
            for k in range(len(dt_ranges)):
                #grab data between each of the date ranges
                temp_data2 = temp_data[((temp_data['Date'] > dt_ranges[k][0]) & 
                                        (temp_data['Date'] <= dt_ranges[k][1]))]
                if len(temp_data2) ==  num_trd_days:#as long as enough data exists...
                    #save the relevant data to a numpy array and save the array to a list
                    x_datas.append(np.array(temp_data2[x_columns]))
                    y_datas.append(np.array(temp_data2[y_columns].iloc[-1]))                        
    #remove non-consecutive trading day stocks
    x_datas, y_datas = clean_date_ranges(x_datas, y_datas, dt_ranges_all, allowance = 1.5)
    return x_datas, y_datas

def get_rnn_data2(datas, valid_tickers, x_columns1, y_columns, num_trd_days, 
                 num_dates, prop_tickers, x_columns2 = None):
    #pick random set of tickers
    chsn_tkrs = random.sample(valid_tickers, int(prop_tickers * len(valid_tickers)))
    #for each ticker...
    x_datas = []
    x_datas2 = []
    y_datas = []
    dt_ranges_all = []
    for i in tqdm(range(len(chsn_tkrs))):
        #subset the data to the ticker
        tik = chsn_tkrs[i]
        temp_data = datas[datas['Ticker'] == tik]
        #if the data is large enough and the last day is < $75 and > $2
        try:
            price_check = ((temp_data['Close'].iloc[-1] < 75) & (temp_data['Close'].iloc[-1] > 2))
        except:
            price_check = False
        if ((len(temp_data) > num_trd_days + num_dates + 1) & (price_check)):
            #grab M random dates at least N trading days after the start of the ticker sample
            vld_dts = list(temp_data['Date'])
            end_dts = vld_dts[num_trd_days:]
            chsn_dts = random.sample(end_dts, num_dates)
            #get the date ranges by index
            dt_ranges = []
            for j in range(len(chsn_dts)):
                dt_end_ind = vld_dts.index(chsn_dts[j])
                dt_start_ind = dt_end_ind - num_trd_days
                dt_ranges.append((vld_dts[dt_start_ind], vld_dts[dt_end_ind]))
                dt_ranges_all.append((vld_dts[dt_start_ind], vld_dts[dt_end_ind]))
            #for each date range...
            for k in range(len(dt_ranges)):
                #grab data between each of the date ranges
                temp_data2 = temp_data[((temp_data['Date'] > dt_ranges[k][0]) & 
                                        (temp_data['Date'] <= dt_ranges[k][1]))]
                if len(temp_data2) ==  num_trd_days:#as long as enough data exists...
                    #save the relevant data to a numpy array and save the array to a list
                    x_datas.append(np.array(temp_data2[x_columns]))
                    y_datas.append(np.array(temp_data2[y_columns].iloc[-1]))
                    if x_columns2 != None:
                        x_datas2.append(temp_data2[x_columns2].iloc[-1])
    #remove non-consecutive trading day stocks
    x_datas, y_datas, x_datas2 = clean_date_ranges(x_datas, y_datas, x_datas2, dt_ranges_all, allowance = 1.5)
    return x_datas, y_datas, x_datas2




        
def clean_date_ranges(x_datas, y_datas, x_datas2, dt_ranges_all, allowance = 1.5):
    '''
    Remove tickers that have massive date dicrepancies from start to end
    ie: proxy for non-consecutive trading days
    Reused tickers? Paused then resumed trading? Something else?
    '''
    x_datas_real = []
    y_datas_real = []
    x_datas_real2 = []
    allow_diff = int(num_trd_days * allowance)
    for i in range(len(dt_ranges_all)):
        dt1 = datetime.strptime(dt_ranges_all[i][0],'%Y-%m-%d')
        dt2 = datetime.strptime(dt_ranges_all[i][1],'%Y-%m-%d')
        diff = dt2 - dt1
        if diff <= timedelta(days=allow_diff):
            x_datas_real.append(x_datas[i])
            y_datas_real.append(y_datas[i])
            x_datas_real2.append(x_datas2[i])
    return x_datas_real, y_datas_real, x_datas_real2

def remove_arr_outliar(x_datas, y_datas, limit = 7):
    #get the means of each feature, and save values 7std away from the mean
    caps_upper = []
    caps_lower = []
    for i in range(x_datas.shape[-1]):
        mn = np.mean(x_datas[:,:,i])
        std = np.std(x_datas[:,:,i])
        caps_upper.append(mn + (limit * std))
        caps_lower.append(mn - (limit * std))
    #remove arrays that have values outside of 7 STD of the mean
    for i in tqdm(range(x_datas.shape[-1])):
        x_datas_clean = []
        y_datas_clean = []
        for j in range(x_datas.shape[0]):
            check = any(x_datas[j,:,i] > caps_upper[i]) or any(x_datas[j,:,i] < caps_lower[i])
            if check == False:
                x_datas_clean.append(x_datas[j,:,:])
                y_datas_clean.append(y_datas[j])
        x_datas = np.array(x_datas_clean)
        y_datas = np.array(y_datas_clean)
    return x_datas, y_datas

def remove_arr_outliar2(x_datas, y_datas, x_datas2, limit = 7):
    #get the means of each feature, and save values 7std away from the mean
    caps_upper = []
    caps_lower = []
    for i in range(x_datas.shape[-1]):
        mn = np.mean(x_datas[:,:,i])
        std = np.std(x_datas[:,:,i])
        caps_upper.append(mn + (limit * std))
        caps_lower.append(mn - (limit * std))
    #remove arrays that have values outside of 7 STD of the mean
    for i in tqdm(range(x_datas.shape[-1])):
        x_datas_clean = []
        y_datas_clean = []
        x_datas_clean2 = []
        for j in range(x_datas.shape[0]):
            check = any(x_datas[j,:,i] > caps_upper[i]) or any(x_datas[j,:,i] < caps_lower[i])
            if check == False:
                x_datas_clean.append(x_datas[j,:,:])
                y_datas_clean.append(y_datas[j])
                x_datas_clean2.append(x_datas2[j])
        x_datas = np.array(x_datas_clean)
        y_datas = np.array(y_datas_clean)
        x_datas2 = np.array(x_datas_clean2)
    return x_datas, y_datas, x_datas2

def time_sampler(data, timestep_size):
    final_timesteps = ((data.shape[1] - 1) // timestep_size) + 1
    empty_data = np.zeros((data.shape[0], final_timesteps, data.shape[2]))
    for i in range(final_timesteps):
        empty_data[:, i, :] = data[:, -1 - i * 5, :]
    selected_data = empty_data[:,::-1,:]
    return selected_data

def savgol_smooth(data, window, poly):
    smoothed_data = np.empty_like(data)
    #apply savgol filter
    for sample in tqdm(range(data.shape[0])):
        for feature in range(data.shape[-1]):
            data_2_smooth = data[sample,:,feature]
            smoothed = savgol_filter(data_2_smooth, window, poly)
            smoothed_data[sample, :,feature] = smoothed
    return smoothed_data

def savgol_smooth_silent(data, window, poly):
    smoothed_data = np.empty_like(data)
    #apply savgol filter
    for sample in range(data.shape[0]):
        for feature in range(data.shape[-1]):
            data_2_smooth = data[sample,:,feature]
            smoothed = savgol_filter(data_2_smooth, window, poly)
            smoothed_data[sample, :,feature] = smoothed
    return smoothed_data


prop_tickers = 0.15
num_dates_tr = 70
num_dates_te = 25
num_trd_days = 126
x_columns2 = [ 'bb_center21', #'Open','High','Low','Close','Adj Close',
                    'bb_upp21','bb_low21','bb_center84','bb_upp84','bb_low84',
                    'bb_center252','bb_upp252','bb_low252','minmax21','minmax84',
                    'minmax252','atr_21','atr_84','atr_252','rsi_21','rsi_84',
                    'rsi_252','vol_21','vol_84','vol_252']
x_columns = ['Volume','Close']

y_columns = ['gain_10']

b_data = get_rnn_data2(datas_tr, valid_tickers, x_columns, y_columns, 
                                num_trd_days, num_dates_tr, prop_tickers, x_columns2)

x_datas, y_datas, x_datas2 = get_rnn_data2(datas_tr, valid_tickers, x_columns, y_columns, 
                                num_trd_days, num_dates_tr, prop_tickers, x_columns2)
x_datas_val, y_datas_val, x_datas_val2 = get_rnn_data2(datas_val, valid_tickers, x_columns, y_columns, 
                                num_trd_days, num_dates_te, prop_tickers, x_columns2)
x_datas_te, y_datas_te, x_datas_te2 = get_rnn_data2(datas_te, valid_tickers, x_columns, y_columns, 
                                num_trd_days, num_dates_te, prop_tickers, x_columns2)
#convert to arrays
x_datas = np.array(x_datas)
x_datas2 = np.array(x_datas2).reshape(len(x_datas2), len(x_columns2))
x_datas, y_train, x_datas2 = remove_arr_outliar2(x_datas, y_datas, x_datas2, limit = 7)

x_datas_val = np.array(x_datas_val)
x_datas_val2 = np.array(x_datas_val2).reshape(len(x_datas_val2), len(x_columns2))
y_val = np.array(y_datas_val)

x_datas_te = np.array(x_datas_te)
x_datas_te2 = np.array(x_datas_te2).reshape(len(x_datas_te2), len(x_columns2))
y_test = np.array(y_datas_te)


#standardize all data to start
scaler = StandardScaler()
x_train = scaler.fit_transform(x_datas.reshape(-1, x_datas.shape[-1])).reshape(x_datas.shape)
x_val = scaler.transform(x_datas_val.reshape(-1, x_datas_val.shape[-1])).reshape(x_datas_val.shape)
x_test = scaler.transform(x_datas_te.reshape(-1, x_datas_te.shape[-1])).reshape(x_datas_te.shape)

scaler2 = StandardScaler()
x_train2 = scaler2.fit_transform(x_datas2.reshape(-1, x_datas2.shape[-1])).reshape(x_datas2.shape)
x_val2 = scaler2.transform(x_datas_val2.reshape(-1, x_datas_val2.shape[-1])).reshape(x_datas_val2.shape)
x_test2 = scaler2.transform(x_datas_te2.reshape(-1, x_datas_te2.shape[-1])).reshape(x_datas_te2.shape)

#balance the data
rus = RandomUnderSampler()
rus.fit_resample(x_train[:,:,0], y_train)
#use the indices for resampling
x_train = x_train[rus.sample_indices_] 
x_train2 = x_train2[rus.sample_indices_] 
y_train = y_train[rus.sample_indices_] 

rus.fit_resample(x_val[:,:,0], y_val)
#use the indices for resampling
x_val = x_val[rus.sample_indices_] 
x_val2 = x_val2[rus.sample_indices_] 
y_val = y_val[rus.sample_indices_] 

rus.fit_resample(x_test[:,:,0], y_test)
#use the indices for resampling
x_test = x_test[rus.sample_indices_] 
x_test2 = x_test2[rus.sample_indices_] 
y_test = y_test[rus.sample_indices_] 

#sample through time, 1 day of information per week (1 out of each 5 tradin days)
x_train = time_sampler(x_train, timestep_size = 5)
x_val = time_sampler(x_val, timestep_size = 5)
x_test = time_sampler(x_test, timestep_size = 5)

#apply savgol smoothing with 3 week window
x_train = savgol_smooth(x_train, window = 3, poly = 2)
x_val = savgol_smooth(x_val, window = 3, poly = 2)
x_test = savgol_smooth(x_test, window = 3, poly = 2)

print(np.mean(y_train))
print(np.mean(y_val))
print(np.mean(y_test))

#%%
'''
Proper rolling train/test kfold validation

NOTE: Functions from previous chunk needed
'''
#%%

def add_months(date_string, months):
    input_date = datetime.strptime(date_string, '%Y-%m-%d')
    new_date = input_date + timedelta(days=months*30)  # Approximate
    new_date_string = new_date.strftime('%Y-%m-%d')
    return new_date_string

def index_resample(datax, datax2, datay):
    rus = RandomUnderSampler()
    rus.fit_resample(datax[:,:,0], datay)
    random.shuffle(rus.sample_indices_)
    datax = datax[rus.sample_indices_] 
    datax2 = datax2[rus.sample_indices_] 
    datay = datay[rus.sample_indices_] 
    return datax, datax2, datay

prop_tickers = 0.25
num_dates_tr = 100
num_trd_days = 104
x_columns2 = [ 'Date', 'bb_center21', #'Open','High','Low','Close','Adj Close',
                    'bb_upp21','bb_low21','bb_center84','bb_upp84','bb_low84',
                    'bb_center252','bb_upp252','bb_low252','minmax21','minmax84',
                    'minmax252','atr_21','atr_84','atr_252','rsi_21','rsi_84',
                    'rsi_252','vol_21','vol_84','vol_252']
x_columns = ['Volume','Close']

y_columns = ['gain_10']

datas = datas[datas['Date'] < '2023-01-01']

dataset1 = get_rnn_data2(datas, valid_tickers, x_columns, y_columns, 
                                num_trd_days, num_dates_tr, prop_tickers, x_columns2)
dataset2 = get_rnn_data2(datas, valid_tickers, x_columns, y_columns, 
                                num_trd_days, num_dates_tr, prop_tickers, x_columns2)
dataset3 = get_rnn_data2(datas, valid_tickers, x_columns, y_columns, 
                                num_trd_days, num_dates_tr, prop_tickers, x_columns2)

datasets = [dataset1, dataset2, dataset3]

#segment out several folds by start/end date. IE: End day of train should be 1+ day before val start
dataset3 = datasets[2]
#covert to array and drop outliars to start?
bdata_x = dataset3[0]
bdata_x2 = dataset3[2]
bdata_y = dataset3[1]

bdata_x2 = np.array(bdata_x2)
bdata_x = np.array(bdata_x)
bdata_y = np.array(bdata_y)

bdata_x, bdata_y, bdata_x2 = remove_arr_outliar2(bdata_x, bdata_y, bdata_x2, limit = 7)

#get end dates
date_col = bdata_x2[:,0]
date_arr = []
for date in range(len(date_col)):
    date_arr.append(add_months(date_col[date], 6))
date_arr = np.concatenate([date_col.reshape(-1,1), np.array(date_arr).reshape(-1,1)], axis = 1)

#create start/end partitions
z = ((date_arr[:,0] < '2017-06-01') & (date_arr[:,1] < '2018-01-01'))
z2 = ((date_arr[:,0] < '2018-01-01') & (date_arr[:,1] < '2019-06-01'))
z3 = ((date_arr[:,0] < '2019-06-01') & (date_arr[:,1] < '2021-01-01'))
z4 = ((date_arr[:,0] < '2021-01-01') & (date_arr[:,1] < '2022-06-01'))
z5 = ((date_arr[:,0] < '2022-06-01'))

date_partitions = np.concatenate([z.reshape(-1,1), z2.reshape(-1,1), z3.reshape(-1,1),
                                  z4.reshape(-1,1), z5.reshape(-1,1)], axis = 1)

#drop date from bdata_x2
bdata_x2 = bdata_x2[:,1:]

#split datasets out by partitions, 
#balancing EACH parition along the way,
# and sampling through time and smoothing  
train_data = []
val_data = []
for i in range(date_partitions.shape[1] - 1):
    tr_x = bdata_x[date_partitions[:,i]]
    tr_x2 = bdata_x2[date_partitions[:,i]]
    tr_y = bdata_y[date_partitions[:,i]]
    tr_x, tr_x2, tr_y = index_resample(tr_x, tr_x2, tr_y)
    #rescale data
    scaler = StandardScaler()
    tr_x = scaler.fit_transform(tr_x.reshape(-1, tr_x.shape[-1])).reshape(tr_x.shape)
    scaler2 = StandardScaler()
    tr_x2 = scaler2.fit_transform(tr_x2.reshape(-1, tr_x2.shape[-1])).reshape(tr_x2.shape)
    #smooth time
    tr_x = time_sampler(tr_x, timestep_size = 5)
    tr_x = savgol_smooth(tr_x, window = 3, poly = 2)
    train_data.append([tr_x, tr_x2, tr_y])
    #validation data comes from the next partition
    val_x = bdata_x[(date_partitions[:,i + 1]) & (~date_partitions[:,i])]
    val_x2 = bdata_x2[(date_partitions[:,i + 1]) & (~date_partitions[:,i])]
    val_y = bdata_y[(date_partitions[:,i + 1]) & (~date_partitions[:,i])]
    val_x, val_x2, val_y = index_resample(val_x, val_x2, val_y)
    #rescale data
    val_x = scaler.transform(val_x.reshape(-1, val_x.shape[-1])).reshape(val_x.shape)
    val_x2 = scaler2.transform(val_x2.reshape(-1, val_x2.shape[-1])).reshape(val_x2.shape)
    #smooth time
    val_x = time_sampler(val_x, timestep_size = 5)
    val_x = savgol_smooth(val_x, window = 3, poly = 2)
    val_data.append([val_x, val_x2, val_y])


dataset3 = [train_data, val_data]



datasets = [dataset1, dataset2, dataset3]

del datas, bdata_x, bdata_x2, bdata_y, temp_data, date_arr, date_col, z,z2,z3,z4,z5

#%%
'''
Prepare a simple RNN model


Next Steps: Add Online method to update data AND/OR proper Train/test splitting 
for time series
'''
#%%

#using paritions now!

#assemble RNN model
x_inputs = Input(shape=(tr_x.shape[1], tr_x.shape[2]))
x = GRU(10, return_sequences = True, activation = 'tanh', recurrent_dropout=0.3)(x_inputs)
x = GRU(10, return_sequences = True, activation = 'tanh', recurrent_dropout=0.3)(x)
x = GRU(10, return_sequences = False, activation = 'tanh', recurrent_dropout=0.3)(x)

x_inputs2 = Input(shape = (tr_x2.shape[1]))
x2 = Dense(10, activation = 'relu')(x_inputs2)
x2 = Dense(10, activation = 'relu')(x2)

x_final = Concatenate()([x, x2])

x_out = Dense(1, activation = 'sigmoid')(x_final)

rnn_mod1 = Model(inputs = [x_inputs, x_inputs2], outputs = x_out)


#set the optimizer
opt = Nadam(learning_rate = 0.001)
rnn_mod1.compile(optimizer = opt, loss = 'binary_crossentropy',
                 metrics = ['acc'])

#prep callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience= 3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('rnn.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('..//rnn.csv')
]

#train!

#specify data with paritions

train_data = dataset1[0]
val_data = dataset1[1]


d_num = 2
x_train = train_data[d_num][0].astype('float32')
x_train2 = train_data[d_num][1].astype('float32')
y_train = train_data[d_num][2].astype('float32')
x_val = val_data[d_num][0].astype('float32')
x_val2 = val_data[d_num][1].astype('float32')
y_val = val_data[d_num][2].astype('float32')

rnn_mod1.fit([x_train, x_train2], y_train, epochs = 10, batch_size = 128,
             validation_data = ([x_val, x_val2], y_val),
             callbacks = callbacks)

#evaluate
train_pred_raw = rnn_mod1.predict([x_train, x_train2])
#test_pred_raw = rnn_mod1.predict([x_test, x_test2])
val_pred_raw = rnn_mod1.predict([x_val, x_val2])

thresh = 0.5

train_pred = np.where(train_pred_raw < thresh, 0, 1)
#test_pred = np.where(test_pred_raw < thresh, 0, 1)
val_pred = np.where(val_pred_raw < thresh, 0, 1)

train_acc = accuracy_score(y_train, train_pred)
train_conf = confusion_matrix(y_train, train_pred)
train_prec = precision_score(y_train, train_pred)
train_recall = recall_score(y_train, train_pred)

val_acc = accuracy_score(y_val, val_pred)
val_conf = confusion_matrix(y_val, val_pred)
val_prec = precision_score(y_val, val_pred)
val_recall = recall_score(y_val, val_pred)

#test_acc = accuracy_score(y_test, test_pred)
#test_conf = confusion_matrix(y_test, test_pred)
#test_prec = precision_score(y_test, test_pred)
#test_recall = recall_score(y_test, test_pred)

#use heatmaps to display values of interest
ax= plt.subplot()
sns.heatmap(train_conf, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Training Set Accuracy: ' + str(np.round(train_acc * 100, 2)) + '%'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.show()

ax= plt.subplot()
sns.heatmap(val_conf, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Validation Set Accuracy: ' + str(np.round(val_acc * 100, 2)) + '%'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.show()

ax= plt.subplot()
sns.heatmap(test_conf, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Test Set Accuracy: ' + str(np.round(test_acc * 100, 2)) + '%'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.show()


#ROC curve
fpr , tpr , thresholds = roc_curve (y_val , val_pred_raw)
auc = roc_auc_score(y_val, val_pred_raw)
plt.plot(fpr,tpr) 
plt.axis([0,1,0,1]) 
plt.title('AUC: ' + str(auc))
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.show()

rnn_mod1.summary()
plot_model(rnn_mod1)


#%%
'''
ENSEMBLING AND EVALUATING MULTI MODELS!

Prepare a system to run and save key model metrics (AUC, Accuracy, confusion matrices)
over the multiple time partions and dataset samples 


CONCLUSION: ENSEMBLING MAKES SLIGHTLY BETTER, BUT BARELY
TUNE DENSE AND DUAL RNN THEN FINALIZE IF RF IS ACTUALLY BETTER OR NOT
'''
#%%

def build_dual_rnn(x1_shape1, x1_shape2, x2_shape1, rnn_ct, dense_ct,
                   dropout, lr, rnn_act, dense_act):
    x_inputs = Input(shape=(x1_shape1, x1_shape2))
    x = GRU(rnn_ct, return_sequences = True, activation = rnn_act, recurrent_dropout=dropout)(x_inputs)
    x = GRU(rnn_ct, return_sequences = False, activation = rnn_act, recurrent_dropout=dropout)(x)
    
    x_inputs2 = Input(shape = (x2_shape1))
    x2 = Dense(dense_ct, activation = dense_act)(x_inputs2)
    x2 = Dense(dense_ct, activation = dense_act)(x2)
    x2 = Dropout(dropout)(x2)
    
    x_final = Concatenate()([x, x2])
    x_out = Dense(1, activation = 'sigmoid')(x_final)
    rnn_mod1 = Model(inputs = [x_inputs, x_inputs2], outputs = x_out)
    #set the optimizer
    opt = Nadam(learning_rate = lr)
    rnn_mod1.compile(optimizer = opt, loss = 'binary_crossentropy',
                     metrics = ['acc'])
    
    return rnn_mod1

def build_rnn(x1_shape1, x1_shape2, rnn_ct, dropout, lr, rnn_act):
    x_inputs = Input(shape=(x1_shape1, x1_shape2))
    x = GRU(rnn_ct, return_sequences = True, activation = rnn_act, recurrent_dropout=dropout)(x_inputs)
    x = GRU(int(rnn_ct/2), return_sequences = True, activation = rnn_act, recurrent_dropout=dropout)(x)
    x = GRU(int(rnn_ct/3), return_sequences = False, activation = rnn_act, recurrent_dropout=dropout)(x)
    x_out = Dense(1, activation = 'sigmoid')(x)
    rnn_mod1 = Model(inputs = x_inputs, outputs = x_out)
    #set the optimizer
    opt = Nadam(learning_rate = lr)
    rnn_mod1.compile(optimizer = opt, loss = 'binary_crossentropy',
                     metrics = ['acc'])
    
    return rnn_mod1

def build_dense(x2_shape1, dense_ct, dropout, lr, dense_act):
    x_inputs2 = Input(shape = (x2_shape1))
    x2 = Dense(dense_ct, activation = dense_act)(x_inputs2)
    x2 = Dense(int(dense_ct/2), activation = dense_act)(x2)
    x2 = Dense(int(dense_ct/3), activation = dense_act)(x2)
    x2 = Dropout(dropout)(x2)
    x_out = Dense(1, activation = 'sigmoid')(x2)
    dense_mod = Model(inputs = x_inputs2, outputs = x_out)
    #set the optimizer
    opt = Nadam(learning_rate = lr)
    dense_mod.compile(optimizer = opt, loss = 'binary_crossentropy',
                     metrics = ['acc'])
    return dense_mod

def data_extract(train_data, val_data, section_num, fold_num):
    train = train_data[fold_num][section_num].astype('float32')
    val = val_data[fold_num][section_num].astype('float32')
    return train, val


def run_dual_rnn(dataset):
    #deploy data
    train_data = dataset[0]
    val_data = dataset[1]
    #extract each K fold and train model, then save results
    #extract k_fold
    accs = []
    confs = []
    aucs = []
    smpl_cts = []
    raw_preds = []
    for i in range(len(train_data)):
        #build blank model
        dual_rnn_mod1 = build_dual_rnn(x1_shape1 = 21, x1_shape2 = 2, x2_shape1 = 21, 
                                  rnn_ct = 10, dense_ct = 10, dropout = 0.3, lr = 0.001,
                                  rnn_act = 'tanh', dense_act = 'relu')
        #extract data
        x_train, x_val = data_extract(train_data, val_data, section_num = 0, fold_num = i)
        x_train2, x_val2 = data_extract(train_data, val_data, section_num = 1, fold_num = i)
        y_train, y_val = data_extract(train_data, val_data, section_num = 2, fold_num = i)
        #train!
        dual_rnn_mod1.fit([x_train, x_train2], y_train, epochs = 25, batch_size = 128,
                     validation_data = ([x_val, x_val2], y_val),
                     callbacks = callbacks)
        #calcualte key validation metrics (AUC, Acc, conf_matrix)
        val_pred_raw = dual_rnn_mod1.predict([x_val, x_val2])
        thresh = 0.5
        val_pred = np.where(val_pred_raw < thresh, 0, 1)
        #save scores and number of samples (for weighting?)
        accs.append(accuracy_score(y_val, val_pred))
        confs.append(confusion_matrix(y_val, val_pred))
        aucs.append(roc_auc_score(y_val, val_pred_raw))
        smpl_cts.append(len(y_val))
        raw_preds.append(val_pred_raw)
    return [accs, confs, aucs, smpl_cts, raw_preds]

def run_rnn(dataset):
    #deploy data
    train_data = dataset[0]
    val_data = dataset[1]
    #extract each K fold and train model, then save results
    #extract k_fold
    accs = []
    confs = []
    aucs = []
    smpl_cts = []
    raw_preds = []
    for i in range(len(train_data)):
        #build blank model
        rnn_mod1 = build_rnn(x1_shape1 = 21, x1_shape2 = 2, rnn_ct = 8, 
                             dropout = 0.3, lr = 0.001, rnn_act = 'tanh')
        #extract data
        x_train, x_val = data_extract(train_data, val_data, section_num = 0, fold_num = i)
        y_train, y_val = data_extract(train_data, val_data, section_num = 2, fold_num = i)
        #train!
        rnn_mod1.fit(x_train, y_train, epochs = 25, batch_size = 128,
                     validation_data = (x_val, y_val),
                     callbacks = callbacks)
        #calcualte key validation metrics (AUC, Acc, conf_matrix)
        val_pred_raw = rnn_mod1.predict(x_val)
        thresh = 0.5
        val_pred = np.where(val_pred_raw < thresh, 0, 1)
        #save scores and number of samples (for weighting?)
        accs.append(accuracy_score(y_val, val_pred))
        confs.append(confusion_matrix(y_val, val_pred))
        aucs.append(roc_auc_score(y_val, val_pred_raw))
        smpl_cts.append(len(y_val))
        raw_preds.append(val_pred_raw)
    return [accs, confs, aucs, smpl_cts, raw_preds]

def run_dense(dataset):
    #deploy data
    train_data = dataset[0]
    val_data = dataset[1]
    #extract each K fold and train model, then save results
    #extract k_fold
    accs = []
    confs = []
    aucs = []
    smpl_cts = []
    raw_preds = []
    for i in range(len(train_data)):
        #build blank model
        dense_mod1 = build_dense(x2_shape1 = 21, dense_ct = 16, dropout = 0.3, 
                               lr = 0.001, dense_act = 'relu')
        #extract data
        x_train, x_val = data_extract(train_data, val_data, section_num = 1, fold_num = i)
        y_train, y_val = data_extract(train_data, val_data, section_num = 2, fold_num = i)
        #train!
        dense_mod1.fit(x_train, y_train, epochs = 25, batch_size = 128,
                     validation_data = (x_val, y_val),
                     callbacks = callbacks)
        #calcualte key validation metrics (AUC, Acc, conf_matrix)
        val_pred_raw = dense_mod1.predict(x_val)
        thresh = 0.5
        val_pred = np.where(val_pred_raw < thresh, 0, 1)
        #save scores and number of samples (for weighting?)
        accs.append(accuracy_score(y_val, val_pred))
        confs.append(confusion_matrix(y_val, val_pred))
        aucs.append(roc_auc_score(y_val, val_pred_raw))
        smpl_cts.append(len(y_val))
        raw_preds.append(val_pred_raw)
    return [accs, confs, aucs, smpl_cts, raw_preds]

def run_rf(dataset):
    #deploy data
    train_data = dataset[0]
    val_data = dataset[1]
    #extract each K fold and train model, then save results
    #extract k_fold
    accs = []
    confs = []
    aucs = []
    smpl_cts = []
    raw_preds = []
    for i in range(len(train_data)):
        #prep a blank model
        rfc = RandomForestClassifier(max_depth = 7)
        #extract data
        x_train, x_val = data_extract(train_data, val_data, section_num = 1, fold_num = i)
        y_train, y_val = data_extract(train_data, val_data, section_num = 2, fold_num = i)
        y_train = y_train.reshape(len(y_train),)
        y_val = y_val.reshape(len(y_val),)
        #train!
        rfc.fit(x_train, y_train)
        val_pred = rfc.predict(x_val)
        val_pred_raw = rfc.predict_proba(x_val)[:,1]
        #save scores and number of samples (for weighting?)
        accs.append(accuracy_score(y_val, val_pred))
        confs.append(confusion_matrix(y_val, val_pred))
        aucs.append(roc_auc_score(y_val, val_pred_raw))
        smpl_cts.append(len(y_val))
        raw_preds.append(val_pred_raw)
    return [accs, confs, aucs, smpl_cts, raw_preds]

def run_xgboost(dataset):
    #deploy data
    train_data = dataset[0]
    val_data = dataset[1]
    #extract each K fold and train model, then save results
    #extract k_fold
    accs = []
    confs = []
    aucs = []
    smpl_cts = []
    raw_preds = []
    for i in range(len(train_data)):
        #prep blank model
        xgbc = xgb.XGBClassifier(tree_method="hist")
        #extract data
        x_train, x_val = data_extract(train_data, val_data, section_num = 1, fold_num = i)
        y_train, y_val = data_extract(train_data, val_data, section_num = 2, fold_num = i)
        y_train = y_train.reshape(len(y_train),)
        y_val = y_val.reshape(len(y_val),)
        #train!
        xgbc.fit(x_train, y_train)
        val_pred = xgbc.predict(x_val)
        val_pred_raw = xgbc.predict_proba(x_val)[:,1]
        #save scores and number of samples (for weighting?)
        accs.append(accuracy_score(y_val, val_pred))
        confs.append(confusion_matrix(y_val, val_pred))
        aucs.append(roc_auc_score(y_val, val_pred_raw))
        smpl_cts.append(len(y_val))
        raw_preds.append(val_pred_raw)
    return [accs, confs, aucs, smpl_cts, raw_preds]


def run_mars(dataset):
    #deploy data
    train_data = dataset[0]
    val_data = dataset[1]
    #extract each K fold and train model, then save results
    #extract k_fold
    accs = []
    confs = []
    aucs = []
    smpl_cts = []
    raw_preds = []
    for i in range(len(train_data)):
        #prep blank model
        mars = pyearth.Earth()
        #extract data
        x_train, x_val = data_extract(train_data, val_data, section_num = 1, fold_num = i)
        y_train, y_val = data_extract(train_data, val_data, section_num = 2, fold_num = i)
        y_train = y_train.reshape(len(y_train),)
        y_val = y_val.reshape(len(y_val),)
        #train!
        mars.fit(x_train, y_train)
        val_pred_raw = mars.predict(x_val)
        val_pred = np.where(val_pred_raw < 0.5, 0, 1)
        #save scores and number of samples (for weighting?)
        accs.append(accuracy_score(y_val, val_pred))
        confs.append(confusion_matrix(y_val, val_pred))
        aucs.append(roc_auc_score(y_val, val_pred_raw))
        smpl_cts.append(len(y_val))
        raw_preds.append(val_pred_raw)
    return [accs, confs, aucs, smpl_cts, raw_preds]


#specify callbacks for all neural nets
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience= 3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('rnn.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('..//rnn.csv')
]


model_outputs = []
for k in range(3):
    #grab data
    dataset = datasets[k]
    #run each model
    mars_out = run_mars(dataset)
    rf_out = run_rf(dataset)
    xgb_out = run_xgboost(dataset)
    dual_out = run_dual_rnn(dataset)
    rnn_out = run_rnn(dataset)
    dense_out = run_dense(dataset)
    #save results
    model_outputs.append([mars_out, rf_out, xgb_out, dual_out, rnn_out, dense_out])
    

#take the raw outputs, and create a equal power voting ensemble 
#take the raw outputs, and create a logistic regression voting ensemble 
#MUCH BETTER THAN STANDARD VOTING BUTTTTT NO VALIDATION DATA FOR THE LOGISTIC REG
all_log_ens_accs = []
all_log_ens_aucs = []
for i in range(len(model_outputs)): #open dataset
    dataset = datasets[i]
    val_data = dataset[1]
    log_ens_accs = []
    log_ens_aucs = []
    for k in range(4): #in each fold
        y_val = val_data[k][-1]    
        preds_log_ens = []
        for j in range(6): #open model 
            preds = model_outputs[i][j][-1][k] #get logits
            preds_log_ens.append(preds.reshape(len(preds),))
        preds_log_ens = np.array(preds_log_ens).transpose()
        #set up logistic regression classifier
        clf = LogisticRegression().fit(preds_log_ens, y_val.reshape(len(y_val),))
        final_pred = clf.predict(preds_log_ens)
        final_pred_proba = clf.predict_log_proba(preds_log_ens)[:,1]
        log_ens_accs.append(accuracy_score(y_val, final_pred))
        log_ens_aucs.append(roc_auc_score(y_val, final_pred_proba))
    all_log_ens_accs.append(np.mean(log_ens_accs))
    all_log_ens_aucs.append(np.mean(log_ens_aucs))
    
#get the average acc and AUC across time for each dataset
all_accs = []
all_aucs = []
for i in range(len(model_outputs)): #open dataset info
    mod_accs = []
    mod_aucs = []
    for k in range(6): #open model info
         mod_acc = np.mean(model_outputs[i][k][0]) # accuracy
         mod_auc = np.mean(model_outputs[i][k][2]) # AUC
         mod_accs.append(mod_acc)
         mod_aucs.append(mod_auc)
    all_accs.append(mod_accs)
    all_aucs.append(mod_aucs)

all_accs = np.array(all_accs)
all_accs = all_accs.mean(axis = 0)
all_aucs = np.array(all_aucs)
all_aucs = all_aucs.mean(axis = 0)

#compare results
all_accs.mean()
np.mean(all_log_ens_accs)

all_aucs.mean()
np.mean(all_log_ens_aucs)

#%%
'''
Ensemble assement: 
    -Dropping the standard RNN looks to have minimal loss
    -all models agree gives close to best preformace
    -Stacked logisitic regression is close contender
One of these two should be the final Descion model
    -All agree at > 75% or 80%
    -Stacked at > 80%

'''
#%%
data1 = model_outputs[0]
model_1 = data1[0]
predictions = model_1[-1]
pred_fold1 = predictions[0]


def retrieve_predictions(data_output, fold):
    fold_preds = []
    for i in range(len(data_output)):
        model_outputs = data_output[i]
        predictions = model_outputs[-1]
        fold_n_preds = predictions[fold]
        fold_preds.append(fold_n_preds)
    #convert to dataframe
    preds_df = pd.DataFrame(fold_preds).transpose()
    preds_df.columns = ['mars', 'rf', 'xgb', 'dual', 'rnn', 'dense']
    #remove the list container for the NN preds
    preds_df['dual'] = preds_df['dual'].apply(lambda x: x[0])
    preds_df['rnn'] = preds_df['rnn'].apply(lambda x: x[0])
    preds_df['dense'] = preds_df['dense'].apply(lambda x: x[0])
    return preds_df

d1_f1_preds = retrieve_predictions(data1, 0)
d1_f2_preds = retrieve_predictions(data1, 1)
d1_f3_preds = retrieve_predictions(data1, 2)
d1_f4_preds = retrieve_predictions(data1, 3)

#get accuracies individually:
fold = 3
dataset = datasets[0]
val_data = dataset[1]
y_val = val_data[fold][-1]    

data_investigate = d1_f4_preds

for i in range(6):
    y_pred = np.where(data_investigate.iloc[:,i] < 0.5, 0, 1)
    acc_score = precision_score(y_val, y_pred)
    print(str(i) + ': ' + str(np.round(acc_score, 3)))

### look at ensembled accuracies (confusion matrix really) ###
#drop RNN (super low confidence) -- Does help for All agreement at 50% conf!!


data_investigate = data_investigate[['mars', 'rf', 'xgb', 'dual', 'dense']]

#all in agrement

thresh = 0.9

new_preds = []
for i in range(len(data_investigate)):
    new_preds.append(int(np.where(all(data_investigate.iloc[i,:] > thresh), 1, 0)))
    
ens_pre = precision_score(y_val, new_preds)
ens_conf = confusion_matrix(y_val, new_preds)

ax= plt.subplot()
sns.heatmap(ens_conf, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('All Agree Precision ' + str(thresh * 100) + '% conf: ' + str(np.round(ens_pre * 100, 2)) + '%'); 
ax.xaxis.set_ticklabels(['Do Not Buy', 'Buy']); ax.yaxis.set_ticklabels(['Do Not Buy', 'Buy']);
plt.show()

#equal voting average
new_preds = []
for i in range(len(data_investigate)):
    new_preds.append(np.where(np.mean(data_investigate.iloc[i,:]) > thresh, 1, 0))
    
ens_pre = precision_score(y_val, new_preds)
ens_conf = confusion_matrix(y_val, new_preds)

ax= plt.subplot()
sns.heatmap(ens_conf, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('All Vote Precision ' + str(thresh * 100) + '% conf: ' + str(np.round(ens_pre * 100, 2)) + '%'); 
ax.xaxis.set_ticklabels(['Do Not Buy', 'Buy']); ax.yaxis.set_ticklabels(['Do Not Buy', 'Buy']);
plt.show()

#stacked voting by logistic regression
clf = LogisticRegression().fit(data_investigate, y_val.reshape(len(y_val),))
log_preds_prob = clf.predict_proba(data_investigate)[:,1]
log_preds = np.where(log_preds_prob > thresh, 1, 0)

ens_pre = precision_score(y_val, log_preds)
ens_conf = confusion_matrix(y_val, log_preds)

ax= plt.subplot()
sns.heatmap(ens_conf, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Stacked Precision ' + str(thresh * 100) + '% conf: ' + str(np.round(ens_pre * 100, 2)) + '%'); 
ax.xaxis.set_ticklabels(['Do Not Buy', 'Buy']); ax.yaxis.set_ticklabels(['Do Not Buy', 'Buy']);
plt.show()


#%%
'''
Model combination: Train dense and RNN then combine with finetunig
'''
#%%

#extract data from cross fold analysis
x_train = train_data[3]
x_train = x_train[0]
x_train2 = x_train[1]
y_train = x_train[2]
x_val = val_data[3]
x_val = x_val[0]
x_val2 = x_val[1]
y_val = x_val[2]

dropout = 0.3

#build and train RNN
x_inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
x = GRU(10, return_sequences = True, activation = 'tanh', recurrent_dropout=dropout)(x_inputs)
x = GRU(10, return_sequences = True, activation = 'tanh', recurrent_dropout=dropout)(x)
x = GRU(10, return_sequences = False, activation = 'tanh', recurrent_dropout=dropout)(x)
x_out = Dense(1, activation = 'sigmoid')(x)
rnn_mod1 = Model(inputs = [x_inputs], outputs = x_out)
#set the optimizer
opt = Nadam(learning_rate = 0.001)
rnn_mod1.compile(optimizer = opt, loss = 'binary_crossentropy',
                 metrics = ['acc'])

rnn_mod1.fit(x_train, y_train, batch_size = 128,
             epochs = 5)

rnn_pred = np.where(rnn_mod1.predict(x_val) >= 0.5, 1,0)
rnn_acc = accuracy_score(y_val, rnn_pred)

#build and train Dense
x_inputs2 = Input(shape = (x_train2.shape[1]))
x2 = Dense(16, activation = 'tanh')(x_inputs2)
x2 = Dense(16, activation = 'tanh')(x2)
x2 = Dense(16, activation = 'tanh')(x2)
x2 = Dropout(dropout)(x2)
x_out = Dense(1, activation = 'sigmoid')(x2)
dense_mod = Model(inputs = [x_inputs2], outputs = x_out)

opt = Nadam(learning_rate = 0.001)
dense_mod.compile(optimizer = opt, loss = 'binary_crossentropy',
                 metrics = ['acc'])

dense_mod.fit(x_train2, y_train, batch_size = 128,
             epochs = 5)

dense_pred = np.where(dense_mod.predict(x_val2) >= 0.5, 1,0)
dense_acc = accuracy_score(y_val, dense_pred)

#combine and finetune
rnn_xin = rnn_mod1.layers[0].input
rnn_xout = rnn_mod1.layers[3].output
dense_xin = dense_mod.layers[0].input
dense_xout = dense_mod.layers[3].output
x_final = Concatenate()([rnn_xout, dense_xout])
x_out_final = Dense(1, activation = 'sigmoid')(x_final)
combo = Model(inputs = [rnn_xin, dense_xin], outputs = x_out_final)
#freeze layers
for layer in combo.layers[:8]:
    layer.trainable = False

#set the optimizer
opt = Nadam(learning_rate = 0.0001)
combo.compile(optimizer = opt, loss = 'binary_crossentropy',
                 metrics = ['acc'])
combo.fit([x_train, x_train2], y_train, batch_size = 128,
             epochs = 5)

combo_pred = np.where(combo.predict([x_val, x_val2]) >= 0.5, 1,0)
combo_acc = accuracy_score(y_val, combo_pred)


#check against a standard dual RNN
dual = build_dual_rnn(x_train.shape[1], x_train.shape[2], x_train2.shape[1], 16, 10,
                   0.3, 0.001, 'tanh', 'tanh')

dual.fit([x_train, x_train2], y_train, batch_size = 128,
             epochs = 5)
dual_pred = np.where(dual.predict([x_val, x_val2]) >= 0.5, 1,0)
dual_acc = accuracy_score(y_val, dual_pred)

print('RNN Accuracy: ' + str(np.round(rnn_acc, 4)))
print('Dense Accuracy: ' + str(np.round(dense_acc, 4)))
print('Combo Accuracy: ' + str(np.round(combo_acc, 4)))
print('Dual accuracy: ' + str(np.round(dual_acc, 4)))
#%%
'''
Model tuning: Dual RNN, Dense RNN

Dense RNN articture : [lr = 0.001, act = 'tanh', ct = [16, 16, 16], epochs = 50, batch = 128: 0.699/0.766]

'''
#%%
def build_dual_rnn(x1_shape1, x1_shape2, x2_shape1, rnn_ct, dense_ct,
                   dropout, lr, rnn_act, dense_act):
    x_inputs = Input(shape=(x1_shape1, x1_shape2))
    x = GRU(rnn_ct[0], return_sequences = True, activation = rnn_act, recurrent_dropout=dropout)(x_inputs)
    x = GRU(rnn_ct[1], return_sequences = True, activation = rnn_act, recurrent_dropout=dropout)(x)
    x = GRU(rnn_ct[2], return_sequences = False, activation = rnn_act, recurrent_dropout=dropout)(x)
    
    x_inputs2 = Input(shape = (x2_shape1))
    x2 = Dense(dense_ct, activation = dense_act)(x_inputs2)
    x2 = Dense(dense_ct, activation = dense_act)(x2)
    x2 = Dense(dense_ct, activation = dense_act)(x2)
    x2 = Dropout(dropout)(x2)
    
    x_final = Concatenate()([x, x2])
    x_out = Dense(1, activation = 'sigmoid')(x_final)
    rnn_mod1 = Model(inputs = [x_inputs, x_inputs2], outputs = x_out)
    #set the optimizer
    opt = Nadam(learning_rate = lr)
    rnn_mod1.compile(optimizer = opt, loss = 'binary_crossentropy',
                     metrics = ['acc'])
    
    return rnn_mod1


def build_dense(x2_shape1, dense_ct, dropout, lr, dense_act):
    x_inputs2 = Input(shape = (x2_shape1))
    x2 = Dense(dense_ct[0], activation = dense_act)(x_inputs2)
    x2 = Dense(dense_ct[1], activation = dense_act)(x2)
    x2 = Dense(dense_ct[2], activation = dense_act)(x2)
    x2 = Dropout(dropout)(x2)
    x_out = Dense(1, activation = 'sigmoid')(x2)
    dense_mod = Model(inputs = x_inputs2, outputs = x_out)
    #set the optimizer
    opt = Nadam(learning_rate = lr)
    dense_mod.compile(optimizer = opt, loss = 'binary_crossentropy',
                     metrics = ['acc'])
    return dense_mod


def run_dense(dataset, lr, act, ct, epochs, batch):
    #deploy data
    train_data = dataset[0]
    val_data = dataset[1]
    #extract each K fold and train model, then save results
    #extract k_fold
    accs = []
    confs = []
    aucs = []
    smpl_cts = []
    raw_preds = []
    for i in range(len(train_data)):
        #build blank model
        dense_mod1 = build_dense(x2_shape1 = 21, dense_ct = ct, dropout = 0.3, 
                               lr = lr, dense_act = act)
        #extract data
        x_train, x_val = data_extract(train_data, val_data, section_num = 1, fold_num = i)
        y_train, y_val = data_extract(train_data, val_data, section_num = 2, fold_num = i)
        #train!
        dense_mod1.fit(x_train, y_train, epochs = epochs, batch_size = batch,
                     validation_data = (x_val, y_val),
                     callbacks = callbacks)
        #calcualte key validation metrics (AUC, Acc, conf_matrix)
        val_pred_raw = dense_mod1.predict(x_val)
        thresh = 0.5
        val_pred = np.where(val_pred_raw < thresh, 0, 1)
        #save scores and number of samples (for weighting?)
        accs.append(accuracy_score(y_val, val_pred))
        confs.append(confusion_matrix(y_val, val_pred))
        aucs.append(roc_auc_score(y_val, val_pred_raw))
        smpl_cts.append(len(y_val))
        raw_preds.append(val_pred_raw)
    return [accs, confs, aucs, smpl_cts, raw_preds]


def run_dual_rnn(dataset, rnn_ct, batch, epochs, drop_out, rnn_act, dense_ct = 16):
    #deploy data
    train_data = dataset[0]
    val_data = dataset[1]
    #extract each K fold and train model, then save results
    #extract k_fold
    accs = []
    confs = []
    aucs = []
    smpl_cts = []
    raw_preds = []
    for i in range(len(train_data)):
        #build blank model
        dual_rnn_mod1 = build_dual_rnn(x1_shape1 = 21, x1_shape2 = 2, x2_shape1 = 21, 
                                  rnn_ct = rnn_ct, dense_ct = dense_ct, dropout = drop_out, lr = 0.001,
                                  rnn_act = rnn_act, dense_act = 'tanh')
        #extract data
        x_train, x_val = data_extract(train_data, val_data, section_num = 0, fold_num = i)
        x_train2, x_val2 = data_extract(train_data, val_data, section_num = 1, fold_num = i)
        y_train, y_val = data_extract(train_data, val_data, section_num = 2, fold_num = i)
        #train!
        dual_rnn_mod1.fit([x_train, x_train2], y_train, epochs = epochs, batch_size = batch,
                     validation_data = ([x_val, x_val2], y_val),
                     callbacks = callbacks)
        #calcualte key validation metrics (AUC, Acc, conf_matrix)
        val_pred_raw = dual_rnn_mod1.predict([x_val, x_val2])
        thresh = 0.5
        val_pred = np.where(val_pred_raw < thresh, 0, 1)
        #save scores and number of samples (for weighting?)
        accs.append(accuracy_score(y_val, val_pred))
        confs.append(confusion_matrix(y_val, val_pred))
        aucs.append(roc_auc_score(y_val, val_pred_raw))
        smpl_cts.append(len(y_val))
        raw_preds.append(val_pred_raw)
    return [accs, confs, aucs, smpl_cts, raw_preds]


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience= 5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('rnn.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('..//rnn.csv')
]


model_outputs = []
for k in range(3):
    #grab data
    dataset = datasets[k]
    #run model
    dense_out = run_dual_rnn(dataset, rnn_ct = [15,15,10], batch = 128, epochs = 50, 
                             drop_out = 0.3, rnn_act = 'relu', dense_ct = 16)
    #save results
    model_outputs.append(dense_out)
    
    
all_accs = []
all_aucs = []
for i in range(len(model_outputs)): #open dataset info
    mod_acc = np.mean(model_outputs[i][0]) # accuracy
    mod_auc = np.mean(model_outputs[i][2]) # AUC
    all_accs.append(mod_acc)
    all_aucs.append(mod_auc)

print(np.mean(all_accs))
print(np.mean(all_aucs))

###### RESULTS for Dual ######
[rnn_ct = [10,10,10], batch = 128, epochs = 50, drop_out = 0.3, rnn_act = 'tanh', dense_ct = 16, 0.698/0.767]
[rnn_ct = [15,10,5], batch = 128, epochs = 50, drop_out = 0.3, rnn_act = 'tanh', dense_ct = 16, 0.699/0.766]
[rnn_ct = [15,15,10], batch = 128, epochs = 50, drop_out = 0.3, rnn_act = 'tanh', dense_ct = 16, 0./0.]

#%%
'''
MARS!!!

interpretation:
    Num - X;  B * X (Below Number)
    X - Num; B * X(Above Number)
    Num + X; B * X(Above -Number)
    X + Num; B * X(Below -Number)
    
    Need to Unstandardize to interpret directly!
'''
#%%

train = train_data[2]
val = val_data[2]

x_train = train[0]
x_train2 = train[1]
y_train = train[2]

x_val = val[0]
x_val2 = val[1]
y_val = val[2]

mars = pyearth.Earth()

mars.fit(x_train2, y_train)

train_pred_raw = mars.predict(x_train2)
train_pred = np.round(train_pred_raw)
val_pred_raw = mars.predict(x_val2)
val_pred = np.round(val_pred_raw)

train_acc = accuracy_score(y_train, train_pred)
train_conf = confusion_matrix(y_train, train_pred)

val_acc = accuracy_score(y_val, val_pred)
val_conf = confusion_matrix(y_val, val_pred)

print(train_acc)
print(train_conf)
print(val_acc)
print(val_conf)


print(mars.summary())


mars_columns = x_columns2[1:]
mars_nums = list(range(21))

mars_columns = pd.DataFrame([mars_columns, mars_nums]).transpose()


rfc_val_pred_raw = rfc.predict_proba(x_val2)[:,1]
rnn_val_pred_raw = rnn_mod1.predict([x_val, x_val2])
mars_val_pred_raw = mars.predict(x_val2)

rfc_val_pred = np.where(rfc_val_pred_raw < 0.5, 0, 1)
rnn_val_pred = np.where(rnn_val_pred_raw < 0.5, 0, 1)
mars_val_pred = np.where(mars_val_pred_raw < 0.5, 0, 1)


rfc_train_acc = accuracy_score(y_train, rfc_train_pred)
rfc_train_conf = confusion_matrix(y_train, rfc_train_pred)
rfc_train_auc = roc_auc_score(y_train, rfc_train_pred_raw)

rnn_train_acc = accuracy_score(y_train, rnn_train_pred)
rnn_train_conf = confusion_matrix(y_train, rnn_train_pred)
rnn_train_auc = roc_auc_score(y_train, rnn_train_pred_raw)

mars_train_acc = accuracy_score(y_train, mars_train_pred)
mars_train_conf = confusion_matrix(y_train, mars_train_pred)
mars_train_auc = roc_auc_score(y_train, mars_train_pred_raw)


rfc_val_acc = accuracy_score(y_val, rfc_val_pred)
rfc_val_conf = confusion_matrix(y_val, rfc_val_pred)
rfc_val_auc = roc_auc_score(y_val, rfc_val_pred_raw)

rnn_val_acc = accuracy_score(y_val, rnn_val_pred)
rnn_val_conf = confusion_matrix(y_val, rnn_val_pred)
rnn_val_auc = roc_auc_score(y_val, rnn_val_pred_raw)

mars_val_acc = accuracy_score(y_val, mars_val_pred)
mars_val_conf = confusion_matrix(y_val, mars_val_pred)
mars_val_auc = roc_auc_score(y_val, mars_val_pred_raw)


print('rfc train acc: ' + str(np.round(rfc_train_acc, 3)))
print('rfc train auc: ' + str(np.round(rfc_train_auc, 3)))
print(rfc_train_conf)

print('rnn train acc: ' + str(np.round(rnn_train_acc, 3)))
print('rnn train auc: ' + str(np.round(rnn_train_auc, 3)))
print(rnn_train_conf)

print('mars train acc: ' + str(np.round(mars_train_acc, 3)))
print('mars train auc: ' + str(np.round(mars_train_auc, 3)))
print(mars_train_conf)


print('rfc val acc: ' + str(np.round(rfc_val_acc, 3)))
print('rfc val auc: ' + str(np.round(rfc_val_auc, 3)))
print(rfc_val_conf)

print('rnn val acc: ' + str(np.round(rnn_val_acc, 3)))
print('rnn val auc: ' + str(np.round(rnn_val_auc, 3)))
print(rnn_val_conf)

print('mars val acc: ' + str(np.round(mars_val_acc, 3)))
print('mars val auc: ' + str(np.round(mars_val_auc, 3)))
print(mars_val_conf)

#%%
'''
FINAL MODEL CONSTRUCTIONS!

Models:
    RF Depth of 7, 100 Trees
    XGBoost Default Params
    MARS
    Dense ANN - 9 Epochs?
        16-12-8, tanh, batch = 128, Schedule, 5 patience, 0.4 dropout
    TABLED FOR NOW---
    Dual ANN-RNN 9 - Epochs?
        Dense: 12-12, tanh, batch = 128, Schedule, 5 patience, 0.4 dropout
        RNN: 10-10, tanh, batch = 128, Schedule, 5 patience, 0.4 dropout
    
'''

#%%
#get data
os.chdir(path2)
#read in names of each stock
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
            
            temp_data.drop(['ma_21', 'ma_84', 'ma_252'], axis = 1, inplace = True)
            
            datas.append(temp_data)
datas = pd.concat(datas)
#remove useless/redundant columns and na
datas.drop(['end_date', 'end_price'], axis = 1, inplace = True)
datas.dropna(inplace = True)
colz_norm_name = ['bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84', 
                  'bb_low84', 'bb_center252',  'bb_upp252', 'bb_low252', 'atr_21',
                  'atr_84', 'atr_252', 'beta_21', 'beta_84', 'beta_252'] 
datas_col = datas.columns
for i in range(len(colz_norm_name)):
    col = colz_norm_name[i]
    datas[col] = datas[col]/datas['Close']
#convert RSI to 0-1 scale
datas['rsi_21'] = datas['rsi_21']/100
datas['rsi_84'] = datas['rsi_84']/100
datas['rsi_252'] = datas['ris_252']/100
datas.drop(['ris_252'], inplace = True, axis = 1) #remove the typo
datas.reset_index(inplace = True, drop = True)

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

def remove_arr_outliar3(y_datas, x_datas, limit = 7):
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


#data functions and initial grab from the cross val section
prop_tickers = 0.2
num_dates_tr = 100
num_trd_days = 104
x_columns2 = [ 'Volume', 'bb_center21', #'Open','High','Low','Close','Adj Close',
                    'bb_upp21','bb_low21','bb_center84','bb_upp84','bb_low84',
                    'bb_center252','bb_upp252','bb_low252','minmax21','minmax84',
                    'minmax252','atr_21','atr_84','atr_252','rsi_21','rsi_84',
                    'rsi_252','vol_21','vol_84','vol_252', 'beta_21', 'beta_84', 'beta_252']
x_columns3 = [ 'Volume', 'bb_center21', #'Open','High','Low','Close','Adj Close',
                    'bb_upp21','bb_low21','bb_center84','bb_upp84','bb_low84',
                    'bb_center252','bb_upp252','bb_low252','minmax21','minmax84',
                    'minmax252','atr_21','atr_84','atr_252','rsi_21','rsi_84',
                    'rsi_252','vol_21','vol_84','vol_252']
x_columns = ['Volume','Close']

y_columns = ['gain_10']

#datas = datas[datas['Date'] < '2021-06-01']

datas = datas[(datas['Close'] > 2) & (datas['Close'] < 100)]

#set up training data into 4 sections.
datas_tr1 = datas[datas['Date'] < '2015-01-01']
datas_tr2 = datas[((datas['Date'] > '2015-01-01') & (datas['Date'] <= '2017-01-01'))]
datas_tr3 = datas[((datas['Date'] > '2017-01-01') & (datas['Date'] <= '2019-01-01'))]
datas_tr4 = datas[((datas['Date'] > '2019-01-01') & (datas['Date'] <= '2021-01-01'))]

datas_val = datas[datas['Date'] > '2021-01-01'] #validation data 


####### SKIP THIS SECTION UNLESS DOING DUAL RNN #######
#extract the data
datas_tr = get_rnn_data2(datas, valid_tickers, x_columns, y_columns, 
                                num_trd_days, num_dates_tr, prop_tickers, x_columns2)
datas_val = get_rnn_data2(datas_val, valid_tickers, x_columns, y_columns, 
                                num_trd_days, 35, prop_tickers, x_columns2)
#covert to array and drop outliars to start?
x_train = datas_tr[0]
x_train2 = datas_tr[2]
y_train = datas_tr[1]
x_train = np.array(x_train)
x_train2 = np.array(x_train2)
y_train = np.array(y_train)

x_val = datas_val[0]
x_val2 = datas_val[2]
y_val = datas_val[1]
x_val = np.array(x_val)
x_val2 = np.array(x_val2)
y_val = np.array(y_val)

#drop dates
x_train2 = x_train2[:,1:]
x_val2 = x_val2[:,1:]

#remove training outlairs
x_train, y_train, x_train2 = remove_arr_outliar2(x_train, y_train, x_train2, limit = 7)

#balance, standardize and smooth data
train_data = []
val_data = []

#balance the data
tr_x, tr_x2, tr_y = index_resample(x_train, x_train2, y_train)

#rescale data - SKIP FOR TABULAR DATA!!
scaler = StandardScaler()
tr_x = scaler.fit_transform(tr_x.reshape(-1, tr_x.shape[-1])).reshape(tr_x.shape)
scaler2 = StandardScaler()
tr_x2 = scaler2.fit_transform(tr_x2.reshape(-1, tr_x2.shape[-1])).reshape(tr_x2.shape)
#smooth time
tr_x = time_sampler(tr_x, timestep_size = 5)
tr_x = savgol_smooth(tr_x, window = 3, poly = 2)
#repeat for val data
val_x, val_x2, val_y = index_resample(x_val, x_val2, y_val)
#rescale data
val_x = scaler.transform(val_x.reshape(-1, val_x.shape[-1])).reshape(val_x.shape)
val_x2 = scaler2.transform(val_x2.reshape(-1, val_x2.shape[-1])).reshape(val_x2.shape)
#smooth time
val_x = time_sampler(val_x, timestep_size = 5)
val_x = savgol_smooth(val_x, window = 3, poly = 2)

del datas_tr, datas_val

########################################################

#run just with tabular data for now...
max_data_points = 1.5e6
datas_tr1 = datas_tr1.sample(int(max_data_points * 0.2))
datas_tr2 = datas_tr2.sample(int(max_data_points * 0.2))
datas_tr3 = datas_tr3.sample(int(max_data_points * 0.25))
datas_tr4 = datas_tr4.sample(int(max_data_points * 0.35))

#combine data into one dataframe
datas_tr = pd.concat([datas_tr1, datas_tr2, datas_tr3, datas_tr4])
#grab validation data
datas_val = datas_val.sample(200000)

def index_resample_tab(datax, datay):
    rus = RandomUnderSampler()
    rus.fit_resample(datax, datay)
    random.shuffle(rus.sample_indices_)
    datax = datax[rus.sample_indices_] 
    datay = datay[rus.sample_indices_] 
    return datax, datay

x_columns2 = ['Volume',
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
 'trnd_252', 'trnd_84', 'trnd_21']

tr_x2 = np.array(datas_tr[x_columns2])
tr_y = np.array(datas_tr['gain_10'])
tr_x2, tr_y = remove_arr_outliar3(tr_y, tr_x2, limit = 7)
tr_x2, tr_y = index_resample_tab(tr_x2, tr_y)
scaler2 = StandardScaler()
tr_x2 = scaler2.fit_transform(tr_x2)

val_x2 = np.array(datas_val[x_columns2])
val_y = np.array(datas_val['gain_10'])
val_x2, val_y = index_resample_tab(val_x2, val_y)
val_x2 = scaler2.transform(val_x2)


def show_hist(data, column):
    data = pd.DataFrame(data)
    data.iloc[:,column].hist(bins = 100)
    plt.show()

for i in range(21):
    col = i
    show_hist(tr_x2, col)
    show_hist(val_x2, col)


### Build Models ###

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)

#random forest
rfc = RandomForestClassifier(max_depth = 7)
rfc.fit(tr_x2, tr_y)
#xgboost
xgbc = xgb.XGBClassifier(tree_method="hist")
xgbc.fit(tr_x2, tr_y)
#MARS
mars = pyearth.Earth()
mars.fit(tr_x2, tr_y)
#logistic regression
lrc = LogisticRegression()
lrc.fit(tr_x2, tr_y)
#Dense ANN
dense = build_dense(tr_x2.shape[1], dense_ct = 14, dropout = 0.5,
                    lr = lr_schedule, dense_act = 'tanh')

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience= 10, restore_best_weights=True),
    tf.keras.callbacks.CSVLogger('..//dense.csv')
]

dense.fit(tr_x2, tr_y, epochs = 9, batch_size = 128,
             validation_data = (val_x2, val_y),
             callbacks = callbacks)


#Dual ANN-RNN ##############
dual_rnn = build_dual_rnn(tr_x.shape[1], tr_x.shape[2], tr_x2.shape[1],
                          rnn_ct = 10, dense_ct = 12, dropout = 0.5, 
                          lr = lr_schedule, rnn_act = 'tanh', dense_act = 'tanh')

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience= 5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('rnn.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('..//rnn.csv')
]
dual_rnn.fit([tr_x, tr_x2], tr_y, epochs = 9, batch_size = 128,
             validation_data = ([val_x, val_x2], val_y),
             callbacks = callbacks)

##### assess ONLY RF model #####
tr_pred_o = rfc.predict(tr_x2)
val_pred_o = rfc.predict(val_x2)

rfc_tr_raw_o = rfc.predict_proba(tr_x2)[:,1]
rfc_val_raw_o = rfc.predict_proba(val_x2)[:,1]

tr_prec_score_o = precision_score(tr_y, tr_pred_o)
tr_acc_score_o = accuracy_score(tr_y, tr_pred_o)
tr_auc_score_o = roc_auc_score(tr_y, rfc_tr_raw_o)
tr_confusion_o = confusion_matrix(tr_y, tr_pred_o)

val_prec_score_o = precision_score(val_y, val_pred_o)
val_acc_score_o = accuracy_score(val_y, val_pred_o)
val_auc_score_o = roc_auc_score(val_y, rfc_val_raw_o)
val_confusion_o = confusion_matrix(val_y, val_pred_o)



print('New Train accuracy: ' + str(np.round(tr_acc_score_n, 3)))
print('New Train precision: ' + str(np.round(tr_prec_score_n, 3)))
print('New Train auc: ' + str(np.round(tr_auc_score_n, 3)))
print(tr_confusion_n)

print('Old Train accuracy: ' + str(np.round(tr_acc_score_o, 3)))
print('Old Train precision: ' + str(np.round(tr_prec_score_o, 3)))
print('Old Train auc: ' + str(np.round(tr_auc_score_o, 3)))
print(tr_confusion_o)


print('New Val accuracy: ' + str(np.round(val_acc_score_n, 3)))
print('New Val precision: ' + str(np.round(val_prec_score_n, 3)))
print('New Val auc: ' + str(np.round(val_auc_score_n, 3)))
print(val_confusion_n)


print('Old Val accuracy: ' + str(np.round(val_acc_score_o, 3)))
print('Old Val precision: ' + str(np.round(val_prec_score_o, 3)))
print('Old Val auc: ' + str(np.round(val_auc_score_o, 3)))
print(val_confusion_o)


plt.scatter(datas_tr.beta_21, datas_tr.gain_10)
plt.scatter(datas_tr.beta_21[datas_tr.beta_252 > -0.1], datas_tr.beta_252[datas_tr.beta_252 > -0.1])

datas_tr.beta_21[((datas_tr.beta_21 > -100) & (datas_tr.gain_10 == 1))].describe()

datas.beta_252.describe()

cutoff = 0.1

datas['trnd_252'] = np.where(datas['beta_252'] > cutoff, 1, np.where(datas['beta_252'] < -cutoff, -1, 0 ))
datas['trnd_84'] = np.where(datas['beta_84'] > cutoff, 1, np.where(datas['beta_84'] < -cutoff, -1, 0 ))
datas['trnd_21'] = np.where(datas['beta_21'] > cutoff, 1, np.where(datas['beta_21'] < -cutoff, -1, 0 ))



#dat = datas.sample(5000)

print(datas['gain_10'][datas['trnd_21']== -1].mean())
print(datas['gain_10'][datas['trnd_84']== -1].mean())
print(datas['gain_10'][datas['trnd_252']== -1].mean())


sum(datas['trnd_21'] == 1)


datas['beta_21'][((datas.beta_21 > -0.3) & (datas.beta_21 < 0.3))].hist(bins = 100, density = False)
datas['beta_21'][((datas.beta_21 > -0.3) & (datas.beta_21 < 0.3) & (datas.gain_10 == 1))].hist(bins = 100, density = False)




plt.bar(x_columns2, rfc.feature_importances_)
plt.xticks(rotation = 90)
plt.title('Random Forest Feature Importances')
plt.show()



dat = datas.sample(150000)

x = dat[['trnd_252', 'trnd_84', 'trnd_21']]
y = dat['gain_10']

lrc = LogisticRegression()
lrc.fit(x,y)


lrc.coef_
###################

#get raw probability predictions from models
rfc_tr_raw = rfc.predict_proba(tr_x2)[:,1]
xgb_tr_raw = xgbc.predict_proba(tr_x2)[:,1]
mars_tr_raw = mars.predict(tr_x2)
dense_tr_raw = dense.predict(tr_x2)
#dual_tr_raw = dual_rnn.predict([tr_x, tr_x2])

#combine into a training array
tr_mod_ens = pd.DataFrame([rfc_tr_raw, xgb_tr_raw, mars_tr_raw, dense_tr_raw]).transpose()
tr_mod_ens = np.array(tr_mod_ens)
#build stacked Logistic regression 
lr_stk = LogisticRegression().fit(tr_mod_ens, tr_y.reshape(len(tr_y),))
####################


#use all models to predict on val data and store in df
rfc_val_raw = rfc.predict_proba(val_x2)[:,1]
xgb_val_raw = xgbc.predict_proba(val_x2)[:,1]
mars_val_raw = mars.predict(val_x2)
lr_val_raw = lrc.predict_proba(val_x2)[:,1]
dense_val_raw = dense.predict(val_x2)
#dual_val_raw = dual_rnn.predict([val_x, val_x2])

val_mod_ens = pd.DataFrame([rfc_val_raw, xgb_val_raw, mars_val_raw, lr_val_raw,
                            dense_val_raw]).transpose()
val_mod_ens = np.array(val_mod_ens)

#logist_val_raw = lr_stk.predict_proba(val_mod_ens)[:,1]
val_preds = pd.DataFrame(val_mod_ens, columns = ['rf', 'xgb', 'mars', 'lr', 'dense'])
#val_preds['logist'] = logist_val_raw
val_preds['dense'] = val_preds['dense'].apply(lambda x: x[0])

#assess accuracies of each model
precs = []
accs = []
aucs = []
for i in range(5):
    y_pred = np.where(val_preds.iloc[:,i] < 0.5, 0, 1)
    prec_score = precision_score(val_y, y_pred)
    acc_score = accuracy_score(val_y, y_pred)
    auc_score = roc_auc_score(val_y, val_preds.iloc[:,i])
    print(str(i) + 'accuracy: ' + str(np.round(acc_score, 3)))
    print(str(i) + 'precision: ' + str(np.round(prec_score, 3)))
    print(str(i) + 'auc: ' + str(np.round(auc_score, 3)))
    accs.append(acc_score)
    precs.append(prec_score)
    aucs.append(auc_score)
    
    
model_qual = pd.DataFrame([accs, precs, aucs]).transpose()
model_qual.columns = ['Accuracy', 'Precision', 'ROC-AUC']

model_qual.plot.bar()


#build ROC for each model
#AUC curves
fpr1 , tpr1 , thresholds = roc_curve(val_y , val_preds['rf'])
plt.plot(fpr1,tpr1, color = 'blue') 
fpr2 , tpr2 , thresholds = roc_curve(val_y , val_preds['xgb'])
plt.plot(fpr2,tpr2, color = 'green') 
fpr3 , tpr3 , thresholds = roc_curve(val_y , val_preds['mars'])
plt.plot(fpr3,tpr3, color = 'red') 
fpr4 , tpr4 , thresholds = roc_curve(val_y , val_preds['lr'])
plt.plot(fpr4,tpr4, color = 'pink') 
fpr5 , tpr5 , thresholds = roc_curve(val_y , val_preds['dense'])
plt.plot(fpr5,tpr5, color = 'purple') 
plt.axis([0,1,0,1]) 
plt.title('Model Comparision: ROC Curves')
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.text(0.5, 0.75, 'RF Accuracy: ' + str(np.round(accs[0] * 100, 1)) + '%')
plt.text(0.5, 0.7, 'XGB Accuracy: ' + str(np.round(accs[1] * 100, 1)) + '%')
plt.text(0.5, 0.65, 'MARS Accuracy: ' + str(np.round(accs[2] * 100, 1)) + '%')
plt.text(0.5, 0.6, 'LR Accuracy: ' + str(np.round(accs[3] * 100, 1)) + '%')
plt.text(0.5, 0.55, 'Dense ANN Accuracy: ' + str(np.round(accs[4] * 100, 1)) + '%')
plt.text(0.5, 0.35, 'RF Precision: ' + str(np.round(precs[0] * 100, 1)) + '%')
plt.text(0.5, 0.3, 'XGB Precision: ' + str(np.round(precs[1] * 100, 1)) + '%')
plt.text(0.5, 0.25, 'MARS Precision: ' + str(np.round(precs[2] * 100, 1)) + '%')
plt.text(0.5, 0.2, 'LR Precision: ' + str(np.round(precs[3] * 100, 1)) + '%')
plt.text(0.5, 0.15, 'Dense ANN Precision: ' + str(np.round(precs[4] * 100, 1)) + '%')
plt.show()
    

#get 50 Vote, 50 all, 80 vote, 80 all stats (acc and precision)
all_50 = []
all_80 = []
vote_50 = []
vote_80 = []
for i in tqdm(range(len(val_preds))):
    all_50.append(int(np.where(all(val_preds.iloc[i,:] >= 0.5), 1, 0)))
    all_80.append(int(np.where(all(val_preds.iloc[i,:] >= 0.8), 1, 0)))
    vote_50.append(int(np.where(np.mean(val_preds.iloc[i,:]) >= 0.5, 1, 0)))
    vote_80.append(int(np.where(np.mean(val_preds.iloc[i,:]) >= 0.8, 1, 0)))    

all_50_acc = accuracy_score(val_y, all_50)
all_50_prec = precision_score(val_y, all_50)
all_50_confus = confusion_matrix(val_y, all_50)

     
all_80_acc = accuracy_score(val_y, all_80)
all_80_prec = precision_score(val_y, all_80)
all_80_confus = confusion_matrix(val_y, all_80)

vote_50_acc = accuracy_score(val_y, vote_50)
vote_50_prec = precision_score(val_y, vote_50)
vote_50_auc = roc_auc_score(val_y, vote_50)
vote_50_confus = confusion_matrix(val_y, vote_50)

vote_80_acc = accuracy_score(val_y, vote_80)
vote_80_prec = precision_score(val_y, vote_80)
vote_80_confus = confusion_matrix(val_y, vote_80)


ax= plt.subplot()
sns.heatmap(all_50_confus, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('All Models Agree: 50% threshold \n Accuracy: ' + str(np.round(all_50_acc * 100, 1)) + '% \n Precision: ' + str(np.round(all_50_prec * 100, 1)) + '%'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.show()

ax= plt.subplot()
sns.heatmap(all_80_confus, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('All Models Agree: 80% threshold \n Accuracy: ' + str(np.round(all_80_acc * 100, 1)) + '% \n Precision: ' + str(np.round(all_80_prec * 100, 1)) + '%'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.show()

ax= plt.subplot()
sns.heatmap(vote_50_confus, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Mean Vote: 50% threshold \n Accuracy: ' + str(np.round(vote_50_acc * 100, 1)) + '% \n Precision: ' + str(np.round(vote_50_prec * 100, 1)) + '%'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.show()

ax= plt.subplot()
sns.heatmap(vote_80_confus, annot=True, fmt='g', ax=ax); 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Mean Vote: 80% threshold \n Accuracy: ' + str(np.round(vote_80_acc * 100, 1)) + '% \n Precision: ' + str(np.round(vote_80_prec * 100, 1)) + '%'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.show()
    



    
#save models
MODEL_PATH = 'C:\\Users\\q23853\\Desktop\\random_trader\\dual_rnn'
dual_rnn.save(MODEL_PATH, include_optimizer = False)
MODEL_PATH = 'C:\\Users\\q23853\\Desktop\\random_trader\\dense'
dense.save(MODEL_PATH, include_optimizer = False)

os.chdir('C:\\Users\\q23853\\Desktop\\random_trader')
joblib.dump(rfc, 'rf.joblib')
joblib.dump(xgbc, 'xgbc.joblib')
joblib.dump(mars, 'mars.joblib')
joblib.dump(lrc, 'lrc.joblib')
joblib.dump(scaler, 'timeseries_scaler.joblib')
joblib.dump(scaler2, 'tabular_scaler.joblib')



from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
tr_pca = pca.fit_transform(tr_x2)
tr_pca = np.concatenate([tr_pca, tr_y.reshape(len(tr_y), 1)], axis = 1)
tr_pca = tr_pca[tr_pca[:,0] < 100]
tr_pca = tr_pca[tr_pca[:,1] < 100]

plt.scatter(tr_pca[:,0], tr_pca[:,1], c = tr_pca[:,2], alpha = 0.1)


#pca = PCA(n_components = 2)
val_pca = pca.transform(val_x2)
val_pca = np.concatenate([val_pca, val_y.reshape(len(val_y), 1), 
                          np.array(all_50).reshape(len(val_y), 1),
                          np.array(all_80).reshape(len(val_y), 1),
                          np.array(vote_50).reshape(len(val_y), 1),
                          np.array(vote_80).reshape(len(val_y), 1)], axis = 1)
val_pca = val_pca[val_pca[:,0] < 25]
val_pca = val_pca[val_pca[:,1] < 25]

plt.scatter(val_pca[:,0], val_pca[:,1], c = val_pca[:,2], alpha = 0.1)
plt.title('True')
plt.show()

plt.scatter(val_pca[:,0], val_pca[:,1], c = val_pca[:,3], alpha = 0.1)
plt.title('All_50')
plt.show()


plt.scatter(val_pca[:,0], val_pca[:,1], c = val_pca[:,4], alpha = 0.1)
plt.title('All_80')
plt.show()

plt.scatter(val_pca[:,0], val_pca[:,1], c = val_pca[:,5], alpha = 0.1)
plt.title('vote_50')
plt.show()

plt.scatter(val_pca[:,0], val_pca[:,1], c = val_pca[:,6], alpha = 0.1)
plt.title('vote_80')
plt.show()


#%%
'''
Prepare stocks with their 6mo sell date and price for each day in the dataframe
and their 10% crossing threshold
'''
#%%
os.chdir(path2)

def sell_6mo(data):
    end_dates = []
    end_prices = []
    #find the price and date 126 trading days ahead
    for i in range(len(data) - 127):
        end_dates.append(data['Date'].iloc[i + 127])
        end_prices.append(data['Close'].iloc[i + 127])
    #create a mask for the end of the dataframe where no data is
    masks = [-1] * 127
    end_dates = end_dates + masks
    end_prices = end_prices + masks
    data['date_6mo'] = end_dates
    data['price_6mo'] = end_prices
    return data
    
def sell_over_value(data, value = 0.1):
    end_prices = []
    end_dates = []
    for i in range(len(data) - 127):
        threshold = data['Close'].iloc[i] * (1 + value)
        #subset the data to next 126 trading days
        sub_data = data.iloc[i: i + 127]
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
    masks = [-1] * 127
    end_dates = end_dates + masks
    end_prices = end_prices + masks
    data['date_10per'] = end_dates
    data['price_10per'] = end_prices
    return data

def update_selldata(data):
    data = sell_6mo(data)
    data = sell_over_value(data)
    return data
    

#open each stock dataframe 
datas = []
tickers = []
with os.scandir(path) as it:
    for entry in it:
        if entry.name.endswith(".csv"):
            #open the data
            datas.append(pd.read_csv(entry))
            name = entry.name.split('.')[0]
            tickers.append(name)
datas2 = []
for i in tqdm(range(len(datas))):
    data = datas[i]
    datas2.append(update_selldata(data))
    
os.chdir(path2)
for i in tqdm(range(len(tickers))):
    file_name = str(str(tickers[i]) + '.csv')
    df = datas2[i]
    df.to_csv(file_name, sep = ',', index = False, encoding = 'utf-8')
    
#%%
'''
Test out both bare random trader and original 10% gain RT

YFinance data changes overtime, this data is substatially different than the original

I failed to account for temporal changes overtime, hopeing boostrapping would alleviate them
but there are major gain/loss potentials overtime, making the histograms more indicitive of
time than pure returns.

Above does not invalidate older results, but it does warrent caution and explicit comparisons 
overtime
'''
#%%
os.chdir(path2)
def get_tickers_data(path, colz = ['Ticker', 'Date','Close', 'date_6mo', 'price_6mo', 'date_10per', 'price_10per'] ):
    tickers = []
    datas = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".csv"):
                name = entry.name.split('.')[0]
                tickers.append(name)
                temp_data = pd.read_csv(entry)
                temp_data['Ticker'] = [name] * len(temp_data)
                temp_data = temp_data[colz]
                datas.append(temp_data)
    return tickers, datas

colz_train = ['Volume', 'bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84',
       'bb_low84', 'bb_center252', 'bb_upp252', 'bb_low252', 'minmax21',
       'minmax84', 'minmax252', 'atr_21', 'atr_84', 'atr_252', 'rsi_21',
       'rsi_84', 'rsi_252', 'vol_21', 'vol_84', 'vol_252']

#grab avaiable data and tickers
colz_for_pred = ['Ticker', 'Date','Close', 'Volume', 'date_6mo', 'price_6mo', 'date_10per', 'price_10per', 'gain_10'] + colz_train
valid_tickers, datas = get_tickers_data(path2, colz_for_pred)
datas = pd.concat(datas)
datas.dropna(inplace = True) #remove na's
#normalize columns to closing price that need normalization
colz_norm_name = ['bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84', 
                  'bb_low84', 'bb_center252',  'bb_upp252', 'bb_low252', 'atr_21',
                  'atr_84', 'atr_252'] 
datas_col = datas.columns
for i in range(len(colz_norm_name)):
    col = colz_norm_name[i]
    datas[col] = datas[col]/datas['Close']
#convert RSI to 0-1 scale
datas['rsi_21'] = datas['rsi_21']/100
datas['rsi_84'] = datas['rsi_84']/100
datas['ris_252'] = datas['ris_252']/100
datas['rsi_252'] = datas['ris_252']
datas.drop(['ris_252'], inplace = True, axis = 1) #remove the typo
datas.reset_index(inplace = True, drop = True)


#### TIMESERIES SECTION ####
#get the timeseries for seelected columns
time_series_col = ['Close', 'Volume']
num_trd_days = 104

def get_rnn_data_fortrade(data, num_trd_days):
    rnn_data = []
    for i in range(num_trd_days, len(data)):
        rnn_data.append(data[['Close', 'Volume']].iloc[i - num_trd_days:i + 1,:])
        
    z = list([np.nan] * num_trd_days) + rnn_data
    data['rnn_data'] = z
    return data

#get the RNN data
for i in tqdm(range(len(datas))):
    datas[i] = datas[i][datas[i]['Date'] > '2021-12-01']
    datas[i] = get_rnn_data_fortrade(datas[i], num_trd_days)
    
#standard scaler the relevant columns for the timeseries data\
time_scale = joblib.load('timeseries_scaler.joblib')
#get the data into a vectorizable form
rnn_data = []
for i in range(len(datas)):
    rnn_data.append(datas['rnn_data'].iloc[0])
rnn_data = np.array(rnn_data)
#apply scaler
rnn_data = time_scale.transform(rnn_data.reshape(-1, rnn_data.shape[-1])).reshape(rnn_data.shape)
#apply sampling
rnn_data = time_sampler(rnn_data, timestep_size = 5)
#apply smoothing
rnn_data = savgol_smooth(rnn_data, window = 3, poly = 2)
###########################

#subset to Post june 2022 and no stocks under $2 and over $100
datas = datas[datas['Date'] > '2021-01-01']
datas = datas[((datas['Close'] < 100) & (datas['Close'] > 2))]
#get the availble trading days
trading_days = list(set(list(datas['Date'])))
trading_days.sort()
max_date_ind = trading_days.index('2023-01-26') 

#standard scaler the relevant columns for tablular data
os.chdir(path1)
#grab tabular data
tab_data = np.array(datas[colz_train].iloc[:,1:])
#rescale
tab_scale = joblib.load('tabular_scaler.joblib')
#tab_scale = scaler2
tab_data = tab_scale.transform(tab_data)

#instantiate models
#MODEL_PATH = 'C:\\Users\\q23853\\Desktop\\random_trader\\dual_rnn'
#dual = tf.keras.models.load_model(MODEL_PATH)

#### Cheat! Use trained models 
lrc, rfc, xgbc, mars, tab_scaler, dense, valid_tickers = get_models(lrc_path, rf_path, xgb_path, mars_path, scaler_path, dense_path, tickers_path)


#make predictions
rfc_prd = rfc.predict_proba(tab_data)[:,1]
xgbc_prd = xgbc.predict_proba(tab_data)[:,1]
mars_prd = mars.predict(tab_data)
lrc_prd = lrc.predict_proba(tab_data)[:,1]
dense_prd = dense.predict(tab_data)
#optional: sigmoid bounding of mars between 0-1
bound_mars = 1/((1 + np.exp(-mars_prd)))

#dual_prd = dual.predict([rnn_data, tab_data])

pred_frame = pd.DataFrame([rfc_prd, xgbc_prd, mars_prd, lrc_prd, dense_prd]).transpose()
pred_frame.columns = ['rf', 'xgb', 'mars', 'lrc', 'dense']
pred_frame['dense'] = pred_frame['dense'].apply(lambda x: x[0])

pred_corr = pred_frame.corr()

#check base accuracies
#dual_pr = np.where((pred_frame['dual']) >= 0.5, 1, 0)
dense_pr = np.where((pred_frame['dense']) >= 0.5, 1, 0)
rf_pr = np.where((pred_frame['rf']) >= 0.5, 1, 0)
xgb_pr = np.where((pred_frame['xgb']) >= 0.5, 1, 0)
mars_pr = np.where((pred_frame['mars']) >= 0.5, 1, 0)
lrc_pr = np.where((pred_frame['lrc']) >= 0.5, 1, 0)

y = np.array(datas['gain_10'])
y.mean()

mars_acc = accuracy_score(y, mars_pr)
dense_acc = accuracy_score(y, dense_pr)
rf_acc = accuracy_score(y, rf_pr)
xgb_acc = accuracy_score(y, xgb_pr)
lrc_acc = accuracy_score(y, xgb_pr)
print('mars acc: ' + str(np.round(mars_acc, 3)))
print('dense acc: ' + str(np.round(dense_acc, 3)))
print('rfc acc: ' + str(np.round(rf_acc, 3)))
print('xgb acc: ' + str(np.round(xgb_acc, 3)))
print('lrc acc: ' + str(np.round(lrc_acc, 3)))

#prepare voting stragegies 
all_50 = np.where((pred_frame[['rf', 'xgb', 'mars', 'lrc', 'dense']]>=0.5).all(axis=1), 1, 0)
all_75 = np.where((pred_frame[['rf', 'xgb', 'mars', 'lrc', 'dense']]>=0.75).all(axis=1), 1, 0)
all_80 = np.where((pred_frame[['rf', 'xgb', 'mars', 'lrc', 'dense']]>=0.8).all(axis=1), 1, 0)
pred_rowmeans = pred_frame.mean(axis = 1)
vote_50 = np.where(pred_rowmeans >=0.5, 1, 0)
vote_75 = np.where(pred_rowmeans >=0.75, 1, 0)
vote_80 = np.where(pred_rowmeans >=0.8, 1, 0)
#add predictions to dataframe for trading
datas['all_50'] = all_50
datas['all_75'] = all_75
datas['all_80'] = all_80
datas['vote_50'] = vote_50
datas['vote_75'] = vote_75
datas['vote_80'] = vote_80
#set the index as a datetime index for speed
datas.reset_index(inplace = True, drop = True)
datas.index = pd.to_datetime(datas.Date)



def get_data_random_date(data, max_date_ind):
    bdate_ind = random.randint(0,max_date_ind)
    bdate = trading_days[bdate_ind]
    bdate = bdate.replace('-', '')
    temp_data = data.query(f"index == {bdate}")
    return temp_data

def build_portfolio(bank, number, trading_dates, data, max_date_ind,
                    buy_date = None):
    #pick a buying date
    if buy_date == None:
        counter = 0
        temp_data = get_data_random_date(data, max_date_ind)
        while ((counter < 25) & (len(temp_data) < number)): #keep trying dates until a day has enough data
            temp_data = get_data_random_date(data, max_date_ind)
            counter += 1
            #print(counter)            
    else:
        bdate = buy_date
        bdate = bdate.replace('-', '')
        #subset the data to that date
        temp_data = data.query(f"index == {bdate}")
    #drop stocks over $100 and under $2 from the purchase options - done already
    # temp_data = temp_data[((temp_data['Close'] < 100) & (temp_data['Close'] > 2))]
    #sample to the desired number of stocks
    if len(temp_data) < number:
        number = len(temp_data)
    temp_data = temp_data.sample(number)
    #evenly allocate funds from the bank to each purchased stock
    allocation = bank/number
    temp_data['share_num'] = np.floor(allocation/temp_data['Close'])
    temp_data['cost'] = temp_data['share_num'] * temp_data['Close']
    #update the bank
    total_cost = temp_data['cost'].sum()
    bank -= total_cost
    return temp_data, bank
    


def random_trader(bank, number, trading_days, data, trade_10per, max_date_ind):
    #build a portfolio
    portfolio, bank = build_portfolio(bank = bank, number = number, 
                                      trading_dates = trading_days, data = data, 
                                      max_date_ind = max_date_ind)
    if trade_10per == True:
        #subset those that dont go up 10% into a holding frame for the end
        hold_df = portfolio[portfolio['price_10per'] == -1]
        #the rest in an active df
        active_df = portfolio[portfolio['price_10per'] != -1]
        #save date_6mo as a stopping marker and starting dates
        start_date = portfolio['Date'].iloc[0]
        begin_date = start_date.replace('-', '')
        stop_date = portfolio['date_6mo'].iloc[0]
        end_date = stop_date.replace('-', '')
        #subset the data to only a 6mo dataframe to work with
        ref_df = data.query(f"index >= {begin_date} and index <= {end_date}")
        #trade the active portfolio within the 6mo
        while len(active_df) > 0:
            #grab a stock
            stock = active_df.iloc[0]
            #find the sell date
            sell_date = str(stock['date_10per'])
            #check if it is less than the stop date
            if ((sell_date < stop_date) & (sell_date != str(-1)) & (sell_date < ref_df['Date'].iloc[-1])):
                #adjust the bank
                bank += stock['price_10per'] * stock['share_num']
                #replace the row with a new stock
                new_stock, bank = build_portfolio(bank = bank, number = 1, 
                                                  trading_dates = trading_days, 
                                                  data = ref_df, buy_date = sell_date,
                                                  max_date_ind = max_date_ind)
                active_df.iloc[0] = new_stock.iloc[0, :]
            else:#if its not... 
                #lookup the stock price at the stop date
                ticker = stock['Ticker']
                if isinstance(ticker, str):
                    ticker = ticker
                else: 
                    ticker = ticker.values[0]
                #temp_data = ref_df.query(f"index == {end_date}")
                temp_data = ref_df[ref_df['Ticker'] == ticker]
                temp_data = pd.DataFrame(temp_data.iloc[-1]).transpose() #grab closest to the end date as possible
                #replace the 6mo price with the close price for easy final selling
                temp_data['price_6mo'] = temp_data['Close']
                #add in share number
                temp_data['share_num'] = stock['share_num']
                temp_data['cost'] = stock['cost']
                #add this row to the hold df  
                hold_df = pd.concat([hold_df, temp_data])
                #drop this row from the active dataframe
                active_df = active_df[active_df['Ticker'] != ticker]
        return hold_df, bank
    else:
        return portfolio, bank


#specify size of portfolio
number = 15
start_bank = 10000
sim_number = 750

#get the return, start date, and the portfolio dataframe
rets = []
portfolios = []
dates = []
aboves = []
for i in tqdm(range(sim_number)):
    df, bank = random_trader(bank= start_bank, number= 15, trading_days= trading_days, 
                             data = datas, trade_10per = False,
                             max_date_ind = max_date_ind)
    returns = (sum(df['share_num'] * df['price_6mo']) - start_bank)/start_bank
    if ((returns >= 1.25) & (len(df) == 15)): #count the number that are out of range
        aboves.append(returns)
    if ((returns < 1.25) & (len(df) == 15)):
        rets.append(returns)
        portfolios.append(df)
        dates.append(df['Date'].iloc[0])
        
    
#calculate the return through time
base_ret_time = pd.DataFrame([rets, dates]).transpose()
base_ret_time.columns = ['Return', 'Date']
base_ret_time['dtime'] = pd.to_datetime(base_ret_time['Date'])
base_ret_timegrp = base_ret_time.Return.groupby(by=[base_ret_time.dtime.dt.year, base_ret_time.dtime.dt.month]).mean()

#compare to VTI
VTI = yf.download('VTI', start= '2013-01-01', end= '2023-08-01',  progress=False)
forward127 = []
for i in range(len(VTI) - 127):
    forward127.append(VTI['Close'].iloc[i + 127])
forward127 = forward127 + [-1] * 127
VTI['6mo'] = forward127

VTI_ret =(VTI['6mo'] - VTI['Close'])/VTI['Close']
VTI_ret = VTI_ret[VTI_ret > -0.99]
vti_timegrp = VTI_ret.groupby(by=[VTI_ret.index.year, VTI_ret.index.month]).mean()

#run RT again but with the 10% sell thresh 
rets_mod = []
portfolios_mod  = []
dates_mod  = []
aboves = []
for i in tqdm(range(sim_number)):
    df, bank = random_trader(bank= start_bank, number= 15, trading_days= trading_days, 
                             data = datas, trade_10per = True,
                             max_date_ind = max_date_ind)
    returns = (sum(df['share_num'] * df['price_6mo']) - start_bank)/start_bank
    if returns >= 1.25: #count the number that are out of range
        aboves.append(returns)
    if ((returns < 1.25) & (len(df) == 15)):
        rets_mod.append(returns)
        portfolios_mod .append(df)
        dates_mod.append(df['Date'].iloc[0])
    
#calculate the return through time
base10_ret_time_mod  = pd.DataFrame([rets_mod , dates_mod ]).transpose()
base10_ret_time_mod.columns = ['Return', 'Date']
base10_ret_time_mod['dtime'] = pd.to_datetime(base10_ret_time_mod['Date'])
base10_ret_timegrp_mod = base10_ret_time_mod.Return.groupby(by=[base10_ret_time_mod.dtime.dt.year, base10_ret_time_mod.dtime.dt.month]).mean()


#Model with all 50
datas_use = datas[datas['all_50'] == 1]

rets_mod = []
portfolios_mod  = []
dates_mod  = []
aboves_a50 = []
for i in tqdm(range(sim_number)):
    df, bank = random_trader(bank= start_bank, number= 15, trading_days= trading_days, 
                             data = datas_use, trade_10per = True,
                             max_date_ind = max_date_ind)
    returns = (sum(df['share_num'] * df['price_6mo']) - start_bank)/start_bank
    if ((returns >= 1.25) & (len(df) == 15)): #count the number that are out of range
        aboves_a50.append(returns)
    if ((returns < 1.25) & (len(df) == 15)):
        rets_mod.append(returns)
        portfolios_mod .append(df)
        dates_mod.append(df['Date'].iloc[0])
    
#calculate the return through time
all50_ret_time_mod  = pd.DataFrame([rets_mod , dates_mod ]).transpose()
all50_ret_time_mod.columns = ['Return', 'Date']
all50_ret_time_mod['dtime'] = pd.to_datetime(all50_ret_time_mod['Date'])
all50_ret_timegrp_mod = all50_ret_time_mod.Return.groupby(by=[all50_ret_time_mod.dtime.dt.year, all50_ret_time_mod.dtime.dt.month]).mean()


#model with all 75
datas_use = datas[datas['all_75'] == 1]

rets_mod = []
portfolios_mod  = []
dates_mod  = []
aboves_a80 = []
for i in tqdm(range(sim_number)):
    df, bank = random_trader(bank= start_bank, number= 15, trading_days= trading_days, 
                             data = datas_use, trade_10per = True,
                             max_date_ind = max_date_ind)
    returns = (sum(df['share_num'] * df['price_6mo']) - start_bank)/start_bank
    if ((returns >= 1.25) & (len(df) == 15)): #count the number that are out of range
        aboves_a80.append(returns)
    if ((returns < 1.25) & (len(df) == 15)):
        rets_mod.append(returns)
        portfolios_mod .append(df)
        dates_mod.append(df['Date'].iloc[0])
    
#calculate the return through time
all75_ret_time_mod  = pd.DataFrame([rets_mod , dates_mod ]).transpose()
all75_ret_time_mod.columns = ['Return', 'Date']
all75_ret_time_mod['dtime'] = pd.to_datetime(all75_ret_time_mod['Date'])
all75_ret_timegrp_mod = all75_ret_time_mod.Return.groupby(by=[all75_ret_time_mod.dtime.dt.year, all75_ret_time_mod.dtime.dt.month]).mean()


#model with vote 50
datas_use = datas[datas['vote_50'] == 1]

rets_mod = []
portfolios_mod  = []
dates_mod  = []
aboves_v50 = []
for i in tqdm(range(sim_number)):
    df, bank = random_trader(bank= start_bank, number= 15, trading_days= trading_days, 
                             data = datas_use, trade_10per = True,
                             max_date_ind = max_date_ind)
    returns = (sum(df['share_num'] * df['price_6mo']) - start_bank)/start_bank
    if ((returns >= 1.25) & (len(df) == 15)): #count the number that are out of range
        aboves_v50.append(returns)
    if ((returns < 1.25) & (len(df) == 15)):
        rets_mod.append(returns)
        portfolios_mod .append(df)
        dates_mod.append(df['Date'].iloc[0])
    
#calculate the return through time
vote50_ret_time_mod  = pd.DataFrame([rets_mod , dates_mod ]).transpose()
vote50_ret_time_mod.columns = ['Return', 'Date']
vote50_ret_time_mod['dtime'] = pd.to_datetime(vote50_ret_time_mod['Date'])
vote50_ret_timegrp_mod = vote50_ret_time_mod.Return.groupby(by=[vote50_ret_time_mod.dtime.dt.year, vote50_ret_time_mod.dtime.dt.month]).mean()

#model with vote 75
datas_use = datas[datas['vote_75'] == 1]

rets_mod = []
portfolios_mod  = []
dates_mod  = []
aboves_v80 = []
for i in tqdm(range(3000)):
    df, bank = random_trader(bank= start_bank, number= 15, trading_days= trading_days, 
                             data = datas_use, trade_10per = True,
                             max_date_ind = max_date_ind)
    returns = (sum(df['share_num'] * df['price_6mo']) - start_bank)/start_bank
    if ((returns >= 1.25) & (len(df) == 15)): #count the number that are out of range
        aboves_v80.append(returns)
    if ((returns < 1.25) & (len(df) == 15)):
        rets_mod.append(returns)
        portfolios_mod .append(df)
        dates_mod.append(df['Date'].iloc[0])
    
#calculate the return through time
vote75_ret_time_mod  = pd.DataFrame([rets_mod , dates_mod ]).transpose()
vote75_ret_time_mod.columns = ['Return', 'Date']
vote75_ret_time_mod['dtime'] = pd.to_datetime(vote75_ret_time_mod['Date'])
vote75_ret_timegrp_mod = vote75_ret_time_mod.Return.groupby(by=[vote75_ret_time_mod.dtime.dt.year, vote75_ret_time_mod.dtime.dt.month]).mean()

#### VISUALIZE BETTER ####
'''
VTI = Black
Base random trader = Cyan
Base10 Random Trader = Blue

Vote50 = Olive
Vote80 = Green

All50 = Pink
Vote80 = Purple
'''
#RETURNS OVERTIME
date_range = ['Jan 2021', 'Feb 2021', 'Mar 2021', 'Apr 2021', 'May 2021', 'Jun 2021', 
              'Jul 2021', 'Aug 2021', 'Sept 2021', 'Oct 2021', 'Nov 2021', 'Dec 2021',
              'Jan 2022', 'Feb 2022', 'Mar 2022', 'Apr 2022', 'May 2022', 'Jun 2022',
              'Jul 2022', 'Aug 2022', 'Sept 2022', 'Oct 2022', 'Nov 2022', 'Dec 2022', 'Jan 2023']
fig = plt.figure()
ax = fig.add_subplot()
line1, = ax.plot(date_range, np.array(vti_timegrp), color = 'Black')
line2, = ax.plot(date_range, np.array(base_ret_timegrp), color = 'Cyan')
line3, = ax.plot(date_range, np.array(base10_ret_timegrp_mod), color = 'Blue')
line4, = ax.plot(date_range, np.array(vote50_ret_timegrp_mod), color = 'Olive')
line5, = ax.plot(date_range, np.array(vote75_ret_timegrp_mod), color = 'Green')
line6, = ax.plot(date_range, np.array(all50_ret_timegrp_mod), color = 'Pink')
line7, = ax.plot(date_range, np.array(all75_ret_timegrp_mod), color = 'Purple')
plt.xticks(rotation = 90)
plt.title('Returns Overtime')
plt.ylabel('Returns')
plt.xlabel('Starting Investment Month')
handles, labels = ax.get_legend_handles_labels()
ax.legend([line1, line2, line3, line4, line5, line6, line7],
          ['VTI', 'Entirely RT', 'Base RT', 'Vote 50', 'Vote 80', 'All 50', 'All 80'],
          bbox_to_anchor=(0.8, 0.5, 0.5, 0.5))
plt.show()

#HISTOGRAMS
def plot_histogram(rets, color, text, bins = 50):
    rets.hist(bins = 50, color = color)
    plt.title(text + ' Returns Histogram \n' + str('Mean: ' + str(np.round(rets.mean() * 100, 1)) + '% \n') 
              + str('Median: ' + str(np.round(rets.median() * 100, 1)) + '% \n') + 
              str('NegChance: ' + str(np.round(sum(rets < 0)/len(rets) * 100, 1)) + '%'))
    plt.ylabel('Counts')
    plt.xlabel('Returns')
    plt.grid(False)
    plt.show()
    
plot_histogram(VTI_ret, color = 'Grey', text = 'VTI', bins = 50)
plot_histogram(base_ret_time['Return'], color = 'Cyan', text = 'Entirely RT', bins = 50)
plot_histogram(base10_ret_time_mod['Return'], color = 'Blue', text = 'Base RT', bins = 50)
plot_histogram(vote50_ret_time_mod['Return'], color = 'Olive', text = 'Vote 50 RT', bins = 50)
plot_histogram(vote75_ret_time_mod['Return'], color = 'Green', text = 'Vote 75 RT', bins = 50)
plot_histogram(all50_ret_time_mod['Return'], color = 'Pink', text = 'All 50 RT', bins = 50)
plot_histogram(all80_ret_time_mod['Return'], color = 'Purple', text = 'All 75 RT', bins = 50)

#######################
# Plot out VTI, S&P, AAPL, TSLA as buy/no buy predictions
DAL = datas[datas['Ticker'] == 'DAL']
DAL_c = DAL['Close']
GME = datas[datas['Ticker'] == 'GME']
GME_c = GME['Close']
#rescale
DAL = np.array(DAL[colz_train])
DALs = tab_scale.transform(DAL)
GME = np.array(GME[colz_train])
GMEs = tab_scale.transform(GME)

#make predictions
def ens_pred(tab_data):
    rfc_prd = rfc.predict_proba(tab_data)[:,1]
    xgbc_prd = xgbc.predict_proba(tab_data)[:,1]
    mars_prd = mars.predict(tab_data)
    lrc_prd = lrc.predict_proba(tab_data)[:,1]
    dense_prd = dense.predict(tab_data)
    #place into prediction df
    pred_frame = pd.DataFrame([rfc_prd, xgbc_prd, mars_prd, lrc_prd, dense_prd]).transpose()
    pred_frame.columns = ['rf', 'xgb', 'mars', 'lrc', 'dense']
    pred_frame['dense'] = pred_frame['dense'].apply(lambda x: x[0])
    #run voting strategies
    pred_frame['all_50'] = np.where((pred_frame[['rf', 'xgb', 'mars', 'lrc', 'dense']]>=0.5).all(axis=1), 1, 0)
    pred_frame['all_70'] = np.where((pred_frame[['rf', 'xgb', 'mars', 'lrc', 'dense']]>=0.7).all(axis=1), 1, 0)
    pred_frame['all_80'] = np.where((pred_frame[['rf', 'xgb', 'mars', 'lrc', 'dense']]>=0.8).all(axis=1), 1, 0)
    pred_rowmeans = pred_frame.mean(axis = 1)
    pred_frame['vote_50'] = np.where(pred_rowmeans >=0.5, 1, 0)
    pred_frame['vote_70'] = np.where(pred_rowmeans >=0.7, 1, 0)
    pred_frame['vote_80'] = np.where(pred_rowmeans >=0.8, 1, 0)
    return pred_frame

dal_pred = ens_pred(DALs)
amc_pred = ens_pred(GMEs)

#add specific colors for vote50, all80
dal_pred['v50_colors'] = np.where(dal_pred['vote_50'] == 1, 1, 0)
dal_pred['all80_colors'] = np.where(dal_pred['all_80'] == 1, 1, 0)
dal_pred['Close'] = DAL_c.values
dal_pred['Date'] = DAL_c.index

amc_pred['v50_colors'] = np.where(amc_pred['vote_50'] == 1, 1, 0)
amc_pred['all80_colors'] = np.where(amc_pred['all_80'] == 1, 1, 0)
amc_pred['Close'] = AMC_c[3:].values
amc_pred['Date'] = AMC_c.index[3:]

#plot out Vote 50 and All 80 as green for buy red for don't buy


# Convert the 'Date' column to datetime format
def green_red_lineplot(data, title, colors_column):
    day_list = list(range(len(data)))
    xy = np.column_stack((day_list, data['Close']))
    fig, ax = plt.subplots()
    for start, stop, color_code in zip(xy[:-1], xy[1:], data[colors_column]):
        x, y = zip(start, stop)
        color = 'green' if color_code == 1 else 'red'
        ax.plot(x, y, color=color, linestyle='-')
    ax.set_xlabel('Days from Jan 1, 2021')
    ax.set_ylabel('Close Price')
    ax.set_title(title)
    plt.show()

green_red_lineplot(dal_pred, 'Delta Airlines Vote 50%', 'v50_colors')
green_red_lineplot(dal_pred, 'Delta Airlines All 80%', 'all80_colors')
green_red_lineplot(amc_pred, 'AMC Vote 50%', 'v50_colors')
green_red_lineplot(amc_pred, 'AMC 1/1/2023 - 1/26/2023', 'all80_colors')


#plot out the technical indicators
for i in range(len(colz_norm_name)):
    col = colz_norm_name[i]
    DAL[col] = DAL[col] * DAL['Close']


plt.plot(DAL['Close'], color = 'Black')
plt.plot(DAL['bb_upp84'], color = 'Blue', alpha = 0.7)
plt.plot(DAL['bb_low84'], color = 'Blue', alpha = 0.7)
plt.plot(DAL['bb_center84'], color = 'olive', alpha = 0.7)
plt.xticks(rotation = 90)
plt.title('Delta Airlines 84 day Bollinger Bands and Moving Average')

plt.plot(DAL['atr_84'], color = 'Red', alpha = 0.7)
plt.plot(DAL['rsi_84'], color = 'green', alpha = 0.7)
plt.plot(DAL['minmax84'], color = 'purple', alpha = 0.7)
plt.xticks(rotation = 90)




def rma(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1 / period).mean()

def atr3(df, period = 14) -> pd.Series:
    high, low, prev_close = df['High'], df['Low'], df['Close'].shift()
    tr_all = [high - low, high - prev_close, low - prev_close]
    tr_all = [tr.abs() for tr in tr_all]
    tr = pd.concat(tr_all, axis=1).max(axis=1)
    atr_ = rma(tr, period)
    return atr_



