# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 09:33:28 2023

@author: q23853
"""


import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
#import statsmodels.formula.api as smf
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Concatenate, Bidirectional, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Model
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import pyearth
from sklearn.decomposition import PCA
from scipy.stats import norm



path1 = 'C:\\Users\\q23853\\Desktop\\random_trader'
path2 = 'C:\\Users\\q23853\\Desktop\\random_trader\\processed_stocks2'
path3 = 'C:\\Users\\q23853\\Desktop\\random_trader\\production_models'
os.chdir(path3)
#%% FUNCTIONS

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

def index_resample_tab(datax, datay):
    rus = RandomUnderSampler()
    rus.fit_resample(datax, datay)
    random.shuffle(rus.sample_indices_)
    datax = datax[rus.sample_indices_] 
    datay = datay[rus.sample_indices_] 
    return datax, datay

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


#%%

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
datas['rsi_252'] = datas['ris_252']/100
datas.drop(['ris_252'], inplace = True, axis = 1) #remove the typo
datas.reset_index(inplace = True, drop = True)


x_columns = ['bb_center21', #'Open','High','Low','Close','Adj Close',
                    'bb_upp21','bb_low21','bb_center84','bb_upp84','bb_low84',
                    'bb_center252','bb_upp252','bb_low252','minmax21','minmax84',
                    'minmax252','atr_21','atr_84','atr_252','rsi_21','rsi_84',
                    'rsi_252','vol_21','vol_84','vol_252']

y_columns = ['gain_10']


datas = datas[(datas['Close'] > 2) & (datas['Close'] < 75)]
#split data into 4 temporal quarters
datas_tr1 = datas[datas['Date'] <= '2016-04-08']
datas_tr2 = datas[((datas['Date'] > '2016-04-08') & (datas['Date'] <= '2018-07-13'))]
datas_tr3 = datas[((datas['Date'] > '2018-07-13') & (datas['Date'] <= '2020-10-16'))]
datas_tr4 = datas[(datas['Date'] > '2020-10-16')]

#sample the data
max_data_points = 2e6
datas_tr1 = datas_tr1.sample(int(max_data_points * 0.2))
datas_tr2 = datas_tr2.sample(int(max_data_points * 0.2))
datas_tr3 = datas_tr3.sample(int(max_data_points * 0.25))
datas_tr4 = datas_tr4.sample(int(max_data_points * 0.35))

#combine data into one dataframe
datas_tr = pd.concat([datas_tr1, datas_tr2, datas_tr3, datas_tr4])

#convert to arrays
x_train = np.array(datas_tr[x_columns])
y_train = np.array(datas_tr['gain_10'])
#remove extreme outliers
x_train, y_train = remove_arr_outliar3(y_train, x_train, limit = 4.5)
#rebalance the data
x_train, y_train = index_resample_tab(x_train, y_train) #1.428Mil datapoints remain
#standardize
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

#BUILD MODELS

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

#save the models
MODEL_PATH = 'C:\\Users\\q23853\\Desktop\\random_trader\\production_models\\dense'
dense.save(MODEL_PATH, include_optimizer = False)

os.chdir('C:\\Users\\q23853\\Desktop\\random_trader\\production_models')
joblib.dump(rfc, 'rf.joblib')
joblib.dump(xgbc, 'xgbc.joblib')
joblib.dump(mars, 'mars.joblib')
joblib.dump(lrc, 'lrc.joblib')
joblib.dump(scaler, 'timeseries_scaler.joblib')


#model predictions
rfc_pred = rfc.predict(x_train)
xgb_pred = xgbc.predict(x_train)
mars_pred = mars.predict(x_train)
mars_pred = np.where(mars_pred >= 0.5, 1, 0)
lr_pred = lrc.predict(x_train)
dense_pred = dense.predict(x_train)
dense_pred = np.where(dense_pred >= 0.5, 1, 0)

#model outputs (Mars and LR)
print(mars.summary())

lrc_coef = lrc.coef_.reshape(-1,1)
lrc_p = logit_pvalue(lrc, x_train)
lrc_p = lrc_p[1:].reshape(-1,1)
lrc_logodds = np.exp(lrc_coef)

lrc_summary = np.concatenate([lrc_coef, lrc_logodds, lrc_p], axis = 1)
lrc_summary = pd.DataFrame(lrc_summary, columns = ['coef', 'log_odds', 'p-val'])
lrc_summary['var'] = x_columns


#visualize the decision boundaries
#PCA
pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x_train)
x_pca = np.concatenate([x_pca, y_train.reshape(len(y_train), 1)], axis = 1)
x_pca = np.concatenate([x_pca, rfc_pred.reshape(len(rfc_pred), 1)], axis = 1)
x_pca = np.concatenate([x_pca, xgb_pred.reshape(len(xgb_pred), 1)], axis = 1)
x_pca = np.concatenate([x_pca, mars_pred.reshape(len(mars_pred), 1)], axis = 1)
x_pca = np.concatenate([x_pca, lr_pred.reshape(len(lr_pred), 1)], axis = 1)
x_pca = np.concatenate([x_pca, dense_pred.reshape(len(dense_pred), 1)], axis = 1)

#x_pca = x_pca[x_pca[:,0] < 100]
#x_pca = x_pca[x_pca[:,1] < 100]

plt.scatter(x_pca[:,0], x_pca[:,1], c = x_pca[:,2], alpha = 0.1)
plt.title('PCA with true labels')
plt.xlabel('1st Componet')
plt.ylabel('2nd Componet')
plt.show()

plt.scatter(x_pca[:,0], x_pca[:,1], c = x_pca[:,3], alpha = 0.1)
plt.title('PCA with RF labels')
plt.xlabel('1st Componet')
plt.ylabel('2nd Componet')
plt.show()

plt.scatter(x_pca[:,0], x_pca[:,1], c = x_pca[:,4], alpha = 0.1)
plt.title('PCA with XGB labels')
plt.xlabel('1st Componet')
plt.ylabel('2nd Componet')
plt.show()

plt.scatter(x_pca[:,0], x_pca[:,1], c = x_pca[:,5], alpha = 0.1)
plt.title('PCA with MARS labels')
plt.xlabel('1st Componet')
plt.ylabel('2nd Componet')
plt.legend()
plt.show()

plt.scatter(x_pca[:,0], x_pca[:,1], c = x_pca[:,6], alpha = 0.1)
plt.title('PCA with LR labels')
plt.xlabel('1st Componet')
plt.ylabel('2nd Componet')
plt.show()

plt.scatter(x_pca[:,0], x_pca[:,1], c = x_pca[:,7], alpha = 0.1)
plt.title('PCA with FF-ANN labels')
plt.xlabel('1st Componet')
plt.ylabel('2nd Componet')
plt.show()

