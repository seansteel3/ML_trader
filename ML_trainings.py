#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:22:21 2025

@author: seansteele
"""
import os
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tempfile
import requests
import joblib
from io import BytesIO
import matplotlib.pyplot as plt

from random_trader_utils.training_utils import gain_feats, neg_feats, keeps, \
    index_resample_tab, build_dense, predict_purchases, heatmap, roc_curve, auc


#%% SET GLOBAL VARIABLES FOR THIS TRAINING RUN; note all model specific optimizations besides LRs are in model specific sections
train_test_date_cutoff = '2023-01-01'
neg_model_target = 'end_n30'
pos_model_target = 'gain_10'

neg_lr = ExponentialDecay(
    initial_learning_rate = 0.001,
    decay_steps = 2000,
    decay_rate = 0.96,
)

pos_lr = ExponentialDecay(
    initial_learning_rate = 0.001,
    decay_steps = 10000,
    decay_rate = 0.96,
)

gain_url = "https://api.github.com/repos/seansteel3/ML_trader/contents/production_models/modelsg10/"
neg_url = "https://api.github.com/repos/seansteel3/ML_trader/contents/production_models/modelsn30/"

save_negpath = '/Users/seansteele/Desktop/Random Trader/production_models/modelsn30_123124/'
save_gainpath = '/Users/seansteele/Desktop/Random Trader/production_models/modelsg10_123124/'


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

def plot_rocauc(test_y, prediction_consensus, title):
    fpr, tpr, threshold = roc_curve(test_y, prediction_consensus)
    roc_auc = auc(fpr, tpr)
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
#%% LOAD DATA
data = pd.read_csv('.//datapull_123124//alldata_123124.csv')

#normalize columns tied to close price
price_norm_colz = ['bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 'bb_upp84', 
                  'bb_low84', 'bb_center252',  'bb_upp252', 'bb_low252', 'atr_21',
                  'atr_84', 'atr_252', 'macd_21', 'macd_84', 'macd_252']
for i in range(len(price_norm_colz)):
    col = price_norm_colz[i]
    data[col] = data[col]/data['close']
    
#remove unneeded columns
data = data[keeps]

#remove untrainable data after subsetting to columns used for ML
data.dropna(inplace = True)
data = data[data['price_6mo'] != -1]

#super subsets into train/test
train = data[data['date'] < train_test_date_cutoff]
test = data[data['date'] >= train_test_date_cutoff]

#%% NEGATIVE MODEL TRAINING

#rescale
qt30 = QuantileTransformer(n_quantiles = 500, subsample = 10000, 
                         output_distribution = 'normal')
x_train30 = qt30.fit_transform(train[neg_feats])
x_test30 = qt30.transform(test[neg_feats]) 

#balance data
x_train_30, y_train_30 = index_resample_tab(x_train30, train[neg_model_target].values)
x_test_30, y_test_30 = index_resample_tab(x_test30, test[neg_model_target].values)

#init models
nn1_30 = build_dense(22, neg_lr, 0.5, layer_cts = [9,4], activations = ['relu', 'relu'])
nn2_30 = build_dense(22, neg_lr, 0.5, layer_cts = [7,5], activations = ['tanh', 'tanh'])
xg_30 = xgb.XGBClassifier(max_depth = 3, eta = 0.1, gamma = 0, subsample = 0.9, n_jobs = 12)
rf_30 = RandomForestClassifier(max_depth=7,n_estimators = 100,criterion='entropy',min_samples_split=200,min_samples_leaf=500,n_jobs = 12)
lrc_30 = LogisticRegression()

#training
nn1_30.fit(x_train_30, y_train_30, epochs = 8, batch_size = 252
           ,validation_data = (x_test_30, y_test_30)
           )
nn2_30.fit(x_train_30, y_train_30, epochs = 8, batch_size = 252
           ,validation_data = (x_test_30, y_test_30)
           )
xg_30.fit(x_train_30, y_train_30)
rf_30.fit(x_train_30, y_train_30)
lrc_30.fit(x_train_30, y_train_30)

models30 = [nn1_30, nn2_30, xg_30, rf_30, lrc_30, qt30]

#%% GAIN MODEL TRAINING

#rescale
qt10 = QuantileTransformer(n_quantiles = 500, subsample = 10000, 
                         output_distribution = 'normal')
x_train10 = qt10.fit_transform(train[gain_feats])
x_test10 = qt10.transform(test[gain_feats]) 

#balance data
x_train_10, y_train_10 = index_resample_tab(x_train10, train[pos_model_target].values)
x_test_10, y_test_10 = index_resample_tab(x_test10, test[pos_model_target].values)

#init models
nn1_10 = build_dense(len(gain_feats),pos_lr, 0.5, [8], ['relu'])
nn2_10 = build_dense(len(gain_feats),pos_lr, 0.5, [7,7], ['tanh', 'tanh'])
xg_10 = xgb.XGBClassifier(max_depth = 3, eta = 0.1, gamma = 0, 
                          subsample = 0.9, n_jobs = 12)
rf_10 = RandomForestClassifier(max_depth=7,n_estimators = 100,criterion='entropy',
                               min_samples_split=200,min_samples_leaf=50,n_jobs = 12)
lrc_10 = LogisticRegression()

#training
nn1_10.fit(x_train_10, y_train_10, epochs = 8, batch_size = 252
           ,validation_data = (x_test_10, y_test_10)
           )
nn2_10.fit(x_train_10, y_train_10, epochs = 8, batch_size = 252
           ,validation_data = (x_test_10, y_test_10)
           )
xg_10.fit(x_train_10, y_train_10)
rf_10.fit(x_train_10, y_train_10)
lrc_10.fit(x_train_10, y_train_10)

models10 = [nn1_10, nn2_10, xg_10, rf_10, lrc_10, qt10]

#%% PREDICT AND ASSESS VS OLD/CURRENT MODELS

#download current models
current_neg, current_neg_trans = download_models(neg_url)
current_gain, current_gain_trans = download_models(gain_url)

#predict on all test data (no RF, old models incompatable)
#negative prediction
old_x_test30 = current_neg_trans.transform(test[neg_feats]) 
old_pred30_test = predict_purchases(old_x_test30, 0.5, [current_neg[0], current_neg[1],
                                                        current_neg[2], current_neg[4]])
new_x_test30 = qt30.transform(test[neg_feats]) 
new_pred30_test = predict_purchases(new_x_test30, 0.5, [models30[0], models30[1],
                                                        models30[2], models30[4],
                                                        models30[3]])
#positive prediction
old_x_test10 = current_gain_trans.transform(test[gain_feats]) 
old_pred10_test = predict_purchases(old_x_test10, 0.5, [current_gain[0], current_gain[1],
                                                        current_gain[2], current_gain[4]])
new_x_test10 = qt10.transform(test[gain_feats]) 
new_pred10_test = predict_purchases(new_x_test10, 0.5, [models10[0], models10[1],
                                                        models10[2], models10[4],
                                                        models10[3]])

#assess confusion matrix
heatmap(test.end_n30, old_pred30_test.Prediction, 'Old Neg Models Unbalanced')
heatmap(test.end_n30, new_pred30_test.Prediction, 'New Neg Models Unbalanced')

heatmap(test.gain_10, old_pred10_test.Prediction, 'Old Gain Models Unbalanced')
heatmap(test.gain_10, new_pred10_test.Prediction, 'New Gain Models Unbalanced')

#assess ROC-AUC
plot_rocauc(test['end_n30'].values, old_pred30_test.Consensus, 'Old Neg ROC-AUC')
plot_rocauc(test['end_n30'].values, new_pred30_test.Consensus, 'New Neg ROC-AUC')
plot_rocauc(test['gain_10'].values, old_pred10_test.Consensus, 'Old Gain ROC-AUC')
plot_rocauc(test['gain_10'].values, new_pred10_test.Consensus, 'New Gain ROC-AUC')

    

#%% TRAIN OVER ALL DATA IF MEETS QA: overwrite above models

### NEGATIVE FULL MODEL ###
#rescale
qt30 = QuantileTransformer(n_quantiles = 500, subsample = 10000, 
                         output_distribution = 'normal')
x_30 = qt30.fit_transform(data[neg_feats])

#balance data
x_30, y_30 = index_resample_tab(x_30, data[neg_model_target].values)

#init models
nn1_30 = build_dense(22, neg_lr, 0.5, layer_cts = [9,4], activations = ['relu', 'relu'])
nn2_30 = build_dense(22, neg_lr, 0.5, layer_cts = [7,5], activations = ['tanh', 'tanh'])
xg_30 = xgb.XGBClassifier(max_depth = 3, eta = 0.1, gamma = 0, subsample = 0.9, n_jobs = 12)
rf_30 = RandomForestClassifier(max_depth=7,n_estimators = 100,criterion='entropy',min_samples_split=200,min_samples_leaf=500,n_jobs = 12)
lrc_30 = LogisticRegression()

#training
nn1_30.fit(x_30, y_30, epochs = 8, batch_size = 252
           )
nn2_30.fit(x_30, y_30, epochs = 8, batch_size = 252
           )
xg_30.fit(x_30, y_30)
rf_30.fit(x_30, y_30)
lrc_30.fit(x_30, y_30)

models30 = [nn1_30, nn2_30, xg_30, rf_30, lrc_30, qt30]


### GAIN FULL MODEL ###
#rescale
qt10 = QuantileTransformer(n_quantiles = 500, subsample = 10000, 
                         output_distribution = 'normal')
x_10 = qt10.fit_transform(data[gain_feats])

#balance data
x_10, y_10 = index_resample_tab(x_10, data[pos_model_target].values)

#init models
nn1_10 = build_dense(len(gain_feats),pos_lr, 0.5, [8], ['relu'])
nn2_10 = build_dense(len(gain_feats),pos_lr, 0.5, [7,7], ['tanh', 'tanh'])
xg_10 = xgb.XGBClassifier(max_depth = 3, eta = 0.1, gamma = 0, 
                          subsample = 0.9, n_jobs = 12)
rf_10 = RandomForestClassifier(max_depth=7,n_estimators = 100,criterion='entropy',
                               min_samples_split=200,min_samples_leaf=50,n_jobs = 12)
lrc_10 = LogisticRegression()

#training
nn1_10.fit(x_10, y_10, epochs = 8, batch_size = 252
           )
nn2_10.fit(x_10, y_10, epochs = 8, batch_size = 252
           )
xg_10.fit(x_10, y_10)
rf_10.fit(x_10, y_10)
lrc_10.fit(x_10, y_10)

models10 = [nn1_10, nn2_10, xg_10, rf_10, lrc_10, qt10]

# Make sure Confusion matrices look acceptable
x_30all = qt30.transform(data[neg_feats])
x_10all = qt10.transform(data[gain_feats])

pred_30 = predict_purchases(x_30all, 0.5, models30[:-1])
pred_10 = predict_purchases(x_10all, 0.5, models10[:-1])

heatmap(data.end_n30, pred_30.Prediction, 'Neg Models Unbalanced')
heatmap(data.gain_10, pred_10.Prediction, 'Gain Models Unbalanced')


#%% SAVE MODELS - local save then upload to github post testing

#save neg30
os.chdir(save_negpath)

names = ['nn_neg30_0', 'nn_neg30_1', 'xgb_neg30', 'rf_neg30', 'lr_neg30', 'qt30']
for i in range(len(models30)):
    model = models30[i]
    if isinstance(model, tf.keras.Model):
        name = names[i] + '.h5'
        model.save(name, save_format = 'h5')
    else:
        name = names[i] + '.joblib'
        joblib.dump(model, name)

#save gain10
os.chdir(save_gainpath)

names = ['nn_gain10_0', 'nn_gain10_1', 'xgb_gain10', 'rf_gain10', 'lr_gain10', 'qt10']
for i in range(len(models10)):
    model = models10[i]
    if isinstance(model, tf.keras.Model):
        name = names[i] + '.h5'
        model.save(name, save_format = 'h5')
    else:
        name = names[i] + '.joblib'
        joblib.dump(model, name)
