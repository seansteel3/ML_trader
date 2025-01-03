#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:18:41 2025

@author: seansteele
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA,FastICA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import random
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from tensorflow.keras.models import Model, Sequential


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
            pred = list(models[i].predict(data, batch_size = 2048))
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

def build_train_validate_test(data, n_partitions, date_column):
    """
    Parameters
    ----------
    data : DF
    n_partitions : number of total timeseries partitions
    date_column : string denoting date column

    Returns
    -------
    trains/valids : list of pandas dataframes
    test: pandas dataframe
    """
    all_dates = list(set(data[date_column]))
    all_dates.sort()
    partition_size = int(len(all_dates)/(n_partitions + 1))

    breaks = []
    for i in range(n_partitions + 1):
        breaks.append(all_dates[partition_size * (i + 1)])

    trains = []
    tests = []
    for i in range(n_partitions + 1):
        s1 = data[data[date_column] < breaks[i]]
        s1 = s1.reset_index(inplace = False, drop = True)
        trains.append(s1)
        if i != (n_partitions + 1) - 1:
            s2 = data[((data[date_column] >= breaks[i]) & (data[date_column] < breaks[i + 1]))]
            s2 = s2.reset_index(inplace = False, drop = True)
            tests.append(s2)
        else:
            s3 = data[data[date_column] >= breaks[i]]
            s3 = s3.reset_index(inplace = False, drop = True)
            tests.append(s3)
    trains = trains[:-2]
    valids = tests[:-2]
    test = pd.concat(tests[-2:])
    return trains, valids, test


def generate_arrays(trains, valids, test, x_cols, y_col):
    """
    Parameters
    ----------
    trains : list of train dataframes
    valids : list of validation dataframes
    test : test dataframe
    x_cols : list of x_columns
    y_col : y_column

    Returns
    -------
    trains : Tuple of a list of ndarrays in format (x_array_list, y_array_list)
    valids : Tuple of a list of  ndarrays in format (x_array_list, y_array_list)
    tests : Tuple of a list of  ndarrays in format (x_array, y_array)

    """
    trains_x = []
    valids_x = []
    trains_y = []
    valids_y = []
    for i in range(len(trains)):
        trains_x.append(np.array(trains[i][x_cols]))
        valids_x.append(np.array(valids[i][x_cols]))
        trains_y.append(trains[i][y_col].values)
        valids_y.append(valids[i][y_col].values)
    trains = (trains_x, trains_y)
    valids = (valids_x, valids_y)
    tests = (np.array(test[x_cols]), test[y_col].values)
    return trains, valids, tests


def apply_quantile_transformer(trains, valids, tests, n_quantiles = 500,
                               subsample = 10000, output_distribution = 'uniform'):
    """
    Parameters
    ----------
    trains : tuple of form (xdata_array_list, ydata_array_list)
    valids : tuple of form (xdata_array_list, ydata_array_list)
    tests : tuple of form (xdata_array, ydata_array)
    n_quantiles : Optional, The default is 500.
    subsample : Optional, The default is 10000.
    output_distribution : Otional, The default is 'uniform'.

    Returns
    -------
    trains, valids, splits: Same format as input, 
                            but with quantile rescaling applied to xdata

    """
    for i in range(len(trains[0])):
        if i != len(trains[0]):
            qt = QuantileTransformer(n_quantiles = n_quantiles, subsample = subsample, 
                                     output_distribution = output_distribution)
            trains[0][i] = qt.fit_transform(trains[0][i])
            valids[0][i] = qt.transform(valids[0][i]) 
        else:
            qt = QuantileTransformer(n_quantiles = n_quantiles, subsample = subsample, 
                                     output_distribution = output_distribution)
            trains[0][i] = qt.fit_transform(trains[0][i])
            valids[0][i] = qt.transform(valids[0][i]) 
            tests[0] = qt.transform(tests[0]) 
    return trains, valids, tests

def apply_undersampler(trains, valids, tests):
    '''
    Parameters
    ----------
    trains : tuple of form (xdata_array_list, ydata_array_list)
    valids : tuple of form (xdata_array_list, ydata_array_list)
    tests : tuple of form (xdata_array, ydata_array)

    Returns
    -------
    Same type and format as input, but with random undersampling applied

    '''
    trains_bal_x = []
    trains_bal_y = []
    valids_bal_x = []
    valids_bal_y = []
    for i in range(len(trains[0])):
        x1b, y1b = index_resample_tab(trains[0][i], trains[1][i])
        trains_bal_x.append(x1b)
        trains_bal_y.append(y1b)
        
        x2b, y2b = index_resample_tab(valids[0][i], valids[1][i])
        valids_bal_x.append(x2b)
        valids_bal_y.append(y2b)

    x3b, y3b = index_resample_tab(tests[0], tests[1])
    tests_bal = (x3b, y3b)
    trains_bal = (trains_bal_x, trains_bal_y)
    valids_bal = (valids_bal_x, valids_bal_y)
    return trains_bal, valids_bal, tests_bal


def heatmap(ys, preds, dataset_type):
    acc = accuracy_score(ys, preds)
    prec = precision_score(ys, preds)
    conf = confusion_matrix(ys, preds)
    anti_prec = conf[0][0]/(conf[0][0] + conf[1][0])
    ax= plt.subplot()
    sns.heatmap(conf, annot=True, fmt='g', ax=ax); 
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(f'{dataset_type} Accuracy: ' + str(np.round(acc * 100, 2)) + '%' 
                 + f'\n {dataset_type} Precision: '+ str(np.round(prec * 100, 2)) + '%'
                 + f'\n {dataset_type} Negative Precision: '+ str(np.round(anti_prec * 100, 2)) + '%'); 
    ax.xaxis.set_ticklabels(['Pred: 0', 'Pred: 1']); ax.yaxis.set_ticklabels(['Pred: 0', 'Pred: 1']);
    plt.show()
    
def index_resample_tab(datax, datay):
    rus = RandomUnderSampler()
    rus.fit_resample(datax, datay)
    random.shuffle(rus.sample_indices_)
    datax = datax[rus.sample_indices_] 
    datay = datay[rus.sample_indices_] 
    return datax, datay

def build_dense(in_shape, lr, dropout, layer_cts, activations):
    model = Sequential()
    model.add(Input(shape = (in_shape,)))
    for i in range(len(layer_cts)):
        model.add(Dense(layer_cts[i], activations[i], name = str('Layer' + str(i))))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation = 'sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer = opt, loss = 'binary_crossentropy',
                     metrics = ['acc'])
    return model

def assess_models(xtrains, ytrains, xtests, ytests, epochs):
    accs = []
    unprecs = []
    precs = []
    for i in range(len(xtrains)):
        #prepare a reset model
        nn_model = build_dense(93, 25, 0.5, 0.001, 'tanh')
        nn_model.load_weights('empty_model.h5')
        #fit
        nn_model.fit(xtrains[i], ytrains[i], epochs = epochs)
        #get preds and store metrics
        preds = np.round(nn_model.predict(xtests[i]))
        conf = confusion_matrix(ytests[i], preds)
        accs.append(accuracy_score(ytests[i], preds))
        unprec = conf[0,0]/(conf[0,0] + conf[1,0])
        unprecs.append(unprec)
        precs.append(precision_score(ytests[i], preds))
    return np.mean(accs), np.mean(unprecs), np.mean(precs)

def qt_all_prep(n_quantiles, subsample, output_distribution, xtrains, xtests):
    xtrains_list = []
    xtests_list = []
    for i in range(len(xtrains)):
        qt = QuantileTransformer(n_quantiles = n_quantiles, subsample = subsample, output_distribution = output_distribution)
        xtrain = qt.fit_transform(xtrains[i])
        xtest = qt.transform(xtests[i])
        xtrains_list.append(xtrain)
        xtests_list.append(xtest)
    return xtrains_list, xtests_list

def under_smote(datax, datay, und_prop):
    xy = np.concatenate((datax,datay.reshape(-1,1)), axis = 1)
    np.random.shuffle(xy)
    end = int(len(xy) * und_prop)
    #split data and undersample
    xy_und = xy[:end,:]
    x_und = xy_und[:,:-1]
    y_und = xy_und[:,-1]
    x_und, y_und = index_resample_tab(x_und, y_und)
    xy_und = np.concatenate((x_und,y_und.reshape(-1,1)), axis = 1)
    #return split off data and SMOTE
    xy_imb = xy[end:,:]
    xy_smote = np.concatenate((xy_imb,xy_imb), axis = 0)
    x_smote = xy_smote[:,:-1]
    y_smote = xy_smote[:,-1]
    smt = SMOTE()
    x_bal, y_bal = smt.fit_resample(x_smote, y_smote)
    return x_bal, y_bal

def apply_rebalancer(trains, valids, tests, strategy):
    '''
    Parameters
    ----------
    trains : tuple of form (xdata_array_list, ydata_array_list)
    valids : tuple of form (xdata_array_list, ydata_array_list)
    tests : tuple of form (xdata_array, ydata_array)

    Returns
    -------
    Same type and format as input, but with random undersampling applied

    '''
    trains_bal_x = []
    trains_bal_y = []
    valids_bal_x = []
    valids_bal_y = []
    for i in range(len(trains[0])):
        if strategy == 'Under':
            x1b, y1b = index_resample_tab(trains[0][i], trains[1][i]) 
        elif strategy == 'SMOTE':
            smplr = SMOTE()
            x1b, y1b = smplr.fit_resample(trains[0][i], trains[1][i])
        elif strategy == 'SMOTE_borderline1':
            smplr = BorderlineSMOTE(kind = 'borderline-1')
            x1b, y1b = smplr.fit_resample(trains[0][i], trains[1][i])
        elif strategy == 'SMOTE_borderline2':
            smplr = BorderlineSMOTE(kind = 'borderline-2')
            x1b, y1b = smplr.fit_resample(trains[0][i], trains[1][i])
        elif strategy == 'ADASYN':
            smplr = ADASYN()
            x1b, y1b = smplr.fit_resample(trains[0][i], trains[1][i])
        elif strategy == 'ENN':
            smplr = EditedNearestNeighbours()
            x1b, y1b = smplr.fit_resample(trains[0][i], trains[1][i])
        elif strategy == 'NearMiss':
            smplr = NearMiss()
            x1b, y1b = smplr.fit_resample(trains[0][i], trains[1][i])
        elif strategy == 'Tomek':
            smplr = TomekLinks()
            x1b, y1b = smplr.fit_resample(trains[0][i], trains[1][i])
        elif strategy == 'SMOTETomek':
            smplr = SMOTETomek()
            x1b, y1b = smplr.fit_resample(trains[0][i], trains[1][i])
        elif strategy == 'SMOTEENN':
            smplr = SMOTEENN()
            x1b, y1b = smplr.fit_resample(trains[0][i], trains[1][i]) 
        elif strategy == 'UnderSMOTE':
            x1b, y1b = under_smote(trains[0][i], trains[1][i], und_prop = 0.9)
        
        trains_bal_x.append(x1b)
        trains_bal_y.append(y1b)
        
        x2b, y2b = index_resample_tab(valids[0][i], valids[1][i]) #leave valdiation on undersampling?
        valids_bal_x.append(x2b)
        valids_bal_y.append(y2b)

    x3b, y3b = index_resample_tab(tests[0], tests[1])
    tests_bal = (x3b, y3b)
    trains_bal = (trains_bal_x, trains_bal_y)
    valids_bal = (valids_bal_x, valids_bal_y)
    return trains_bal, valids_bal, tests_bal


def neg_precision_score(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    score = conf_mat[0][0]/(conf_mat[0][0] + conf_mat[1][0])
    return score


x_cols = ['volume', 'bb_center21', 'bb_upp21', 'bb_low21', 'bb_center84', 
          'bb_upp84', 'bb_low84', 'bb_center252', 'bb_upp252', 'bb_low252', 
          'minmax21', 'minmax84', 'minmax252', 'atr_21', 'atr_84', 'atr_252', 
          'rsi_21', 'rsi_84', 'rsi_252', 'vol_21', 'vol_84', 'vol_252', 
          'revenuePerShare', 'netIncomePerShare', 'operatingCashFlowPerShare', 
          'freeCashFlowPerShare', 'cashPerShare', 'bookValuePerShare', 
          'interestDebtPerShare', 'marketCap', 'totalDebtToCapitalization',
          'peRatio', 'priceToSalesRatio', 'pocfratio', 'pfcfRatio', 'pbRatio', 
          'enterpriseValueOverEBITDA', 'earningsYield', 'freeCashFlowYield', 
          'debtToEquity', 'debtToAssets', 'netDebtToEBITDA', 'currentRatio', 
          'interestCoverage', 'incomeQuality', 'dividendYield', 'payoutRatio', 
          'researchAndDdevelopementToRevenue', 'intangiblesToTotalAssets', 
          'capexToRevenue', 'capexToDepreciation', 'stockBasedCompensationToRevenue', 
          'roic', 'returnOnTangibleAssets', 'workingCapital', 'tangibleAssetValue', 
          'netCurrentAssetValue', 'averageReceivables', 'averagePayables', 
          'daysSalesOutstanding', 'daysPayablesOutstanding', 'daysOfInventoryOnHand', 
          'receivablesTurnover', 'payablesTurnover', 'inventoryTurnover', 'roe', 
          'capexPerShare', 'assetTurnover', 'capitalExpenditureCoverageRatio', 
          'cashFlowCoverageRatios', 'cashRatio', 'companyEquityMultiplier', 
          'shortTermCoverageRatios', 'effectiveTaxRate', 'fixedAssetTurnover', 
          'longTermDebtToCapitalization', 'netIncomePerEBT', 'priceEarningsToGrowthRatio', 
          'quickRatio', 'returnOnCapitalEmployed']


y_cols = [ 'gain_10', 'end_n30']


gain_feats = ['atr_21', 'atr_84', 'vol_252', 'vol_84', 'atr_252', 'vol_21', 
               'bb_upp21', 'bb_upp84', 'bb_upp252', 'bb_low84', 'bb_low252', 
               'bb_center252', 'dividendYield', 'bb_low21', 'bb_center84', 
               'bb_center21', 'bookValuePerShare', 'netCurrentAssetValue']

neg_feats = ['atr_252', 'atr_84', 'atr_21', 'returnOnTangibleAssets', 'peRatio', 
             'netIncomePerShare', 'earningsYield', 'roic', 'returnOnCapitalEmployed',
             'pocfratio', 'bb_low252', 'vol_252', 'enterpriseValueOverEBITDA', 
             'operatingCashFlowPerShare', 'freeCashFlowPerShare', 'bb_upp252', 
             'vol_84', 'freeCashFlowYield', 'bb_upp84', 'capitalExpenditureCoverageRatio', 
             'pfcfRatio', 'roe']

ml_cols = gain_feats + neg_feats


ml_cols = list(set(neg_feats + gain_feats)) #+ neg50_feats

keeps = ['date', 'ticker', 'high', 'low', 'open', 'close', 'volume', 'gain_5', 
         'gain_10', 'gain_15', 'lose_5', 'lose_10', 'lose_15', 'end_p5', 'end_p10', 
         'end_p30', 'end_n10', 'end_n30', 'end_n50', 'date_6mo', 'price_6mo', 
         'date_10per', 'price_10per_high', 'price_10per_low', 
         'price_10per_close', 'price_10per_thresh'] + ml_cols

