# ML_trader
An upgrade to the original random trader analysis, now including ML models.

<!-- TABLE OF CONTENTS -->
## Table of Contents 

* [Introduction](#introduction)
  * [Run With](#run-with)
* [Methods](#methods)
  * [Data Preparation](#data-preparation)
  * [Model Construction](#model-construction)
  * [Back Testing](#back-testing)
* [Results](#results)
* [References](#references)


<!-- Introduction -->
## Introduction



<!-- Run With -->
### Run With

Python 3.9.17
* Keras 1.13.1
* Joblib 1.3.2
* Numpy 1.23.0
* Pandas 2.1.0
* Sklearn 1.3.0
* Sklearn-contrib-py-earth 0.1.0 (requires numpy < 1.24.0 & Python = 3.9.X)
* Scipy 1.8.1
* Tensorflow 2.13.0
* Tqdm 4.66.1
* Xgboost 1.7.6
* Yfinace 0.2.28


<!-- Methods-->
## Methods

<!-- Data Preparation-->
### Data Preparation 
All tickers for securities on the NYSE, AMEX, and NASDAQ were collected. All existing security’s price information (open, high, low, close) were then downloaded through the YFinance API between 1/1/2013 and 8/1/2023. The following technical indicators were then calculated and added to each security’s price information dataframe for 21, 84, and 252 consecutive trading day windows as proxies for 1, 4, and 12 months of security activity. 

Technical Indicators:
* RSI
* MinMax Ratio
* Moving Average
* Bollinger Bands
* Daily True Average Range
* Volatility (as measured by historical standard deviation)

All security’s dataframes were written to disk as csv files. All subsequent analysis was conducted using data from securities with a closing price less than $100 and greater than $2. These thresholds were chosen since trading stocks under $2 often have penny stock trading restrictions and stocks greater than $100 are harder to evenly allocate in a small-scale portfolio.

Final training data was sampled from 1/1/2013 – 12/31/2020 and testing data was sampled from 1/1/2021 – 8/1/2023. In preparation for model training, 1.5 million trading days were sampled at random with a bias towards more recent data as follows:

* 20% between 1/1/2013 – 12/31/2014
* 20% between 1/1/2015 – 12/31/2016
* 25% between 1/1/2017 – 12/31/2018
* 35% between 1/1/2019 – 12/31/2020

This breakout was chosen since more recent data theoretically would be more important, however spot checking this assumption did not hold. For instance, training on data from 2013 to 2015 and testing on post 2022 data had nearly as accurate predictions as training data from 2019 to 2021. Due to time constraints these temporal interactions were not thoroughly explored, and the bias for more recent data was chosen to hedge against any errors which may have occurred during spot checking temporal importance.

Outliers in any of the technical indicators plus daily volume were then removed. Outliers were defined as instances where the data was greater than 4.5 standard deviations from the mean, and the threshold was chosen since it retained 95% of the data. This also clearly implies that the data does not follow a normal distribution. After removing outliers, the training dataset was rebalanced by random under sampling of the majority class leaving a final training dataset of 1.07 million data points. The data was finally rescaled to have mean of 0 and standard deviation of 1 using SKlearn’s StandardScaler.

Final testing dataset was built by randomly sampling 100,000 data points between 1/1/2021 and 1/26/2023 then rebalancing and standardizing the data. 

<!-- Model Construction-->
### Model Construction

Model hyper-parameters were tuned using the ROC-AUC metric under a 5-fold time-series cross validation similar to the image below. The total training data began as the same final training dataset mentioned above, but the removal of outliers, rebalancing and standardization was repeated for each fold independently. 

![image](https://github.com/seansteel3/ML_trader/assets/67161057/b4684f82-ecc4-43ec-a25d-52833e26154b)

Final model architectures: 

* Random Forest: max depth = 7, 100 Trees
* XGBoost: 100 Trees, LR = 0.3
* MARS: Default parameters
* Logistic Regression: Default SKlearn parameters
* FF-ANN: 3 hidden layers (14-7-5), Tanh Activation between hidden layers, 50% dropout before final layer with a sigmoid activation, trained for 9 epochs.

Models were then ensembled in three ways:

* Logistic regression stacked model
* Averaging of logits (voting)
* All model agreement

Logistic regression stacked model was rapidly discarded due to poor relative accuracy (61%) and ROC-AUC (0.57) and the averaging and agreement ensemble methods were then tested at several thresholds (50% and 80%) comparing accuracy and precision.

<!-- Back Testing-->
### Back Testing

Backtests were conducted using only testing data with a purchase date between 1/1/2021 and 1/26/2023. Backtests simulated buying 15 securities evenly allocated across $10,000 on a set purchase date, then selling and replacing securities if they increase in value by 10% or more, or after 126 consecutive trading days (~6 months), whichever occurred first. Each of the following conditions were simulated 1,500 times (with the exception of VTI) with random start dates for each simulation.

Conditions:

* VTI as a market proxy: Buy $10,000 on one day then sell it 126 trading days later for each starting day in the date range.
* Entirely Random Trader: Buy 15 securities entirely at random, then sell only after 126 consecutive trading days.
* Baseline Random Trader: Buy 15 securities entirely at random, then sell and replace securities if they increase in value by 10% or after 126 consecutive trading days.
* Model Average (Vote) at 50% threshold: Buy 15 random securities whose average logit prediction from the models is above 50%, then sell and replace securities if they increase in value by 10% or after 126 consecutive trading days.
* Model Average (Vote) at 80% threshold: Buy 15 random securities whose average logit prediction from the models is above 80%, then sell and replace securities if they increase in value by 10% or after 126 consecutive trading days.
* Model Agreement at 50% threshold: Buy 15 random securities whose logit predictions from all of the models are above 50%, then sell and replace securities if they increase in value by 10% or after 126 consecutive trading days.
* Model Agreement at 80% threshold: Buy 15 random securities whose logit predictions from all of the models are above 80%, then sell and replace securities if they increase in value by 10% or after 126 consecutive trading days.



<!-- Results-->
## Results

The five final individual models all preformed remarkably similar in terms of accuracy, precision, and ROC-AUC on the test data.
*Random Forest 
** Accuracy: 67.3%
** Precision: 62.4%
** ROC-AUC: 0.719
* XGBoost
** Accuracy: 66.8%
** Precision: 61.9%
** ROC-AUC: 0.718
* MARS
** Accuracy: 66.9%
** Precision: 62.1%
** ROC-AUC: 0.718
* Logistic Regression
** Accuracy: 66.7%
** Precision: 64.3%
** ROC-AUC: 0.712
* FF-ANN
** Accuracy: 66.9%
** Precision: 62.4%
** ROC-AUC: 0.721

Despite similar overall performance of each model, PCA projections make it clear the decision boundaries are not identical between models, potentially indicating value in model ensembling.




<!-- References-->
## References
