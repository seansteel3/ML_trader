# ML_trader
An upgrade to the original random trader analysis, now including ML models.

![image](https://github.com/seansteel3/ML_trader/assets/67161057/5c342a37-357c-41df-835d-1b65a6c19adf)


<!-- TABLE OF CONTENTS -->
## Table of Contents 

* [Introduction](#introduction)
  * [Run With](#run-with)
* [Methods](#methods)
  * [Data Preparation](#data-preparation)
  * [Model Construction](#model-construction)
  * [Back Testing](#back-testing)
* [Results](#results)
  * [Individual Models](#individual-models)
  * [Model Ensembles](#model-ensembles)
  * [Back Testing](#back-testing)
  * [Shap Analysis](#shap-analysis)
* [Conclusion](#conclusion)


<!-- Introduction -->
## Introduction

This project aimed to be an extension of the original Random Trader analysis, specifically investigating if machine learning could improve upon the random trading strategy. The trading strategy that came out of the original analysis settled upon creating portfolios by choosing 15 securities entirely at random whose share value was between $2 and $75. These securities were then sold and replaced after six months, or if the security increased in value by 10% or more. The goal of this project was to replace the random selection process with machine learning models that picked securities statistically more likely to increase in value by 10% or more within 6 months.

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
All tickers for securities on the NYSE, AMEX, and NASDAQ were collected from the financialmodelingprep API. All existing security’s price information (open, high, low, close, volume) were then downloaded through the YFinance API between 1/1/2013 and 8/1/2023. The following technical indicators were then calculated and added to each security’s price information dataframe for 21, 84, and 252 consecutive trading day windows as proxies for 1, 4, and 12 months of security activity. 

Technical Indicators:
* RSI
* MinMax Ratio
* Moving Average (N)
* Bollinger Bands (N)
* Daily True Average Range (N)
* Volatility (N) (as measured by historical standard deviation)

Technical indicators with a (N) beside them were normalized by dividing by the closing price since they all are natively in the scale of the close price.

All security’s dataframes were written to disk as csv files. All subsequent analysis was conducted using data from securities with a closing price less than $100 and greater than $2. These thresholds were chosen since trading stocks under $2 often have penny stock trading restrictions and stocks greater than $100 are harder to evenly allocate in a small-scale portfolio.

Final training data was sampled from 1/1/2013 – 12/31/2020 and testing data was sampled from 1/1/2021 – 8/1/2023. In preparation for model training, 1.5 million trades were sampled at random with a bias towards more recent data as follows:

* 20% between 1/1/2013 – 12/31/2014
* 20% between 1/1/2015 – 12/31/2016
* 25% between 1/1/2017 – 12/31/2018
* 35% between 1/1/2019 – 12/31/2020

This breakout was chosen since more recent data theoretically would be more important, however spot checking this assumption did not hold. For instance, training on data from 2013 to 2015 and testing on post 2022 data had nearly as accurate predictions as training data from 2019 to 2021. A likely, but not fully confirmed, explaination for this behavior is the fact the majority of features in the data exibit little to no temporal trend. However, the data cannot be considered stationary due to significant seasonality within several features. Due to time constraints these temporal interactions were not thoroughly explored, and the bias for more recent data was chosen to hedge against any errors which may have occurred during spot checking temporal importance.

Outliers in any of the technical indicators plus daily volume were then removed. Outliers were defined as instances where the data was greater than 4.5 standard deviations from the mean, and the threshold was chosen since it retained 95% of the data. This also clearly implies that the data does not follow a normal distribution. After removing outliers, the training dataset was rebalanced by random under sampling of the majority class leaving a final training dataset of 1.07 million data points. The data was finally rescaled to have mean of 0 and standard deviation of 1 using SKlearn’s StandardScaler.

Final testing dataset was built by randomly sampling 100,000 data points between 1/1/2021 and 1/26/2023 then rebalancing and standardizing the data. Outliers were not removed from testing data.

<!-- Model Construction-->
### Model Construction

Model hyper-parameters were tuned using the ROC-AUC metric under a 5-fold time-series cross validation similar to the image below. The total training data began as the same final training dataset mentioned above, but the removal of outliers, rebalancing and standardization was repeated for each fold independently. 

![image](https://github.com/seansteel3/ML_trader/assets/67161057/b4684f82-ecc4-43ec-a25d-52833e26154b)

Final model architectures: 

* Random Forest: max depth = 7, 100 Trees
* XGBoost: 100 Trees, LR = 0.3
* MARS: Default parameters
* Logistic Regression: Default SKlearn parameters
* FF-ANN: 3 hidden layers (14-7-5), Tanh Activation between hidden layers, 50% dropout before final layer with a sigmoid activation, trained for 9 epochs with 0.001 starting learning rate using the Nadam optimizer and an exponential decay of 0.9 on the learning rate every 100,000 steps.

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

<!-- Individual Models-->
### Individual Models

The five final individual models all preformed remarkably similar in terms of accuracy, precision, and ROC-AUC on the test data.
* Random Forest 
  * Accuracy: 67.3%
  * Precision: 62.4%
  * ROC-AUC: 0.719
* XGBoost
  * Accuracy: 66.8%
  * Precision: 61.9%
  * ROC-AUC: 0.718
* MARS
  * Accuracy: 66.9%
  * Precision: 62.1%
  * ROC-AUC: 0.718
* Logistic Regression
  * Accuracy: 66.7%
  * Precision: 64.3%
  * ROC-AUC: 0.712
* FF-ANN
  * Accuracy: 66.9%
  * Precision: 62.4%
  * ROC-AUC: 0.721
  
![image](https://github.com/seansteel3/ML_trader/assets/67161057/ab0cce90-a4ab-4ed0-98b6-a7105d309197)


Despite similar overall performance of each model, PCA projections make it clear the decision boundaries are not identical between models, potentially indicating value in model ensembling.

![Screenshot 2023-09-14 210301](https://github.com/seansteel3/ML_trader/assets/67161057/8ce81d61-2fa1-4e6f-a3be-99c5b08514f0)

<!-- Model Ensembles-->
### Model Ensembles

The first ensemble strategy explored was model stacking with a logistic regression as the final classifier. The stacked model was quickly discarded as the stacked model accuracy dropped to just 61.1% and ROC-AUC score dropped to 0.58.

The next ensemble strategy implemented was averaging of model logit predictions referred to as “average/mean voting” in the slide presentation. This strategy nominally yielded the highest accuracy of 67.3% and a comparable 62.8% precision. This ensembling method was similar to the final method of total agreement of models with an accuracy of 66.8% and overall highest precision of 64.6%. 

Accuracy is not necessarily the overall most important factor when modeling investment strategies. False negative results certainly incur a cost, but that cost is theoretical and only affects future potential earnings. Meanwhile false positive results can result in real losses to current values. Therefore, precision is likely more important than accuracy when assessing model performance.

Raising the thresholds for both the mean voting and all model agreement ensembles from the standard 50% to 80% nearly trades percentage points of accuracy for precision evenly. The mean vote ensemble’s accuracy dropped 10.2 percentage points, but gained 9.3 percentage points in precision, and the all-model agreement ensemble’s accuracy dropped 13.2 percentage points but gained 10.9 percentage points in precision.

![Screenshot 2023-09-14 210446](https://github.com/seansteel3/ML_trader/assets/67161057/7fd9e941-25ef-4606-88cd-2fb8748c1ca8)


<!-- Back Testing-->
### Back Testing

Back-tests during the testing date ranges were conducted to ascertain if precision is in fact more important than accuracy, and if these models improve trading performance over the baseline random trader. Specifically, 7 back-tests were conducted each with 1,500 simulations. Four ensemble models, one on the baseline random trader from the previous analysis, one on a completely random trader (no sale and replacement when a security increases in value by 10%), and one apples-to-oranges comparison with VTI as a loose proxy for a market performance comparison.

Back-tests of VTI yielded mildly positive results with a mean return of 0.4% and a 45% negative return chance, while the entirely random trader had remarkably poor performance with an average of -7.8% returns and 74.5% chance of negative returns. The base random trader on the other hand had positive returns of 3.8% on average and only a 41.1% chance of losses. 

Back-tests across all the ensemble models shows a marked improvement over the truly random stock selection that at first glance seems to contradict the implications of the efficient market hypothesis. The vote and all-agreement ensembles at 50% threshold have mean returns of 6.2% and 7.7% respectively, and both under a 40% negative return chance. The 80% threshold vote and all-agreement returns have a massive improvement with mean returns of 19% and 24.2% respectively and right around 30% negative return chance. Additionally, comparison between the 50% and 80% thresholds solidly confirms precision and the tangible cost of false positives to be more important in an investing setting than accuracy and false negatives.

![Screenshot 2023-09-14 210545](https://github.com/seansteel3/ML_trader/assets/67161057/de43e3d7-e8df-49c9-99b3-1d6b498d5018)


Despite the glamorous performance of these models, returns overtime paint a similar overarching story, but with a few critical caveats. During steep market downturns all models and non-model back-tests preform nearly equivalent with expected returns of -12% to -18% approximately matching the market at the time. Where the models shine, especially the 80% threshold models, are during substantial market upswings, averaging as high as 60% returns during some months. 

![image](https://github.com/seansteel3/ML_trader/assets/67161057/067941eb-23dc-46fa-a11e-c920ddd773d5)


The temporal breakout of returns also makes it clear average returns are highly dependent on overall market pressures. The adage that “past performance does not guarantee future success” absolutely holds true here, and deployment of these models into different timeframes will undoubtedly see entirely different return distributions. Further, in alignment of the efficient market hypothesis, it is unlikely that the performance of these models in a trading setting can be forecasted at all. 

To further emphasize the efficient market hypothesis, the random forest classifier was reconfigured to predict if the overall value of the security would increase or decrease at the *end* of the next 126 trading days with respective accuracies of 51.4% and 53.1% on the test data. While the classifier was nominally able to do better than a random guess on test data, creating 95% confidence intervals by bootstrapping reveals this preformance to likely be due to random sampling of the test data. 

* Average accuracy for predicting increase in value: 51.4% (+/- 2.5%)
* Average accuracy for predicting decrease in value: 53.1% (+/- 3.3%)

<!-- Shap Analysis -->
### Shap Analysis

Finally, shap beeswarm analysis of the original random forest model (not the one reconfigured to predict if a security would increase at the end of 126 days)  gives some insight into how the models calculate if a security will increase by 10% in the next six months nor not. Securities with high yearly and monthly historical volatility, as well as 4 and 1-month average true range, but lower yearly average true range tend to be predicted as a buy. Additionally, 265 of 300 randomly chosen 80% threshold all model agreement ensemble predictions were below the security’s historical annual mean and all 300 were below 1 standard deviation above the security’s historical mean.  Taken together, these data imply the models likely work in a regression to the mean style strategy. Securities who are below their mean value, with high volatility, and therefore high chance for larger price swings, tend to be chosen by the models. 

![shaps](https://github.com/seansteel3/ML_trader/assets/67161057/be12db3c-10b7-4cfd-bad2-f6c6d61a19e2)

<!-- Conclusion-->
## Conclusion

In the end, using an ensemble of machine learning models, analyzing a series of volatility based technical indicators, does appear to significantly increase performance over randomly picking securities to buy and sell after 6 months, or if the price of the security increase by 10% or more. While this machine learning security selection system does often pick securities who increase by 10% or more within six months, the system does not actually challenge the efficient market hypothesis. For instance, the models are no better than random guessing when configured to predict if a security will ultimately increase or decrease in value after a set amount of time.

Instead, these models all appear to take advantage of volatility in the style of a regression-to-the-mean trading strategy. Specifically, the models do not pick “winners” or avoid “losers” but rather tend to avoid securities which have seen major recent upswings and pick securities which have had rapid downswings. Essentially, the models find their success by picking out securities that are well below their mean and have a low stability. This behavior is not unexpected given the fact the majority of the input data can be viewed as a measure of an element of volatility. 

Future analysis will investigate different fund allocations and portfolio structures, time-series trend analysis in addition to volatility, and investigations into variations of the trading strategy.

