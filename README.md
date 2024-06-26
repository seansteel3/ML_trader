# Phase 2 ML Trader

<!-- Important Notice -->
## Important Notice

The purpose of this project is to explore applications of machine learning, and the investment setting was chosen due to data availability and interest. Therefore nothing in this repo should be taken as financial advice. In general, past performance cannot predict future success, and small errors in analysis can result in dramatic shifts in expected performance. This repo is academic in nature and should NOT be used as the basis for any legitimate investment strategy.

<!-- TABLE OF CONTENTS -->
## Table of Contents 

* [Summary](#summary)
* [Introduction](#introduction)
* [Methods](#methods)
  * [Data](#data)
  * [Model Construction](#model-construction)
  * [Back Testing](#back-testing)
* [Results](#results)
  * [Data Stationarity](#data-stationarity)
  * [Feature Selection](#feature-selection)
  * [Data Standardization](#data-standardization)
  * [Data Resampling](#data-resampling)
  * [Final Model Construction](#final-model-construction)
  * [Model Explainations: SHAP](#model-explainations-shap)
  * [Individual Returns Analysis](#individual-returns-analysis)
  * [Backtesting: Six Month Portfolio Returns](#backtesting-six-month-portfolio-returns)
  * [Backtesting: Returns Overtime](#backtesting-returns-overtime)
* [Conclusion](#conclusion)
* [Supplementals](#supplementals)
  * [Additional Resources and Readings](#additional-resources-and-readings)
  * [Supplemental Figure](#supplemental-figure)
  * [Run With](#run-with)


<!-- Summary -->
## Summary

Phase 0 of this project explored if trading securities at random could be an effective investment strategy. During Phase 0, portfolios of 15 securities were created by random selection and individual securities were sold if they increased in value by 10% or more, or after six months. Phase 1 aimed to improve this system by adding machine learning (ML) models to increase the chance of correctly choosing securities which increase in value by 10% or more within a six month period. 

While the Phase 1 models tended to make fairly accurate predictions, this accuracy came at the cost of significantly decreased returns when the models were wrong. For instance, if an incorrectly predicted security ended up losing 50% of its value within six months, the models would need to be correct five times just to break even on that one incorrect prediction. Further, incorrectly predicted securities tended to have far lower average returns than securities chosen at random that failed to reach the 10% threshold.

Phase 2 of this project corrects the prediction issue of Phase 1 in two ways. The first was improved data quality and reliability using Financial Modeling Prep (FMP). The second was by building an additional ensemble of models to predict what securities may decrease in value by 30% or more, and therefore should be avoided. Phase 2 also expands the backtesting with a series of simulations which allow for an apples to apples comparison to market indices.

Thanks to FMP’s improved data quality, the models could also leverage around 80 new fundamental ratios and metrics coming from company quarterly reports, instead of relying solely on technical indicators like in Phase 1. Additionally, due to improved hardware (Apple M3 Max 16 core CPU + 64GB integrated memory), Phase 2 was able to formally assess a broader number of model architectures and data resampling and standardization methods in reasonable time. A greater depth and variety of backtests could also be run in reasonable time for Phase 2. 

The improvements in data quality and model optimizations resulted in a more accurate ensemble of models predicting if a security would increase in value by 10% within 6 months (Gain10 models). The fundamental data also allowed for a highly accurate ensemble of models predicting if a security would decrease in value by 30% or more at the end of 6 months (Neg30 models). Unsurprisingly, the Gain10 models do very well during general market upswings, but do very poorly during sustained market downswings. While the Neg30 models do well during sustained market downswings, but have lackluster performance during upswings.

Combining these two accurate model systems into a “Dual Model” strategy successfully mitigated the problems of the Phase 1 models. With the Dual Model system, securities were only purchased if they were predicted to increase by 10% or more within 6 months, and simultaneously predicted not to decrease by 30% or more by the end of 6 months. This system took the best qualities of both ensembles enjoying far more stable gains, and even beat or matched the S&P 500 for the majority of the backtests. However, no models beat the Investco QQQ index for more than a few continuous months during the backtests.


<!-- Introduction -->
## Introduction

Phase 0 of this project simulated Burton Malkiel’s blindfolded monkey from A Random Walk Down Wall Street by selecting securities at random to build an investment portfolio. That analysis roughly upheld Malkiel’s conclusions and also revealed that a small scale “retail investment money” can slightly improve its returns on average if it sells and replaces its randomly chosen securities when they increase in value by 10% or after 6 months. 

Phase 1 of the ML trader project expanded upon the idea of selling securities if they increase in value by 10% or after 6 months, by incorporating machine learning models to predict if a security will increase in value by 10%. The data for phase 1 was sourced from python’s yfinance library and included a series of technical indicators built off of the standard daily OLHCV (open, low, high, close, volume) data for about 5000 tickers. This data was also restricted to only include securities greater than $2 and less than $100 to simulate reasonable investment options for a small scale “retail money investor.” 

The current Phase 2 project addresses several key failings of Phase 1 and reworks the machine learning models accordingly. Phase 2 of the project also removes the “small retail monkey” restriction and enables sales of any security so long as its price is below $800. 

The reliability of OLHCV yfinance data is surprisingly high for a free service, but does end up having significant inconsistencies and unexpected issues particularly when tickers are delisted and later relisted for a new company. To increase reliability of the overall data, the financialmodelingprep (FMP) APIs were used to identify and download OLHCV ~7000 currently tradable securities and another ~5000 delisted securities. Additionally, FMP enabled expansion of the dataset to include a series of about 80 fundamental financial ratios and metrics sourced from company annual and quarterly reports. Further, the backtesting during Phase 1 had a serious error where securities were sold at their highs instead of at their 10% thresholds, which significantly biased the returns upwards. Phase 2 backtesting and the added return analysis corrects this mistake, and extends the simulations further for better apples to apples comparisons to market indices.

Phase 1 models also contained a shortcoming where securities incorrectly predicted to buy tended to lose substantially more value (-35% compared to -20% average returns with random guessing in the test set) on average. Which meant that while the models did tend to be far more accurate than random guessing at picking securities that increased in value by 10%, the gains from this accuracy were almost entirely negated by the increased loss of value when the models were wrong. Phase 2 addresses this problem by building an ensemble of models to predict if securities will drop in value by 30% or more at the end of a 6 month trading period. 

The increased data quality and integration of the new negative model ensemble, results in a system which greatly outperforms random guessing, and roughly matches the market’s overall returns on average during the test date range. 


<!-- Methods -->
## Methods

<!-- Data -->
### Data

The financialmodelingprep (FMP) API was used to source all tradable and delisted tickers on the NYSE, NASDAQ, and ASE using the *available-traded* and *delisted-companies* endpoints respectively. OLHCV (open, low, high, close, volume) data was then collected from the FMP historical-price-full endpoint and a collection of 80 fundamental metrics and ratios were collected from the FMP *key-metrics* and *ratios* endpoints respectively. Duplicated features from these endpoints were removed. All subsequent analysis and training was completed only on data where both quarterly financial data and OLHCV data was available for at least 18 months (1 year required for training, an additional 6 months required for testing).

The OLHCV data was not directly used in any modeling, but rather transformed into a series of common technical indicators to be used as features. Moving averages, Bollinger bands, minmax ratios, daily moving average true ranges, moving historical volatility (standard deviation), stochastic oscillators, MACD, RSI, and AROON indicators were all built with 3 time variations of 21, 84, and 252 previous consecutive trading day periods as proxies for 1, 4 and 12 months. Average directional index was also calculated, but at only the 21 day period, and volume weighted average price (VWAP) was calulated once for each trading day.

The moving averages, Bollinger bands, average true ranges, and MACD were normalized by dividing by the daily close since they are all natively in the scale of the close price. Following normalization, all features were checked for presence of a trend (presence of a unit root) using the Augmented Dickey-Fuller test. Data was then rescaled using SKlearn’s quantile transformer, though SKlearn’s StandardScaler and RobustScaler were tested but were less effective. Finally, the input data was rebalanced using random under sampling, which was the most effective rebalancing method compared to a series of other balancing techniques. Final assessment of all relevant metrics occurred occasionally on balanced validation or test data where appropriate, or otherwise on unbalanced testing data.


| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/63709f52-e8a0-4404-8872-c505a973f78c) |
|:--:| 
| *FIGURE 1: Data Sourcing Pipeline* |

Separate feature selection pipelines were then run on both the models predicting the probability of gaining 10% (Gain10) in value and the models predicting loss of 30% (Neg30) in value to reduce sources of noise and the chance of overfitting. Features were selected by fitting a random forest model on the balanced training data set with max depth of 4 and 250 trees, then keeping features which had a cumulative feature importance between 90-91%. Duplicate features coming from the metrics and ratios APIs were then removed if present.

<!-- Model Construction -->
### Model Construction

Training data for all models included all data from 1/1/2012 to 12/31/2021, subset to the aforementioned best features for each ensemble. Two artificial neural network (ANN) models (one with ReLU activations and one with Tanh), one random forest, one xgboost, and one logistic regression were created to make each ensemble. These models were chosen since they will likely "look" at the data in slightly different ways, which is beneficial when combining their predictions into one final ensemble. Specifically, the ReLU activation of the first ANN will ensure it processes data in a fundamentally different way than the Tanh based ANN. Both ANN’s will process the data differently from the random forest and the boosted trees of XGBoost. While the logistic regression will view the data in a strictly linear fashion, which is substantially different from the other models which will be able to handle non-linearities.

The ANN architectures were optimized in 2 stages. The first stage leveraged Keras hyperband tuner over a grid search of hidden layer sizes and depths across the entire training dataset. While the second stage tested the best three architectures from the hyperband tuner using a 5-fold time series roling cross validation. Meanwhile, tree based model architectures were chosen by a “funneled grid search” over an array of parameters. 

The use of the hyperband tuner as well as the funneled grid search saved significant computational time and burden, while giving the most attention to the most important hyper parameters. The hyperband tuner enabled quick incompletely trained versions of the neural networks to be tested to find the configurations which warranted spending additional time running a full 5-fold rolling cross validation. Meanwhile, the funneled grid search optimized the most influential hyper parameters of the tree based models such as maximum depth and total tree counts first, then optimized the less impactful hyper parameters later.

All models were optimized by the ROC-AUC metric, and used the relevant precision score for any near tie-breakers if needed. The relevant precision score for the Gain10 ensemble was the standard precision = True Positive Class / (True Positive Class + Predicted Positive Class) while the precision score for the Neg30 ensemble was “negative precision” = True Zero Class / (True Zero Class + Predicted Zero Class). 

All final scores and assessments after model optimization was calculated from predictions made on unseen balanced test data in the date range from 1/1/2022 to 10/25/2023. 

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/197c8f72-e7a4-4b74-bf75-e43fb77bc9de) |
|:--:| 
| *FIGURE 2: Full model optimization pipelines. "Funneled Grid Search"* |

Average “voting” was used for the final predictions of each ensemble. A security is chosen as a “buy” with the new system by receiving a prediction of 1 from the Gain10 ensemble and a prediction of 0 from the Neg30 ensemble. This is interpreted as “the security is predicted to increase in value by 10% or more **within** the next 6 months AND predicted not to decrease in value by 30% or more at the **end** of 6 months.”

More stringent criteria to make predictions, such as all model agreement or voting above some threshold, were not used as they were in Phase 1 as this increased the financial cost of incorrect predictions. For example, increasing the prediction threshold of the Gain10 ensemble to be at least 0.75 from 0.5 also decreased the average return when the models were incorrect from -30% to -43%. Additionally chaining together any variation of “stringency” between the Gain10 and Neg30 predictions resulted in a very small set  which could not be used for back testing reliably.

<!-- Backtesting -->
### Backtesting

Backtests compared the trading results of old Phase 1 Gain10 ensemble, the new Gain10 ensemble, new Neg30 ensemble, and the new Gain10 + Neg30 (Dual Model) ensembles. Additionally, a pool of “all winning” securities was tested to see the theoretical maximum returns if the models had 100% accuracy, as well all the returns of investing in random securities as a baseline. Backtests were also run with the S&P 500 and the Invesco QQQ index as market return proxies. All portfolios in backtests, except for the market proxies, held 15 securities initialized with an even allocation of $10,000.

Two backtest methods were used, one included “random start” 6 month portfolios where securities chosen under the above ensemble predictions and/or conditions were purchased, and sold if they increased in value by 10% or all securities sold after 6 months. This generated histograms and average return metrics to see what trading any given 6 month period may look like during the test date range (1/1/2022 to 10/25/2023). 

The second backtest method began all the above ensemble predictions and/or conditions by initializing each of them with 50 random portfolios on 1/1/2022, and plotting out their daily returns until the end of the test range on 10/25/2023. However, since the overall market on 1/1/2022 was at a near all time high, the majority of the time from 1/1/2022 to 10/25/2023 the market is in the red. Therefore this backtest method was repeated with a start date on 10/1/2022 (which is near a local market low) to see how the models perform during a mostly “bullish” market. This backtest method provides the nearest apples-to-apples comparison between market proxies and the machine learning based portfolios.

A final backtest was conducted by rebuilding the models and testing them similar to the 5-fold rolling cross validation split. In this back test, models were built under the 1st training fold then used to predict the next testing fold. For instance, models were initially trained from 1/1/2013 to 12/31/2015 and then used to predict each strategy from 1/1/2016 until 112/31/2017 as a “test” set. The test set was collapsed into a new training set and the next 2 year period was predicted as a “test” set. The predicted unseen test sets were all concatenated into one data frame where predictions were all made on “unseen” data, but over the period from 1/1/2016 until 10/25/2023. 


<!-- Results -->
## Results

<!-- Data Stationarity -->
### Data Stationarity

All features chosen to be included in the final models were tested by the Augmented-Dickey-Fuller test for the presence of a unit root as a check for stationarity (Figure 2). The Augmented Dickey-Fuller test was able to reject the presence of a unit root, and therefore an obvious trend, at the 1% significance level following a bonferroni multiple testing correction. 

Despite the lack a statistical trend overtime, visual inspection of the features does show rather strong seasonality in the scale of months, quarters, or even years across most features. However, in this context of classification forecasting, trends overtime are likely to be more destructive to machine learning models than seasonality, so this type of non-stationary data will not inherently destroy the model’s ability to generalize to unseen data. It will require the models to learn this seasonality in order to perform optimally since no seasonal adjustments or decompositions were performed.


| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/3bd8786a-8e45-40ce-a597-ca8be0a2b8cd) |
|:--:| 
| *FIGURE 3: Augmented Dickey-Fuller test results on included features* |

<!-- Feature Selection -->
### Feature Selection

Features for both the Gain10 and Neg30 models were selected by fitting a random forest with 250 trees and a max depth of 4 over the entire training dataset. The features with a cumulative importance score of <0.91 were kept and the rest discarded. This prevented the chance of overfitting and reduced the risks of running into issues finding a reasonable decision boundary which are associated with the "curse of dimensionality." Figure 3 shows the features importance scores on the final random forest model for illustration. 

Interestingly, the most important features for predicting if a security will increase in value within the next 6 months (Gain10) are almost entirely technical indicators. Even four of the five fundamental indicators are still heavily tied to share prices. Additionally, the most predictive features are also heavily tied to historical volatility. This may likely stem from the fact we are trying not to predict ultimate increases in value, which may be more connected to company fundamentals, but rather transient increases value of at least 10% at least 1 time within the near future. 

Conversely, the most important features for predicting if a security will lose 30% of its value or more at the end of six months contains a much more balanced mixture of technical and fundamental features. Average true range is still the most informative set of features, indicating volatility is still an important aspect, but the inclusion of more fundamental information makes sense when we are trying to predict if a company loses its value at a set time in the future. 

|  ![image](https://github.com/seansteel3/ML_trader/assets/67161057/ce0f94cf-6199-432b-9989-9237c03b26c4=320x320) |
|:--:| 
| *FIGURE 4a: Final feature importances for the Gain10 Ensemble from the final optimized Random Forest model* |

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/3166083a-6c07-4719-852c-380c6386aa6f=320x640)  |
|:--:| 
| *FIGURE 4b: Final feature importances for the Neg30 Ensemble from the final optimized Random Forest model* |

The importance of feature selection and reduction is illustrated by the TSNE projection plots of Figure 4c. Both the Gain10 the Neg30 training data show extremely noisy decision boundaries, which can easily lead to overfitting, particularly if unimportant features are not removed.

These projections also suggest that the Gain10 data is relatively noisier than the Neg30 data and thus the Gain10 models likely will have lower overall accuracy. However since both projections contain significant noise, it is likely both models will not achieve anywhere near perfect accuracy.

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/f3025cbf-556f-4dc9-98eb-321aa0e11156) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/25481c29-62e0-47fb-acd1-685bd7bd64e2) |
|:--:| 
| *FIGURE 4c: TSNE embedding projection plots for balanced Neg30 and Gain 10 training data sets. Purple is class 0, yellow is class 1* |



<!-- Data Standardization -->
### Data Standardization

Due to limited available compute power and its straightforward approach, Phase 1 removed outliers then standardized the data to have a mean of 0 and variance 1 using SKLearn’s StandardScaler. However, its likely the outliers which were removed during Phase 1 were “true” outliers, not errors in the data, and therefore should not have been deleted. 

Thus for phase 2 no outliers were removed and SKLearn’s quantile transformer with either uniform or normal distributions, SKLearn’s RobustScaler, and StandardScaler were tested using a 5-Fold time series cross-validation. All reported metrics were calculated as the mean of the metric across each of the 5-Folds. Data for both the training and validation was rescaled by random under sampling to ensure the baseline value that equates to random guessing for all metrics would be 0.5.

For assessment, a basic ANN (artificial neural network) with 2 hidden layers (10-5), ReLU activation, and the same weight initialization for all experiments was used. This configuration was chosen since ANN’s are highly sensitive to the scale of input data. The primary metric of consideration was the ROC-AUC metric, but the positive precision score and negative precision scores were used in “near tie” cases for the Gain10 and Neg30 models respectively. 

For the Gain10 model, the quantile transformer - normal distribution had both the highest AUC (0.658) and positive precision (68.0%) with the quantile transformer - uniform distribution following closely behind (AUC: 0.654, positive precision: 67.4%).  Meanwhile the Neg30 model had the highest ROC-AUC with the quantile transformer - normal distribution (0.754) but third highest negative precision (75.6%). The RobustScaler had the lowest ROC-AUC (0.698) but highest negative precision (77.7%). 

Since the ROC-AUC score was highest, and the negative precision fairly high, and since the same style transformer was best for the Gain10 model, the quantile transformer - normal distribution was chosen as the rescaling method for both models (fit separately for each ensemble). 

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/61fcdceb-474f-4a8e-b6cc-bc1438964342) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/77b7e513-7161-4f46-95ac-d5bf13dff904) |
|:--:| 
| *FIGURE 5: Metrics of interest for various rescaling techniques for both the Gain10 and Neg30 ensembles* |


<!-- Data Resampling -->
### Data Resampling

Following model architecture optimization, which was conducted with data balanced by random under sampling, various methods to address class imbalance were tested using the average ROC-AUC metric computed across the same 5-fold time series rolling cross validation. The ROC-AUC metric also corresponds to the ROC-AUC of the entire 5 model ensembles (2 ANNs, 1 Random Forest, 1 XGBoost, 1 Logistic regression).

Upsampling of the minority class was done by SMOTE, Borderline SMOTE-1, and ADASYN, while down sampling of the majority class was done only by random under-sampling. One hybrid method of random under-sampling bridging 90% of the class imbalance followed by SMOTE upsampling was also tested. All of these resampling methods were compared to the ROC-AUC scores of unbalanced data.

For the Gain10 ensemble, no resampling (imbalanced data) had the highest ROC-AUC (0.6661) followed extremely closely by random under sampling and SMOTE and the Under-SMOTE hybrid. For the Neg30 ensemble, ROC-AUCs were all very tight with no resampling having the highest score (0.7565). 

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/eb2ca336-a6e3-42c2-9ac1-7770674ce1fb) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/654990df-7c9b-4dad-b5d5-c93ac1afaf29)  |
|:--:| 
| *FIGURE 6: Average ROC-AUC scores from 5-fold time series rolling cross validation for various resampling strategies for the Gain10 and Neg30 ensembles* |


Since no resampling requires calibration of the threshold to denote a prediction as class 1 or class 0, the thresholds for each of the 5 rolling cross validation folds were explored. Using the average ROC curve threshold and the average proportion of class 1 in the training data are commonly used methods of setting the threshold to denote a predicted class 1. These two methods were compared against each other and to random under sampling as a control. For random under sampling the training mean threshold is always >= 0.5 results in class 1, while the ROC thresholding will oscillate slightly around 0.5.

For the Gain10 ensemble, the overall average accuracy across all 5 rolling cross validation folds was highest when setting the threshold using the imbalanced average ROC threshold (imbalanced ROC Accuracy = 66.16%). Meanwhile, the balanced ROC and balanced training mean thresholds all hovered around 64%. However, in the case of predicting if a security will increase in value, the positive precision score is more important than overall accuracy and in this case using the balanced data’s ROC threshold had the highest precision (74.91%) vs the imbalanced data’s ROC threshold (74.02%). 


| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/bb8ce620-3b89-488e-9709-730ba43b15ba) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/ff390135-ddd3-4a75-8949-0ef1ed5c2063)|
|:--:| 
| *FIGURE 7a: Average accuracy and postive precision for the Gain10 ensemble from 5-fold time series rolling cross validation* |

For the Neg30 ensemble the overall average accuracy across all 5 rolling cross validation folds was also highest setting the threshold using the imbalanced average ROC threshold (imbalanced ROC Accuracy = 74.12%). However, in the case of predicting if a security will decrease in value, negative precision is more important than overall accuracy, and using the imbalanced data’s training mean to set the threshold for class 1 resulted in the highest negative precision (90.63%) compared to the imbalanced ROC threshold (86.72%) with the balanced training mean not far behind in second place (90.26%).

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/61a1265e-3c0a-4852-a4ff-d3e3146ce274) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/eae7f2fa-acfe-4ca6-b8d2-cb2c93652d6a)|
|:--:| 
| *FIGURE 7b: Average accuracy and negative precision for the Neg30 ensemble from 5-fold time series rolling cross validation* |

Additionally, setting thresholds for class 1 prediction using either imbalanced training data’s mean, or using ROC threshold averages can be noisy and can vary overtime. Figure 8 below shows the Gain10 and Neg30 ensemble threshold values for each of the 5 rolling cross validation folds with light blue bars equal to fold 1, orange fold 2, green fold 3, red fold 4, and purple fold 5. Since only the balanced training mean’s thresholds are perfectly constant overtime, and since the balanced training mean’s respective precision scores were nearly as high or better than any alternatives, the data for the final models was balanced by random under-sampling. Random under-sampling was also chosen over the slightly better performing SMOTE-Undersampling hybrid due to its massively lower computational resource requirement. 

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/c62cc6a6-5b62-4d8b-87a6-d392b1c4d429) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/ccd86737-d823-4b9d-b349-a459a3bffa5b)|
|:--:| 
| *FIGURE 8: Average thresholds to declare class 1 for the Gain10 and Neg30 ensembles for each of the 5 rolling cross validation folds (fold 1 is in blue, fold 2 in orange, fold 3 in green fold 4 in red, fold 5 in purple).* |

<!-- Final Model Construction -->
### Final Model Construction

The final models were built with data transformed into a normal distribution centered around 0 with the QunatileTransformer - Normal distribution. Class imbalance was removed by random under sampling of the majority class, and final architectures optimized by the “funneled grid search” shown in figure 2 above. Training of the final models was done by collapsing all 5-fold time series rolling cross validation splits into 1 training set, then the following final metrics below were calculated twice, once on balanced data and once on imbalanced unadjusted data, using unseen test data ranging (1/1/2022 to 10/25/2023). 

The final Gain10 ensemble saw an accuracy of 64.37% over balanced test data and 70.61% over the full imbalanced data. Its precision was 61.99% on balanced data and 78.09% on imbalanced data. In the imbalanced data, 69.7% of observations were class 1, making a precision of 78.09% represent 8.39 percentage point improvement over random guessing. Its negative precision is rather unimpressive (51.84%) but also not of concern in the setting where it will be used to predict securities to buy (predicted class 1). Its overall ROC-AUC score was 0.70.

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/d4f63ecb-d244-439c-a130-a4d33e4c0f13) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/052334df-a29d-47ec-9371-b7951c2dddf7) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/806532c8-329f-479e-8684-4b01bbeddd52)|
|:--:| 
| *FIGURE 9a: Gain10 ensemble confusion matrices for balanced and imbalanced data, plus the plotted ROC curve (dashed red line represents useless model and/or random guessing)* |

The final Neg30 ensemble saw 70.7% accuracy over balanced data and 62.14% accuracy over the full imbalanced dataset. Its negative precision was 74.87% on balanced data and 91.9% on imbalanced data. In the imbalanced data 73.6% of the observations were class 0, making a 91.9% negative precision represent a 18.3 percentage point improvement over random guessing. Its positive precision is sub optimal (40.03%) but also not of concern in the setting where it will be used to predict what securities to buy (predicted class 0). Its overall ROC-AUC score was 0.78.


| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/7c842f91-134f-42fd-b397-aacf24c96ebb) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/1bbf571b-efd6-456b-a015-f22a11b9dc87) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/cb797a72-e611-411c-bbac-bf6d4903f4c7) |
|:--:| 
| *FIGURE 9b: Neg30 ensemble confusion matrices for balanced and imbalanced data, plus the plotted ROC curve (dashed red line represents useless model and/or random guessing)* |

<!-- Model Explainations: SHAP -->
### Model Explainations: SHAP

SHAP bee-swarm plots were generated for the final random forest model in each ensemble to get an idea of how the features drive the final outputs. For both the Gain10 and Neg30 models, higher volatility, as measured by the average true ranges (ATRs) and historical volatilities (VOLs), results in a less likely chance of being predicted for class 1. 

The fact that favorable predictions (Gain10 class 1, Neg30 class 0) from the models work in opposition is not surprising since the models predict effectively opposite outcomes. However, it is spurring that the desired prediction of class 1 from the Gain10 models comes from securities with **lower** volatility, while the desired prediction of class 0 from the Neg30 models comes from **higher** volatility.  It appears in this data, when controlling for other factors, higher volatility is associated with decreased chance of increasing in value by 10% within six months, and decreased chance of losing 30% of value at the end of six months. 

The fact favorable predictions from Gain10 and Neg30 are based on opposite levels of volatility also explains why "stringent" criteria for each model could not be combined. Since Gain10 is looking for low volatility and Neg30 looking for high volatility, there are very few securities that both models could simultaneously be highly confident on (greater than 80% confidence). For example, if a security has high volatility, its likely the Neg30 models would confidently classify it as a "buy" (class 0), but since Gain10 looks for *low* volatility it is **unlikely** it would simultaneously classify that security confidently as a buy (class 1).

The Gain10 models also unsurprisingly appear favor securities that are not near the upper 84 day and 21 day Bollinger bands, but rather near the lower 84 day Bollinger bands. Additionally the Neg30 models unsurprisingly predict that companies with higher return on tangible assets, higher price to equity ratios, and higher free cash flow are more likely to not lose 30% of their value.

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/7e6e55b1-1362-4911-b1aa-38aa4e87c777) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/10a23840-2127-4d6c-94ac-3600b4d905e3) |
|:--:| 
| *FIGURE 10: SHAP Beeswarm Plots for Gain10 and Neg30 **final** Random Forest Models* |

<!-- Individual Returns Analysis -->
### Individual Returns Analysis

The predictions of the final model ensembles were used to create individual investment return distributions. The “Randomly Chosen” returns were built by randomly sampling over the entire test set. The “Gain10 Only” returns were built by randomly sampling over securities predicted as a buy (predicted class 1) by the Gain10 ensemble. The “Neg30 Only” returns were built by randomly sampling over securities predicted as a buy (predicted class 0) by the Neg30 ensemble. Finally the “Dual Model” returns were built by sampling over securities predicted to buy by both models (predicted 1 from Gain10 AND predicted 0 from Neg30).

All returns were plotted twice, once with all returns in the strategy graphed, and once “zoomed in” where only returns of less than 10% are graphed. The first plot gives a **holistic** view of the individual returns from each strategy, while the second gives a view of what happens when the strategy of “selling when the security increases by 10%” **fails**.

Both the randomly chosen security (ie: no ML model) and the Gain10 ensemble only models have a slightly negative return of -0.16% and -0.1% respectively. The Gain10 model succeeds in picking far more “winning securities” than random guessing does. However, those “wins” are almost entirely lost by the fact that the average return of **failed** random guesses is just -15% while its almost -21% for those predicted to buy from Gain10 models. Essentially the Gain10 ensemble succeeds at predicting more "winners", but at the cost of substantially decreasing the returns of “losers.”

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/ed1446dc-0b85-47e2-b96e-49ca61a726a1) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/23751d75-bdb2-4e8b-854d-f9442423ff17) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/e2ebc15a-3383-409c-99cb-0f7752e1eaf4) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/d20e64c7-f84c-4ca2-91cf-705e0987cb4e) |
|:--:| 
| *FIGURE 11a: Average per-investment returns for Random Guessing and Gain10 Only strategies. The (<10%) histograms showcase returns when the models "fail"* |

Meanwhile the Neg30 only ensemble does fairly well with +0.43% overall average returns despite the fact it picks “winners” at a lower rate than random guessing. The "point of failure" histogram shows the relative success of the Neg30 ensemble is because it weeds out nearly all -90% return securities, and dramatically reduces the number of -30% or more securities. Therefore, the losses when choosing a “loser” are not nearly as bad (-8.5%) as the Gain10 ensemble or random guessing, allowing it to perform better.

Combining both the Gain10 and Neg30 predictions into a Dual Model ensemble has the desired effect of leveraging the strengths of both ensembles and minigating the weaknesses. The Dual Model strategy performs the best with the average returns of +1.57% per investment over the whole training data and only -12.95% when it picks “losers.” 


| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/55515114-887e-45d8-9a3e-2013b12cd868) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/8fa4ddb2-30b2-4c61-800f-ea3843e866df) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/9a94bda6-2e33-4c46-96ef-24d30cf1b7f6) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/6e889f7f-c48f-4477-ab6c-c77eba213af4) |
|:--:| 
| *FIGURE 11b: Average per-investment returns for Neg30 Only and Dual Model strategies. The (<10%) histograms showcase returns when the models "fail"* |

<!-- Backtesting: Six Month Portfolio Returns -->
### Backtesting: Six Month Portfolio Returns 

To assess the actual performances, these these strategies were used to build a $10,000 portfolio evenly distributed across 15 securities. During the simulation securities were sold if they increased in value by 10%, or after 6 months. All securities were “sold” 6 months from start time get an assessment of how the portfolio performed. Each strategy (random guessing, Gain10 predictions, Neg30 predictions, Dual Model predictions, Old Phase 1 models, and All Win) was simulated 1,500 times to build the below histograms. All Win represents a “perfect” scenario with 100% accuracy for predicting 10% gains in value at some point in 6 months.

The random choice strategy returned an average of -1.73% with a 57.6% chance of negative portfolios, while the perfect All Win strategy returned an average of 45.95% (negative chance is an artifact of ceasing trading after 6 months where some securities decreased in value and would have recovered, but were not given the chance in this simulation). The phase 1 models are an “applies to oranges” comparison since they were actually trained over this timeframe (ie: this data is NOT testing but rather training). Despite this fact, the old models perform poorly with a -3.3% and 62.5% negative return chance.

The new Gain10 ensemble actually performs the worst with -4.47% and 61.9% negative return chance, likely because of the fact its “failure” picks tend to return far more negative results. However, the Neg30 ensemble nearly breaks even only -0.81% mean and 53.7% chance of negative returns. Finally the Dual model performs by far the best, with positive returns of +0.39% on average and less than 46.6% negative return chance. 

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/d5bd46e6-308e-4162-9eb9-1f13819a898d) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/1b08363e-052a-40ca-8495-8d26baad9ae4) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/206cc657-6399-4777-b9f9-b431116babb3) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/bd2615fb-f387-48f7-b6c6-0e44b89e3c67) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/59437c3c-80d0-4960-bab3-673f2b77b1bb) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/e17bf980-a0be-4ef7-bcec-f01915510c9b) |
|:--:| 
| *FIGURE 12: 6 month simulated returns with random start dates for Random Guessing, All Win, Old Gain10 ensemble, New Gain10 ensemble, Neg30 ensemble, and Dual Model ensemble strategies* |


<!-- Backtesting: Returns Overtime -->
### Backtesting: Returns Overtime

To test how the various ML (machine learning) and non-ML powered trading strategies compare to investing in the overall market, each strategy was simulated 50 times with an initial investment date of 1/1/2022 and a final date of 10/25/2023. The value of each portfolio was checked using the daily close prices and compared to the S&P 500 and to the Invesco QQQ index as market performance proxies. 

The perfect accuracy scenario of 100% of securities increase in value by 10% sometime within  six months unsurprisingly performs exceptionally well with over 200% returns during the timeframe. However the market itself during this range about breaks even and is “in the red” during the majority of the timeframe. Also unsurprisingly, all of the other investment conditions perform much closer to the market than they do the theoretical maximum.

The Random Choice (ie: No ML) strategy tends to slightly underperform the market proxies the entire time range, until the end where is ends significantly below. Interestingly, a similar but slightly worse situation occurs with the old models, despite the fact the old models are technically predicting over training data while the rest are predicting over testing data. 

Meanwhile the new Gain10 only predicted portfolios perform by far the worst of any strategy throughout the timeframe. This is directly due to the fact that when Gain10 ensemble makes an error, it tends to choose securities which perform significantly worse than the average. Specifically, average time for a security to increase in value by 10% is 33 days and 80% sell within 60 days, while the time low performing securities are held is the full six months. Therefore, as low performing securities accumulate the portfolio, they tend to erode all the gains made by positive predictions. This can be seen by the jumps in value at roughly 6 month intervals where the portfolios shed their "losers" and have a chance to replace them with "winners." These 6 month intervals also roughly correspoind with the market and other strategies' increases around July 2022 and mid January 2023, but shortly after these jumps the Gain10 Only portfolios quickly start losing value far faster than any other strategy as "losers" begin to creep back in.

The Neg30 only and the Dual model predicted portfolios end at about the same value as both market indices, but the Neg30 model outperforms all strategies and the market for the majority of the time range. Since the Neg30 ensemble tends to filter out the worst performing securities, and since the market performance during this range is mostly negative, the Gain30 only models tend to perform the best. Meanwhile the Dual model portfolios tends to do well during upswings, but still suffers more from the downward draw from incorrect predictions than the Neg30 only portfolios. 


| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/783c603c-c0d9-4fde-b0d2-09fa3eb0f4b6) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/33bcb638-cfaf-44ba-912a-1bf9508590c4) |
|:--:|  
| *FIGURE 13: All stratgey average returns from 1/1/2022 to 10/25/2023. Plot 1 includes the All Win theoretical maximum, while plot 2 excludes it* |

To test wether or not the generally negative performance of the market during the full testing range was the root cause of the Neg30 only portfolios’ relative strong performance, the simulations were repeated during a generally positive portion of the test range (10/1/2022 to 10/25/2023). During this timeframe the Gain10 only, Old model and Random Choice portfolios all perform the worst as before, but the Dual model now closely tracks the S&P 500 average returns. Additionally, as expected, the Neg30 only portfolios now perform rather lack luster as the frequency of poorly performing securities drops, and therefore the Neg30’s competitive advantage evaporates. 

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/c8cac5ba-7d78-4c9e-a599-e363294f98a2) |
|:--:|  
| *FIGURE 14: All stratgey average "postive market" returns from 10/1/2022 to 10/25/2023* |

Finally, to explore how periodic retraining, as well as how these strategies and models perform over a longer period of time, models were sequentially retrained and tested across the entire date ranges with the same time breaks as the 5-fold time series rolling cross validation. This enabled all data from 1/1/2016 to 10/25/2023 to act as "test" data for trading simulations.

This final experiment confirms that the Gain10 model ensemble does in fact do very well during major market upswings, but as expected, performs by far the worst during sustained market down swings. Meanwhile the Neg30 model ensemble does not catch much of the upswings, but does tend to resist losing value during sustained market drops. The Dual market ensemble takes the “best of both worlds” from these models and can see significant increases in value during bull markets, while seeing a dampened loss in value during bear markets. The Gain10, and by extension the Dual model portfolios, outperform the random choice portfolios and surprisingly also outperform the S&P 500 for the majority of the time range. However unsurprisingly, they both underperform the Investco QQQ from about June 2019 until the end of the simulations.

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/2596c7c8-2ea3-47cd-a6b1-7df4befb57b3) |
|:--:|  
| *FIGURE 15: All Market returns from 1/1/2016 to 10/25/2023* |

<!-- Conclusion -->
## Conclusion

The tendency of the Gain10 ensemble to choose highly volatile securities was a double edged sword. While the models produced fairly accurate predicted probabilities for a security to increase in value by 10% or more within 6 months, the accuracy came at the cost of substantially decreased average returns when the models were incorrect. 

The improved data quality and additional fundamental features provided by Financial Modeling Prep (FMP) substantially increased the accuracy of the Gain10 ensemble. Additionally, the added features enabled the creation of accurate “Neg30” models to predict if a security will lose 30% of its value at the end of six months. The combination of the Gain10 and Neg30 ensembles into a “Dual Model” strategy successfully mitigated a substantial amount of the risk when using the original Gain10 models on their own. 

Impressively, backtests indicate that the Dual Model strategy has the potential to at least match, if not outperform the S&P 500 when used optimally. However, the noisy decision boundaries provided by the features, and the lack of ability to consistently match the Investco QQQ index’s returns, indicate that improvements or expansions in the data could likely further enchance the models' performance. 

Future phases of this project will explore adding additional data cleaning and data mining for additional features. Future phases may also experiment with recurrent neural networks and transformers to generate meaningful embeddings of integrated quarterly report fundamental data with security OLHCV data.

<!-- Supplementals -->
## Supplementals

<!-- Additional Resources and Readings -->
### Additional Resources and Readings
* [*A Random Walk Down Wallstreet*](https://en.wikipedia.org/wiki/A_Random_Walk_Down_Wall_Street)
* [*Financial Modeling Prep*](https://site.financialmodelingprep.com/developer/docs)
* [*Augmented Dickey Fuller Test and Stationary Data*](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
* [*The Curse of Dimensionality*](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
* [*What is Overfitting?*](https://www.datacamp.com/blog/what-is-overfitting)
* [*Feature Selection Basics*](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)
* [*Timeseries Cross Validation*](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4)
* [*Introduction to Neural Networks*](https://datascientest.com/en/dense-neural-networks-understanding-their-structure-and-function)
* [*Activation Functions*](https://www.v7labs.com/blog/neural-networks-activation-functions)
* [*Random Forest Models*](https://builtin.com/data-science/random-forest-algorithm)
* [*XGBoost*](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
* [*Logistic Regression*](https://www.ibm.com/topics/logistic-regression)
* [*Model Ensembling Basics*](https://medium.com/@datasciencewizards/guide-to-simple-ensemble-learning-techniques-2ac4e2504912)
* [*SHAP Analysis*](https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability)
* [*Basics of ROC-AUC analysis*](https://medium.com/@shaileydash/understanding-the-roc-and-auc-intuitively-31ca96445c02)
  

<!-- Run With -->
### Run With

* Python 3.11.5
* Pandas 2.1.4
* Numpy 1.24.3
* SKLearn 1.3.0
* Imblearn 0.11.0
* Tensorflow 2.15.0
* XGBoost 2.0.3

<!-- Supplemental Figure -->
### Supplemental Figure

| ![image](https://github.com/seansteel3/ML_trader/assets/67161057/106f1441-a5fe-417a-81d8-58290b6217f7) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/86e2e675-8bcc-4968-a6d7-24113a5c0995) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/f9cb51dd-fbd4-4181-af02-24fd5490a2f3) ![image](https://github.com/seansteel3/ML_trader/assets/67161057/0c0cb43c-363b-4cd7-972c-052a81fd8c06) |
|:--:|  
| *Supplemental Figure: Typical expected return ranges for each strategy from all 50 simulated portfolio returns between 1/1/2022 to 10/25/2023* |




