# Objective

Create load forecasting model that takes temperature into account.

Model should be able to train and then predict next 24 hours.

Both in-sample and out-of-sample validation should be done.


# Data
The kWh data is 15 minute interval data and runs from roughly 11/1/12 to 12/1/13.


# EDA

Please see the notebook for EDA


# Modeling

## Model selection

After doing EDA, it doesn't seem that temperature is the most important feature in predicting load. There are significant periods of time where temperature varies quite dramatically but load remains relatively constant.

It seems like the most significant predictor for what the load will be is what the load was. However, depending on the situation, the lag to use here varies. For example, when trying to predict Saturday, the data from Friday is probably not as useful as the data from last Friday due to the reduced usage and different trends on weekdays vs weekends.

Based on this, I think that having multiple models to predict various periods of time within the 24 hours would be the best approach. It would also be helpful to have separate weekday and weekend models. The most extreme - and potentially best - implementation of this would be to have 192 models: one for every 15 minute interval for the next 24 hours and weekend and weekday versions of each. However, given the time constraints here I will just build a few.

From EDA, it was also clear that there are significant correlations between temperature and the time variables (time of day and month for instance).

In order to incorporate the idea that both temperature and previous load are important features in predicting future load and also include the interactions between all these, I decided to try ensemble tree algorithms with lag variables rather than time series modelling as well as linear regression with dummy variables for seasonality.

### Short-term model

For this model I will just use a simple regression to project at most a few hours into the future using a few hours from the past.

### Weekend model

This model will be trained as a regression on last weekend

## Starting point

In order to accurately make any kind of prediction for a full 24 hours in the future, I think at least 7 days of data is necessary. This means that I will not be doing any test error evaluation on the first week of data. Given that there are over 56 weeks of data, I think the remaining data will still give sufficient opportunities for evaluating test error.

## Issues with temperature

Trying to predict using temperature as a feature poses a problem here because in reality we would only have __forecasted__ temperature but we are given __actual__ temperature.

The question becomes: what data should be used for temperature in order to predict the next 24 hours? Testing the model with future actual temperature data is obviously not an accurate representation of what would happen in reality. However, predicting temperature itself in order to use it as a feature in the model to predict load is also not representative.

Ultimately, I decided to use future actual temperature data to test and validate. Given more time, I would instead hit a weather API and get forecasted temperature for the next 24 hours. The critical assumption here is that temperature forecasts are relatively accurate within a 24 hour period, i.e. forecasted temperature for the next 24 hours would not have differed that much from the actual temperature.


# Validation

## Regression results

Here's an example output of the regression results for the short-term forecaster model. p-value for temperature was not significant generally, and I think this confirms the idea that - at least in short-term - the most important predictor of load is simply past load.

```
                        OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.640
Model:                            OLS   Adj. R-squared:                  0.520
Method:                 Least Squares   F-statistic:                     5.337
Date:                Wed, 22 Feb 2017   Prob (F-statistic):             0.0466
Time:                        23:21:53   Log-Likelihood:                -12.358
No. Observations:                   9   AIC:                             30.72
Df Residuals:                       6   BIC:                             31.31
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const        202.0711    118.000      1.712      0.138       -86.665   490.807
x1            -2.9455      2.330     -1.264      0.253        -8.647     2.756
x2            -0.7250      0.262     -2.772      0.032        -1.365    -0.085
==============================================================================
Omnibus:                        0.534   Durbin-Watson:                   1.851
Prob(Omnibus):                  0.766   Jarque-Bera (JB):                0.147
Skew:                           0.269   Prob(JB):                        0.929
Kurtosis:                       2.679   Cond. No.                     1.53e+04
==============================================================================
```
