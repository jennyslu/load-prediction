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

After doing EDA, it doesn't seem that temperature is the most important feature in predicting load. There are significant periods of time where temperature appears to vary quite dramatically but load appears to remain relatively constant.

It does seem clear that the most significant predictor for what the load will be is

I recognized that having multiple models to predict various periods of time within the 24 hours would be most useful. Predicting load 15 minutes in the future

The most extreme - and potentially best - implementation of this would be to have 96 models: one for every 15 minute interval for the next 24 hours. However, given the time constraints here I will just build a few.

There are also clear interactions between some of the variables. Temperature and time of day, as well as month are related to each other and including these interaction variables will probably be important.

In order to incorporate the idea that both temperature and previous load are important features in predicting future load
