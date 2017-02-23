from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm


class BaseModel(metaclass = ABCMeta):
    """BaseModel abstract class"""
    @abstractmethod
    def train(self):
        '''train method should train model given data and current time'''
        pass

    @abstractmethod
    def predict(self):
        '''predict method should predict 24 hours into future
        given "forecasted" temperature'''
        pass

    @abstractmethod
    def score(self):
        '''score method should return errors for 96 future predictions'''
        pass

class ShortTermModel(BaseModel):
    """
    Predict next x hours with simple linear regression based on past x hours
    """
    def __init__(self, x):
        super().__init__()
        self._x = x
        self._mask = [1]*(x*4)
        self._mask.extend([0]*(96-x*4))

    def train(self, df, curr_time):
        '''takes in dataframe with temperature and load and trains simple
        linear regression of past x hours'''
        self._curr = curr_time
        # start of training period
        self._start = curr_time - np.timedelta64(self._x, 'h')
        # get training data
        train = df.loc[self._start:curr_time]
        # dummy time variable for linear regression
        train['time'] = np.arange(1,train.shape[0]+1)
        self._train = train
        y_train = train.pop('load').values
        X_train = train.values
        # add constant
        X_train = np.insert(X_train, 0, 1, axis=1)
        # train model
        self._model = sm.OLS(y_train, X_train)
        self._result = model.fit()

    def predict(self, temps):
        # predict 2 hours into future
        self._end = self._curr + np.timedelta64(24, 'h')
        pred_range = pd.date_range(self._start, periods = 96, freq='15Min')
        # get "forecasted" temperatures
        test = temps[pred_range].toframe()
        # dummy time variable
        test['time'] = np.arange(1,test.shape[0]+1)
        # start at next 15min interval
        test_start = self._curr + np.timedelta64(15, 'm')
        X_test = test.values
        X_test = np.insert(X_test, 0, 1, axis=1)
        # predict
        test['predicted'] = self._result.predict(X_test)
        self._test = test
        return test.loc[test_start:,'load']

    def score(self):
        test['actual'] = df.loc[start:end, 'load']


class WeekendModel(BaseModel):
    """
    Polynomial linear regression model for weekend
    To be used from 04:00 Saturday through to 08:00 Monday
    """
    def __init__(self):
        super().__init__()

    def train(self, df, curr_time):
        ''''train on last weekend'''
        self._curr = curr_time
        # start of training period
        self._start = curr_time


class LongRangeModel(BaseModel):
    """
    Predict from 12 hrs away to 24 hrs away
    """
    def __init__(self, arg):
        super().__init__()

    def train(self, df, curr_time):

class Ensemble(BaseModel):
    """
    Putting everything together
    """
    def __init__(self, models):
        super().__init__()
        self._models = models

    def train(self, df, curr_time):
        pass




def main():
    # load data and use first column as datetime index
    df = pd.read_csv('data/equity_RN628A_hbase_data.csv', header=0,
                     names=['timestamp', 'load', 'temp', 'date', 'time', 'dow', 'month'],
                     index_col=0, parse_dates=[0,3])
    df.index.name = None
    # set freq of time series
    ts_range = pd.date_range(df.index.min(), df.index.max(), freq='15Min')
    df = df.reindex(ts_range)
    df.drop(['date', 'time', 'dow', 'month'], axis=1, inplace=True)
    # linearly interpolate temperature
    df['temp'] = df['temp'].interpolate(method='time')
    # cut off time with missing load data
    df.dropna(inplace=True)

    # get temperatures as proxy for forecasted temps
    temps = df.loc[:,'temp']

    # create lag variables
    lagged = df.copy()
    week = df.shift(668)
    lagged['last_week'] = week['load'].values
    day = df.shift(96)
    lagged['yesterday'] = day['load'].values
    df.dropna(inplace=True)

    # create models


    # list of timestamps to 'go through time'
    times = df.index.tolist()
    # data starts at 2012-11-02 07:15:00
    # start forecasting from 2012-11-09 07:15:00
    for t in times[668:]:



if __name__ == '__main__':
    main()
