from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import sys


class BaseModel(metaclass = ABCMeta):
    """BaseModel abstract class"""
    @abstractmethod
    def train(self):
        '''train method should train model given data and current time
        also returns training error'''
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


class WeekendModel(BaseModel):
    """
    Polynomial linear regression model for weekend
    To be used from 04:00 Saturday through to 08:00 Monday
    """
    def __init__(self, p):
        super().__init__()
        self._p = p

    def train(self, train):
        '''given dataframe of training values, train linear model'''
        y_train = train.pop('load').values
        # dummy time variable for linear regression
        timevars = ['time']
        train['time'] = np.arange(1,train.shape[0]+1, dtype=np.int64)
        for p in np.arange(2, self._p):
            colname = 'time^{}'.format(p)
            timevars.append(colname)
            train[colname] = np.power(train['time'], p)
        timevars.extend(['temp', 'hour'])
        X_train = train.loc[:,timevars].values
        # add constant
        X_train = np.insert(X_train, 0, 1, axis=1)
        self._model = sm.OLS(y_train, X_train)
        self._results = self._model.fit()
        print(self._results.summary())
        return mean_squared_error(y_train, self._results.predict(X_train))**0.5

    def predict(self, test):
        '''given dataframe of test values, predict next 24 hours'''
        timevars = ['time']
        test['time'] = np.arange(1,test.shape[0]+1, dtype=np.int64)
        for p in np.arange(2, self._p):
            colname = 'time^{}'.format(p)
            timevars.append(colname)
            test[colname] = np.power(test['time'], p)
        timevars.extend(['temp', 'hour'])
        X_test = test.loc[:,timevars].values
        # add constant
        X_test = np.insert(X_test, 0, 1, axis=1)
        self._y_pred = self._results.predict(X_test)
        return self._y_pred

    def score(self, y_test):
        '''return RMSE of predictions'''
        return mean_squared_error(y_test, self._y_pred)**0.5


class WeekdayModel(BaseModel):
    """
    RandomForest regressor for weekdays
    To be used from 8:15 Monday through 3:45 Saturday"""
    def __init__(self):
        super().__init__()
        self._model = RandomForestRegressor()

    def train(self, train):
        '''given dataframe of training values, train RF regressor'''
        y_train = train.pop('load').values
        train.drop('wknd', axis=1, inplace=True)
        X_train = train.values
        self._model.fit(X_train, y_train)
        print('''                            RandomForest Regression Results\n
==============================================================================
Feature importances: \n''')
        pprint(train.columns[np.argsort(self._model.feature_importances_)[::-1]].tolist())
        print("----------------------------\n")
        return mean_squared_error(y_train, self._model.predict(X_train))**0.5

    def predict(self, test):
        '''given dataframe of test values, predict next 24 hours'''
        X_test = test.values
        self._y_pred = self._model.predict(X_test)
        return self._y_pred

    def score(self, y_test):
        '''return RMSE of predictions'''
        return mean_squared_error(y_test, self._y_pred)**0.5


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

    # create lag variables
    df['last_week'] = df['load'].shift(668)
    df['yesterday'] = df['load'].shift(96)
    for i in np.arange(1,13):
        lag_name = 'lag-{}'.format(i)
        df[lag_name] = df['load'].shift(i)
    # remove first week that does not have sufficient data for all lag variabless
    df.dropna(inplace=True)

    # create time variables
    df['dow'] = df.index.dayofweek
    df['hour'] = df.index.hour

    # create weekend variable
    holidays = ['2012-11-22', '2012-11-23',
                '2012-12-24', '2012-12-25', '2013-01-01',
                '2013-05-27', '2013-07-04',
                '2013-11-28', '2013-11-29']
    df['wknd'] = np.logical_or.reduce(((df['dow'] == 6).values,
                    np.logical_and(df['dow']==5, df['hour']>3).values,
                    np.logical_and(df['dow']==0, df['hour']<8).values,
                    np.in1d(df.index.strftime("%Y-%m-%d"), holidays))).astype(int)

    # create models
    weekday_model = WeekdayModel()
    weekend_model = WeekendModel(5)

    # stdout logging
    saveout = sys.stdout
    fsock = open('out.log', 'w')
    sys.stdout = fsock
    # list of timestamps to 'go through time'
    times = df.index.tolist()
    # iniitialize values
    last_trained = times[672]
    train = df.loc[:last_trained]
    train_weekday = train.loc[df['wknd'] == 0]
    train_weekend = train.loc[df['wknd'] == 1]
    weekday_model.train(train_weekday)
    weekend_model.train(train_weekend)
    test_errors = []
    train_errors = []
    for t in times[673:-3]:
        # predict 24 hours into future - using actual temp in lieu of forecasted
        test = df.loc[t+np.timedelta64(15, 'm'):t+np.timedelta64(24, 'h')]
        y_test = test.pop('load')
        # pop weekend mask values
        wkday_mask = test.pop('wknd').values
        wkend_mask = 1-wkday_mask
        ## PREDICT only weekday
        if (wkday_mask == 0).all():
            weekday_model.predict(test)
            test_errors.append(weekday_model.score(y_test))
        ## PREDICT only weekends
        elif (wkday_mask == 1).all():
            weekend_model.predict(test)
            test_errors.append(weekend_model.score(y_test))
        ## PREDICT using both
        else:
            wkday_pred = ma.array(weekday_model.predict(test), mask=wkday_mask)
            wkend_pred = ma.array(weekend_model.predict(test), mask=wkend_mask)
            y_pred = wkday_pred + wkend_pred
            test_errors.append(mean_squared_error(y_test, y_pred.data)**0.5)
        '''retrain afer 24 hours'''
        if (t - last_trained) == np.timedelta64(1, 'D'):
            print(t.strftime("%a, %b %d, %Y"))
            start = last_trained - np.timedelta64(8, 'D')
            # retrain weekday model
            if df.loc[t, 'wknd'] == 0:
                train = df.loc[start:t]
                train_weekday = train.loc[df['wknd'] == 0]
                train_errors.append(weekday_model.train(train_weekday))
            # retrain weekend model
            else:
                train = df.loc[start:t]
                train_weekend = train.loc[df['wknd'] == 1]
                train_errors.append(weekend_model.train(train_weekend))
            last_trained = t
    # finish logging
    sys.stdout = saveout
    fsock.close()
    '''SOME RESULT PLOTS'''
    plt.style.use('ggplot')
    dates = pd.date_range('2012-11-16', periods=len(train_errors), freq='D')
    trainerrdf = pd.DataFrame(train_errors, dates, ['training'])
    times = pd.date_range('2012-11-16', periods=len(test_errors), freq='15Min')
    testerrdf = pd.DataFrame(test_errors, times, ['testing'])
    errdf = testerrdf.join(trainerrdf)
    errdf.fillna(method='ffill', inplace=True)
    errdf['dow'] = errdf.index.dayofweek
    months = ['2012-11', '2012-12']
    months.extend(['2013-{}'.format(m) for m in range(1,13)])
    f, axes = plt.subplots(14, sharey=True)
    for i, ax in enumerate(axes):
        current = errdf.loc[months[i]]
        current[['testing', 'training']].plot(ax=ax, title=months[i])
        ax.fill_between(current.index, current['testing'], where = current['dow'] > 4,
                    facecolor = 'red', alpha = 0.5, label='weekend')
        ax.legend()
    f.set_size_inches(13, 60)
    plt.tight_layout()
    return test_errors, train_errors


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    test_errors, train_errors = main()
