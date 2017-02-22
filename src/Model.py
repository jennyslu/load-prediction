from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BaseModel(metaclass = ABCMeta):
    """docstring for BaseModel."""
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def score(self):
        pass


class ClassName(object):
    """docstring for ."""
    def __init__(self, arg):
        super(, self).__init__()
        self.arg = arg


def main():
    # load data and use first column as datetime index
    df = pd.read_csv('data/equity_RN628A_hbase_data.csv', index_col=0, parse_dates=[0,3])
    # linearlly interpolate missing temperature values
    df['actual_temperature'] = df['actual_temperature'].interpolate()

if __name__ == '__main__':
    main()
