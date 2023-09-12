import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
import yfinance as yf
from datetime import datetime
import os

class MovingAverageCrossover():
    ''' Class for the vectorized backtesting of SMA-based trading strategies.
    '''
    
    def __init__(self, symbol, SMA_S, SMA_L, start, end, tc = 0):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
        start: str
            start date for data import
        end: str
            end date for data import
        '''
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None 
        self.get_data()
        self.prepare_data()
        
    def get_data(self):
        ''' Imports the data from Yahoo Finance.
        '''
        data = yf.download(self.symbol, start=self.start, end=self.end)
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        self.data = data

        
    def prepare_data(self):
        '''Prepares the data for strategy backtesting (strategy-specific).
        '''
        data = self.data.copy()
        data["SMA_S"] = data["Close"].rolling(self.SMA_S).mean()
        data["SMA_L"] = data["Close"].rolling(self.SMA_L).mean()
        self.data = data
          
    def test_strategy(self):
        ''' Backtests the SMA-based trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"] - self.tc * data["position"].diff().fillna(0).abs()
        data.dropna(inplace=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        print(data)
       
        if len(data) > 0:
            perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
            outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
            return round(perf, 6), round(outperf, 6)
        else:
            return None, None
    
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | Short Moving Average = {} Days | Long Moving Average = {} Days | Trading Costs = {}".format(self.symbol, self.SMA_S, self.SMA_L, self.tc)
            plt.figure(figsize=(12, 8))
            plt.plot(self.results["creturns"], label="Buy and Hold")
            plt.plot(self.results["cstrategy"], label="Strategy")
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Cumulative Returns")
            plt.legend()
            project_dir = os.path.expanduser('/home/strategyalchemist/mysite')
            image_path = os.path.join(project_dir, 'static/images/strategy_graph.png')
            plt.savefig(image_path)