import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
import yfinance as yf

class MeanReversionBollingerBands():
    ''' Class for the vectorized backtesting of Bollinger Bands-based trading strategies.
    '''
    
    def __init__(self, symbol, SMA, dev, start, end, tc = 0):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        SMA: int
            moving window in bars (e.g. days) for SMA
        dev: int
            distance for Lower/Upper Bands in Standard Deviation units
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.SMA = SMA
        self.dev = dev
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
        data['returns'] = np.log(data["Close"] / data["Close"].shift(1))
        self.data = data

        
    def prepare_data(self):
        '''Prepares the data for strategy backtesting (strategy-specific).
        '''
        data = self.data.copy()
        data["SMA"] = data["Close"].rolling(self.SMA).mean()
        data["Lower"] = data["SMA"] - data["Close"].rolling(self.SMA).std() * self.dev
        data["Upper"] = data["SMA"] + data["Close"].rolling(self.SMA).std() * self.dev
        self.data = data
        
    def test_strategy(self):
        ''' Backtests the Bollinger Bands-based trading strategy.
        '''
        data = self.data.copy().dropna()
        data["distance"] = data["Close"] - data["SMA"]
        data["position"] = np.where(data["Close"] < data["Lower"], 1, np.nan)
        data["position"] = np.where(data["Close"] > data["Upper"], -1, data["position"])
        data["position"] = np.where(data["distance"] * data["distance"].shift(1) < 0, 0, data["position"])
        data["position"] = data["position"].ffill().fillna(0)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace = True)
        
        # determine the number of trades in each bar
        data["trades"] = data["position"].diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        data["strategy"] = data["strategy"] - data["trades"] * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
       
        if len(data) > 0:
            perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
            outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
            return round(perf, 6), round(outperf, 6)
        else:
            return None, None
        
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        title = "{} | Moving Average = {} Days | Deviation = {} | Trading Costs = {}".format(self.symbol, self.SMA, self.dev, self.tc)
        plt.figure(figsize=(12, 8))
        plt.plot(self.results["creturns"], label="Buy and Hold")            
        plt.plot(self.results["cstrategy"], label="Strategy")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.savefig('static/images/reversion_strategy_graph.png')    