import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class TrendFollowingMACD():
    def __init__(self, symbol, short_period, long_period, signal_period, start, end, tc=0):
        self.symbol = symbol
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        self.prepare_data()
        
    def get_data(self):
        data = yf.download(self.symbol, start=self.start, end=self.end)
        data['Close'] = pd.to_numeric(data["Close"], errors= 'coerce')
        data['returns'] = np.log(data["Close"] / data["Close"].shift(1))
        self.data = data
        
    def prepare_data(self):
        data = self.data.copy()
        data['ShortEMA'] = data['Close'].ewm(span=self.short_period, adjust=False).mean()
        data['LongEMA'] = data['Close'].ewm(span=self.long_period, adjust=False).mean()
        data['MACD'] = data['ShortEMA'] - data['LongEMA']
        data['SignalLine'] = data['MACD'].ewm(span=self.signal_period, adjust=False).mean()
        self.data = data
        
    def test_strategy(self):
        data = self.data.copy().dropna()
        data['position'] = np.where(data['MACD'] > data['SignalLine'], 1, -1)
        data['strategy'] = data['position'].shift(1) * data['returns']
        
        data.dropna(inplace=True)
        data['trades'] = data['position'].diff().fillna(0).abs()
        data['strategy'] = data['strategy'] - data['trades'] * self.tc
        
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        
        if len(data) > 0:
            perf = data['cstrategy'].iloc[-1]
            outperf = perf - data['creturns'].iloc[-1]
            return round(perf, 6), round(outperf, 6)
        else:
            return None, None
        
    def plot_results(self):
        title = "{} | Short EMA = {} | Long EMA = {} | Signal Period = {} | Trading Costs = {}".format(
            self.symbol, self.short_period, self.long_period, self.signal_period, self.tc)
        plt.figure(figsize=(12, 8))
        plt.plot(self.results["creturns"], label="Buy and Hold")
        plt.plot(self.results["cstrategy"], label="Strategy")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.savefig('static/images/macd_strategy_graph.png')
