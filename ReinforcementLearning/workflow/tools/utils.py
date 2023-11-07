import pandas as pd
import numpy as np

def get_dji():
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    return pd.read_html(url)[1]["Symbol"].to_list()

class Indicators:
    def __init__(self, data):
        self.prices = data["Adj Close"]
        self.high = data["High"]
        self.low = data["Low"]
        self.close = data["Close"]
    
    def get(self):
        macd = self.get_macd(self.prices)
        rsi = self.get_rsi(self.prices)
        cci = self.get_cci(self.high, self.low, self.close)
        adx = self.get_adx(self.high, self.low, self.close)
        v = self.get_turbulence(self.prices)
        return pd.concat([macd, rsi, cci, adx, v], axis=1)

    @staticmethod
    def get_macd(stock_prices):
        ema12 = stock_prices.ewm(span=12, adjust=False).mean()
        ema26 = stock_prices.ewm(span=26, adjust=False).mean()

        MACD = ema12 - ema26
        signal = MACD.ewm(span=9, adjust=False).mean()
        hist = MACD - signal

        MACD.columns = [f"{s}_M" for s in MACD]
        hist.columns = [f"{s}_H" for s in hist]

        return pd.concat([MACD, hist], axis=1)
    
    @staticmethod
    def get_rsi(stock_prices):
        delta = stock_prices.diff()
        gains = delta.where(delta > 0, 0).rolling(14).mean()
        losses = -(delta.where(delta < 0, 0).rolling(14).mean())
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        rsi.columns = [f"{s}_R" for s in rsi]
        return rsi

    @staticmethod
    def get_cci(high, low, close):
        TP = (high + low + close) / 3
        CCI = (TP - TP.rolling(window=20).mean()) / (0.015 * TP.rolling(window=20).std())
        CCI.columns = [f"{s}_C" for s in CCI]
        return CCI

    @staticmethod
    def get_adx(high, low, close):
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        tr = pd.DataFrame(index=high.index, columns=high.columns)
        
        for stock in high.columns:
            tr[stock] = pd.concat([high[stock] - low[stock], 
                                high[stock] - close[stock].shift(), 
                                close[stock].shift() - low[stock]], 
                                axis=1).max(axis=1)
            plus_dm[stock] = np.where((plus_dm[stock] > minus_dm[stock]) & (plus_dm[stock] > 0), plus_dm[stock], 0)
            minus_dm[stock] = np.where((minus_dm[stock] > plus_dm[stock]) & (minus_dm[stock] > 0), minus_dm[stock], 0)
        
        atr = tr.rolling(window=14, min_periods=14).mean()
        smooth_plus_dm = plus_dm.rolling(window=14, min_periods=14).sum()
        smooth_minus_dm = minus_dm.rolling(window=14, min_periods=14).sum()
        
        plus_di = 100 * smooth_plus_dm / atr
        minus_di = 100 * smooth_minus_dm / atr
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=14, min_periods=14).mean()
        adx.columns = [f"{s}_X" for s in high]
        
        return adx
    
    @staticmethod
    def get_turbulence(stock_prices):
        stock_rets = stock_prices.pct_change().dropna()
        mu = stock_rets.mean()
        inv_cov = np.linalg.inv(stock_rets.cov())
        t = pd.Series(index=stock_rets.index)

        for date in stock_rets.index:
            rt = stock_rets.loc[date]
            diff = rt - mu
            vt = np.dot(np.dot(diff, inv_cov), diff)
            t.loc[date] = vt
        
        t.name = "TURB"

        return t
