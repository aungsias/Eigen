import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from ..utils import standardize
from ..ou_model import OUModel

class WalkForwardStatArb:
    def __init__(self, market_returns, stock_returns, lookback, reversion_window=30, entry_th=1.25, exit_long_th=.5, exit_short_th=.75):
        self.stock_returns = stock_returns
        self.market_returns = market_returns
        self.stocks = stock_returns.columns
        self.etfs = self.market_returns.columns
    
        self.s_scores = pd.DataFrame(columns=stock_returns.columns, index=stock_returns.index[lookback:])
        self.betas = pd.DataFrame(columns=stock_returns.columns, index=stock_returns.index[lookback:])
        self.signals = pd.DataFrame(columns=stock_returns.columns, index=stock_returns.index[lookback:])
        self.weights = pd.DataFrame(columns=stock_returns.columns, index=stock_returns.index[lookback:])

        self.trading_days = 252
        self.k_threshold = self.trading_days / reversion_window
        self.lookback = lookback
        self.entry_th = entry_th
        self.exit_long_th = exit_long_th
        self.exit_short_th = exit_short_th

        self.param_memory = {}
    
    def run(self, progress=True):
        self.get_weights(progress_bar=progress)
        self.s_scores = self.s_scores.astype(float)
        self.signals = self.signals.astype(int)
        self.returns = pd.concat([self.stock_returns, self.market_returns], axis=1)
    
    def get_weights(self, progress_bar=True):
        
        if self.signals.isna().all().all():
            self.generate_signals(progress_bar=progress_bar)

        if progress_bar:
            print("Determining weights...")
        
        dates = tqdm(self.signals.index) if progress_bar else self.signals.index
        for date in dates:

            signals = self.signals.loc[date]
            weights = signals.copy()

            weights[signals == 1] /= sum(weights == 1)
            weights[signals == -1] /= sum(weights == -1)

            self.weights.loc[date] = weights
        
        self.weights = self.weights.shift()[1:]
        self.betas = self.betas[1:]

        etf_weights = pd.DataFrame(index=self.weights.index)

        for i, etf in enumerate(self.etfs):
            etf_weights[etf] = -np.stack((self.betas * self.weights).sum(axis=1))[:, i]

        self.weights = pd.concat([self.weights, etf_weights], axis=1)

    def generate_signals(self, progress_bar=True):

        if self.s_scores.isna().all().all():
            self.get_s_scores(progress_bar=progress_bar)

        if progress_bar:
            print("Generating signals...")

        stocks = tqdm(self.stocks) if progress_bar else self.stocks

        for stock in stocks:
            
            pos = 0
            stock_pos = []

            for date in self.signals.index:

                s = self.s_scores.loc[date, stock]

                if s <= -self.entry_th:
                    pos = 1
                    stock_pos.append(pos)
                elif s >= self.entry_th:
                    pos = -1
                    stock_pos.append(pos)
                
                elif s >= -self.exit_long_th and pos == 1:
                    pos = 0
                    stock_pos.append(pos)
                
                elif s <= self.exit_short_th and pos == -1:
                    pos = 0
                    stock_pos.append(pos)
                else:
                    stock_pos.append(pos)
            
            self.signals[stock] = stock_pos
        
        self.signals = self.signals.shift().fillna(0)
                    
    def get_s_scores(self, progress_bar=True):

        if progress_bar:
            print("Computing s scores...")
        
        dates = tqdm(self.s_scores.index) if progress_bar else self.s_scores.index

        for date in dates:
            
            past_stock_rets = self.stock_returns.loc[:date][-self.lookback:]
            past_etf_rets = standardize(self.market_returns.loc[:date][-self.lookback:])

            ou_params = pd.DataFrame(index=self.stocks, columns=['a', 'b', 'var(z)', 'k', 'm', 'sigma', 'sigma_eq'])

            for stock in self.stocks:

                X = past_etf_rets[self.etfs].values
                y = past_stock_rets[stock].values

                ou_model = OUModel(X, y)

                B, a, b, z, k, m, sigma, sigma_eq = ou_model.get_params()
    
                self.betas.loc[date, stock] = B

                if k > self.k_threshold:
                    ou_params.loc[stock] = [param.item() for param in [a, b, np.var(z), k, m, sigma, sigma_eq]]

            ou_params.dropna(inplace=True)
            self.param_memory[date] = ou_params

            a = ou_params["a"]
            b = ou_params["b"]
            sigma_eq = ou_params["sigma_eq"]

            m_bar = (a / (1 - b)) - (np.mean(a) / (1 - np.mean(b)))
            s = -m_bar / sigma_eq
            self.s_scores.loc[date] = s