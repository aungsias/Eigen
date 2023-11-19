import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

class OUModel:
    def __init__(self, market_returns, stock_returns):
        self.trading_days = 252
        self.X = market_returns
        self.y = stock_returns
        
        if isinstance(self.X, pd.DataFrame):
            self.X = market_returns.values
        if isinstance(self.y, pd.DataFrame):
            self.y = stock_returns.values
    
    def get_params(self):
        B, e = self._market_regression(self.X, self.y)
        E = e.cumsum()
        X, y = E[:-1].reshape(-1, 1), E[1:]
        a, b, z = self._ou_regression(X, y)
        k = -np.log(b) * self.trading_days
        m = a / (1 - b)
        sigma = np.sqrt((2 * k * np.var(z)) / (1 - b**2))
        sigma_eq = np.sqrt(np.var(z) / (1 - b**2))
        return B, a, b, z, k, m, sigma, sigma_eq
    
    def _market_regression(self, market, stock):
        reg = LinearRegression().fit(market, stock)
        return reg.coef_, stock - reg.predict(market)

    def _ou_regression(self, lagged, non_lagged):
        reg = LinearRegression().fit(lagged, non_lagged)
        return reg.intercept_, reg.coef_, non_lagged - reg.predict(lagged)