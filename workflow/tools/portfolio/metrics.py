import pandas as pd
import numpy as np

class Metrics:

    def __init__(self):
        self.metrics = pd.DataFrame(columns=[
            "Sharpe Ratio", "Cumulative Return", "Annualized Return", "Annualized Volatility"
        ])

    def calculate(self, data):
        for col in data.columns:
            cum_ret = self._cumulative_returns(data[col])
            sharpe = self._sharpe(data[col])
            ann_ret = self._annualized_return(data[col])
            ann_std = self._annualized_risk(data[col])
            metrics = {
                "Cumulative Return": cum_ret,
                "Sharpe Ratio": sharpe,
                "Annualized Return": ann_ret,
                "Annualized Volatility": ann_std
            }
            self.metrics.loc[col] = metrics
        return self.metrics.sort_values(by=["Cumulative Return"], ascending=False)
    
    @staticmethod
    def _cumulative_returns(data):
        return data.cumsum().iloc[-1]

    @staticmethod
    def _sharpe(data):
        return (data.mean() * 252) / (data.std() * np.sqrt(252))
    
    @staticmethod
    def _annualized_return(data):
        return data.mean() * 252
    
    @staticmethod
    def _annualized_risk(data):
        return data.std() * np.sqrt(252)