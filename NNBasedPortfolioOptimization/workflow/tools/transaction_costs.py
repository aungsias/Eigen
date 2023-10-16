import pandas as pd

def net_tc(returns: pd.Series, weights: pd.DataFrame, cost_rate: float=0.005) -> pd.Series:
    delta_weights = weights.diff().abs().sum(axis=1)
    tc = delta_weights * cost_rate
    returns = returns - tc
    return returns