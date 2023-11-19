import pandas as pd
import numpy as np

def backtest(weights, returns):
    # Calculate portfolio returns
    port_rets = (weights * returns).sum(axis=1)

    # Calculate the denominator (average absolute weights)
    denom = np.abs(weights).sum(axis=1) / 2

    # Replace zeros in the denominator with NaN or another small number
    # to avoid division by zero
    denom_safe = np.where(denom == 0, np.nan, denom)

    # Calculate the final return, handling division by zero
    final_rets = port_rets / denom_safe

    return final_rets