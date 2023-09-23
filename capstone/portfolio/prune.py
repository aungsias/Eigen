import pandas as pd
from typing import Tuple, List

def __prune_allocation__(allocation_df: pd.DataFrame, min_weight_th: pd.DataFrame) -> pd.DataFrame:
    """
    Prunes an allocation DataFrame by removing weights below a given threshold and re-normalizing.
    
    Parameters:
    - allocation_df (DataFrame): Original DataFrame containing allocation weights.
    - min_weight_th (float): Minimum weight threshold for pruning.
    
    Returns:
    - DataFrame: Pruned and re-normalized allocation DataFrame.
    """
    # Remove weights below the threshold
    allocation_df = allocation_df[allocation_df > min_weight_th]
    
    # Re-normalize the remaining weights to sum to 1 along each row
    allocation_df = allocation_df.div(allocation_df.sum(axis=1), axis=0)
    
    # Replace NaNs with 0 (arising from removal or division)
    allocation_df = allocation_df.fillna(0)
    
    return allocation_df

def prune_allocations(*allocation_dfs: pd.DataFrame, min_weight_th: float):
    """
    Prunes multiple allocation DataFrames using a common minimum weight threshold.
    
    Parameters:
    - *allocation_dfs (DataFrame): One or more DataFrames containing allocation weights.
    - min_weight_th (float): Minimum weight threshold for pruning.
    
    Returns:
    - tuple: A tuple containing pruned and re-normalized allocation DataFrames.
    """
    # Initialize list to store pruned DataFrames
    trimmed_dfs = []
    
    # Iterate through each DataFrame and prune it
    for df in allocation_dfs:
        trimmed_dfs.append(__prune_allocation__(df, min_weight_th))
        
    return tuple(trimmed_dfs)

def __prune_recommended_portfolio__(portfolio: pd.Series, min_weight_th: float = 0.01) -> pd.Series:
    """
    Prunes a portfolio by removing assets with weights below a specified threshold and re-normalizes the remaining weights.
    
    Parameters:
    - portfolio (pd.Series): A pandas Series containing asset weights.
    - min_weight_th (float): The minimum weight threshold below which assets will be removed. Default is 0.01.

    Returns:
    - pd.Series: A pruned and re-normalized portfolio.
    """
    portfolio = portfolio[portfolio > min_weight_th]
    return portfolio / portfolio.sum()

def prune_recommended_portfolios(*portfolios: pd.Series, min_weight_th: float = 0.01) -> Tuple[pd.Series]:
    """
    Prunes multiple portfolios by removing assets with weights below a specified threshold and re-normalizes the remaining weights.
    
    Parameters:
    - portfolios (pd.Series): One or more pandas Series containing asset weights.
    - min_weight_th (float): The minimum weight threshold below which assets will be removed. Default is 0.01.

    Returns:
    - Tuple[pd.Series]: A tuple of pruned and re-normalized portfolios.
    """
    pruned = []
    for portfolio in portfolios:
        pruned.append(__prune_recommended_portfolio__(portfolio, min_weight_th))
    return tuple(pruned)