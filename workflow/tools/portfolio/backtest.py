from pandas import DataFrame, Series
from typing import Tuple, List

def __backtest_portfolio__(allocation: DataFrame, constituent_returns: DataFrame) -> Series:
    """
    Backtest a single portfolio based on asset allocation and constituent returns.
    
    Parameters:
    - allocation_df (DataFrame): Asset allocation for the portfolio.
    - constituent_returns (DataFrame): Returns for each constituent asset.
    
    Returns:
    - Series: Cumulative returns of the backtested portfolio.
    """
    
    # Align index of allocation DataFrame with that of the constituent returns DataFrame
    strat_rets = allocation.reindex(constituent_returns.index).ffill().dropna()
    
    # Multiply each asset's allocation with its returns
    strat_rets = strat_rets * constituent_returns.reindex(strat_rets.index)
    
    # Sum across all assets to get portfolio return at each time step
    strat_rets = strat_rets.sum(axis=1)
    
    # Set return to zero at the starting point
    strat_rets.loc[strat_rets.index.min()] = 0
    
    return strat_rets

def backtest_portfolios(*allocation_dfs: List[DataFrame], constituent_returns: DataFrame) -> Tuple[Series]:
    """
    Backtest multiple portfolios based on their asset allocations and constituent returns.
    
    Parameters:
    - *allocation_dfs (List[DataFrame]): Asset allocations for multiple portfolios.
    - constituent_returns (DataFrame): Returns for each constituent asset.
    
    Returns:
    - Tuple[Series]: Cumulative returns of all backtested portfolios.
    """
    
    strat_ret_dfs = []
    
    # Iterate through each portfolio's asset allocation DataFrame
    for df in allocation_dfs:
        # Backtest the portfolio and store its cumulative returns
        strat_ret_dfs.append(__backtest_portfolio__(df, constituent_returns))
        
    return tuple(strat_ret_dfs)

def get_transaction_costs(*allocations: List[DataFrame], cost: float) -> Tuple[Series]:
    """
    Calculate transaction costs based on changes in asset allocations.
    
    Parameters:
    - *allocations (List[DataFrame]): List of DataFrames, each representing asset allocations for a portfolio.
    - cost (float): The transaction cost rate as a proportion of the traded volume.
    
    Returns:
    - Tuple[Series]: A tuple containing Series objects, each representing the transaction costs for a portfolio.
    """
    
    costs = []
    for allocation in allocations:
        delta_weights = allocation.diff().abs().sum(axis=1)
        cost_series = cost * delta_weights
        costs.append(cost_series)
    return tuple(costs)

def reindex_costs(cost_series: Series, port_rets: Series) -> Series:
    """
    Reindex the transaction costs series to align with the portfolio returns series, filling missing values with zeros.
    
    Parameters:
    - cost_series (Series): A Pandas Series containing transaction costs for a portfolio.
    - port_rets (Series): A Pandas Series containing portfolio returns.
    
    Returns:
    - Series: A reindexed transaction costs series aligned with the portfolio returns series.
    """
    
    return cost_series.reindex(port_rets.index).fillna(0)
