import numpy as np
from scipy.optimize import minimize

def port_std(weights, returns):
    """
    Calculate the portfolio standard deviation.
    
    Parameters:
    - weights: array-like, portfolio asset weights.
    - returns: DataFrame, historical asset returns.
    
    Returns:
    - float, portfolio standard deviation.
    """
    cov = returns.cov() * 252
    return np.sqrt(weights @ cov @ weights.T)

def port_sharpe(weights, returns):
    """
    Calculate the portfolio Sharpe ratio.
    
    Parameters:
    - weights: array-like, portfolio asset weights.
    - returns: DataFrame, historical asset returns.
    
    Returns:
    - float, portfolio Sharpe ratio.
    """
    exp_rets = weights @ returns.T
    exp_ret = np.mean(exp_rets) * 252
    risk = port_std(weights, returns)
    return exp_ret / risk

def risk_contribution(weights, returns):
    """
    Calculate the risk contribution of each asset in the portfolio.
    
    Parameters:
    - weights: array-like, portfolio asset weights.
    - returns: DataFrame, historical asset returns.
    
    Returns:
    - array, individual asset contributions to portfolio risk.
    """
    cov = returns.cov() * 252
    port_var = port_std(weights, returns) ** 2
    mc = np.dot(cov, weights)
    rc = mc * weights / np.sqrt(port_var)
    return rc

def risk_parity_objective(weights, cov_matrix):
    """
    Objective function for risk parity optimization.
    
    Parameters:
    - weights: array-like, portfolio asset weights.
    - cov_matrix: DataFrame, covariance matrix of asset returns.
    
    Returns:
    - float, sum of squared deviations of asset risk contributions from their mean.
    """
    rc = risk_contribution(weights, cov_matrix)
    return np.sum((rc - np.mean(rc))**2)

def max_sharpe_opt(weights, returns):
    """
    Maximize the portfolio Sharpe ratio.
    
    Parameters:
    - weights: array-like, initial portfolio asset weights.
    - returns: DataFrame, historical asset returns.
    
    Returns:
    - tuple, optimized weights and maximum Sharpe ratio.
    """
    bounds = [(0, 1) for _ in range(len(weights))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    opt = minimize(
        lambda w: -port_sharpe(w, returns=returns),
        weights,
        bounds=bounds,
        constraints=constraints,
    )
    return opt.x, -opt.fun

def min_var_opt(weights, returns):
    """
    Minimize the portfolio variance.
    
    Parameters:
    - weights: array-like, initial portfolio asset weights.
    - returns: DataFrame, historical asset returns.
    
    Returns:
    - tuple, optimized weights and corresponding Sharpe ratio.
    """
    bounds = [(0, 1) for _ in range(len(weights))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    opt = minimize(
        lambda w: port_std(w, returns=returns),
        weights,
        bounds=bounds,
        constraints=constraints
    )
    return opt.x, port_sharpe(opt.x, returns)

def risk_parity_opt(weights, cov_matrix):
    """
    Optimize the portfolio for risk parity.
    
    Parameters:
    - weights: array-like, initial portfolio asset weights.
    - cov_matrix: DataFrame, covariance matrix of asset returns.
    
    Returns:
    - tuple, optimized weights and corresponding Sharpe ratio.
    """
    bounds = [(0, 1) for _ in range(len(weights))]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    opt = minimize(
        lambda w: risk_parity_objective(w, cov_matrix),
        weights,
        bounds=bounds,
        constraints=constraints,
    )
    return opt.x, port_sharpe(opt.x, cov_matrix)
