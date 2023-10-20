import numpy as np

from scipy.optimize import minimize
from .metrics import neg_sharpe_ratio

def max_sharpe_mv_opt(init_weights, past_rets, leverage=False):
    """
    Optimize asset allocation to maximize the Sharpe ratio in a portfolio.

    Parameters:
    - init_weights (array-like): Initial portfolio asset weights.
    - past_rets (array-like): Historical returns for each asset.
    - leverage (bool): Enable leverage if True; asset bounds are [0, 1] if False.

    Returns:
    - array: Optimized asset weights that maximize the Sharpe ratio.
    
    Constraints:
    - Sum of asset weights must equal 1, unless leverage is enabled.
    """
    n_assets = len(init_weights)

    bound = (0, 1) if not leverage else (0, n_assets)
    bounds = [bound for _ in range(n_assets)]

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1} if not leverage else {"type": "eq", "fun": lambda w: np.sum(w) - n_assets}

    opt = minimize(
        lambda w: neg_sharpe_ratio(w, targets=past_rets), 
        init_weights, 
        bounds=bounds, 
        constraints=constraints
    )

    return opt.x
