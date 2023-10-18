import torch
import numpy as np

def neg_sharpe_ratio(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative Sharpe ratio for the given portfolio returns.

    The Sharpe ratio is a measure for calculating risk-adjusted return. This function returns a negative value because
    it's intended to be minimized in an optimization setting.

    Parameters:
    - outputs (torch.Tensor): Predicted portfolio weights.
    - targets (torch.Tensor): Realized asset returns.

    Returns:
    - torch.Tensor: Negative Sharpe ratio of the portfolio.
    """
    portfolio_returns = (outputs * targets).sum(dim=1)
    mean_portfolio_return = portfolio_returns.mean()
    sq_of_mean_portfoliio_return = mean_portfolio_return ** 2
    mean_portfolio_return_sq = (portfolio_returns ** 2).mean()
    volatility = (mean_portfolio_return_sq - sq_of_mean_portfoliio_return) ** 0.5
    return - (mean_portfolio_return / volatility) * (np.sqrt(252))