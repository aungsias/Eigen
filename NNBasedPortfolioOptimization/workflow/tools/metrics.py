import torch

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
    volatility = torch.std(portfolio_returns)
    return - (mean_portfolio_return / volatility)

def portfolio_risk(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the annualized portfolio risk (standard deviation) for the given portfolio returns.

    Parameters:
    - outputs (torch.Tensor): Predicted portfolio weights.
    - targets (torch.Tensor): Realized asset returns.

    Returns:
    - torch.Tensor: Annualized portfolio risk.
    """
    portfolio_returns = (outputs * targets).sum(dim=1)
    return torch.std(portfolio_returns)