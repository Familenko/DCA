import numpy as np


def max_drawdown(portfolio_value, realized_returns):
    """
    Calculate the maximum drawdown of a portfolio.
    Args:
        portfolio_value (list or np.ndarray): The value of the portfolio over time.
        realized_returns (list or np.ndarray): The realized returns over time.
    Returns:
        tuple: (percent_mdd, dollar_mdd) — percent as int (e.g. -20), dollar loss as int (e.g. -400).
    """
    portfolio_value = np.asarray(portfolio_value, dtype=float)
    realized_returns = np.asarray(realized_returns, dtype=float)

    if portfolio_value.shape != realized_returns.shape:
        raise ValueError("portfolio_value and realized_returns must have the same shape")

    if portfolio_value.size == 0:
        return 0, 0

    equity = portfolio_value + realized_returns
    peaks = np.maximum.accumulate(equity)

    dollar_drawdowns = equity - peaks
    dollar_mdd = float(dollar_drawdowns.min())
    dollar_mdd = int(dollar_mdd) if dollar_mdd < 0 else 0

    pct_drawdowns = np.zeros_like(equity, dtype=float)
    valid = peaks != 0
    pct_drawdowns[valid] = dollar_drawdowns[valid] / peaks[valid]

    pct_mdd = float(pct_drawdowns.min())
    pct_mdd = int(pct_mdd * 100) if pct_mdd < 0 else 0

    return pct_mdd, dollar_mdd
    