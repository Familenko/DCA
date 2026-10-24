import numpy as np


def max_drawdown(total_value):
    """
    Calculate the maximum drawdown of a portfolio.
    Args:
        total_value (list or np.ndarray): Portfolio + realized cash over time.
    Returns:
        tuple: (percent_mdd, dollar_mdd) — percent as int (e.g. -20), dollar loss as int (e.g. -400).
    """
    equity = np.asarray(total_value, dtype=float)

    if equity.size == 0:
        return 0, 0

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
