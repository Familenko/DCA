import numpy as np


def return_on_avg_capital(total_value, cash_spent) -> int:
    """
    Time-averaged return on capital along the path.
    Each day: (Total_value - Cash_spent) / Cash_spent.
    Mean over days with Cash_spent > 0.
    Penalizes long underwater stretches; a late spike barely moves the mean.
    Returns percent (e.g. 35 means 35%).
    """
    total_value = np.asarray(total_value, dtype=float)
    cash_spent = np.asarray(cash_spent, dtype=float)
    mask = cash_spent > 0
    if not mask.any():
        return 0
    daily_return = (total_value[mask] - cash_spent[mask]) / cash_spent[mask] * 100
    return int(np.mean(daily_return))
