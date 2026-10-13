import numpy as np


def return_on_avg_capital(total_value, cash_spent, cost_basis):
    """
    Profit relative to average capital deployed in the position.
    Args:
        total_value (float): Final portfolio + realized cash.
        cash_spent (float): Total cash spent on buys.
        cost_basis (list or np.ndarray): Cost basis over time.
    Returns:
        int: return on average capital as percent (e.g. 35 means 35%).
    """
    profit = total_value - cash_spent
    profit = profit / cost_basis * 100

    return int(profit.mean())
