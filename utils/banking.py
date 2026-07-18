import pandas as pd


def complex_percent(returns: pd.Series, rate: float = 0.03) -> int:
    """
    Розраховує ефект складного відсотка для серії отриманих прибутків/повернень.

    Функція бере всі ненульові значення із серії `returns`, вважає,
    скільки днів кожен прибуток міг би додатково зростати до останньої
    дати серії, і обчислює майбутню вартість кожного прибутку
    за формулою складного відсотка.

    Parameters
    ----------
    returns : pandas.Series
        Серія з datetime-індексом, де значення містять
        прибуток/дохід за кожен період.

    rate : float, optional, default=0.03
        Річна ставка складного відсотка у десятковому форматі.

    Returns
    -------
    int
        Абсолютний додатковий прибуток від складного відсотка.

    Notes
    -----
    Формула розрахунку:
        FV = PV * (1 + rate) ** (days / 365)
    де:
        - FV = future value (майбутня вартість)
        - PV = present value (поточний прибуток)
        - days = кількість днів до кінця періоду

    Якщо сума всіх returns дорівнює нулю,
    функція повертає `0`.
    """

    max_day = returns.index.max()
    non_zero_returns = returns[returns != 0]

    future_total = 0.0
    for date, profit in non_zero_returns.items():
        days = (max_day - date).days
        future_total += profit * (1 + rate) ** (days / 365)

    sum_returns = float(non_zero_returns.sum())
    profit = future_total - sum_returns

    return int(profit)
