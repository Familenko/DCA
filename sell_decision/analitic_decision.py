import pandas as pd


def sell_portfolio(portfolio_current: int,
                   warmup_invest: int = 1000,
                   invest_years: int = 3) -> tuple[float, str]:
    """
    Параметри:
    - portfolio_current: поточна вартість портфелю
    - warmup_invest: сума портфелю, яку ми вважаємо "оптимальною" після періоду розігріву
    - invest_years: поріг для визначення максимальної суми портфелю
    """

    invest_limit = warmup_invest * invest_years
    if portfolio_current >= invest_limit:
        return True, f"Limit: {int(portfolio_current)}$"
         
    return False, f"Limit: {int(portfolio_current)}$"


def sell_ma200(prices: pd.Series,
               threshold: float = 0.5) -> tuple[float, str]:
    """
    Продає частину портфелю пропорційно до відхилення ціни від 200-денної MA.

    Параметри:
    - prices: серія цін
    - max_sell_fraction: максимальна частка портфелю, яку можна продати
    """

    if len(prices) < 200:
        return False, "MA200: N/A"

    ma200 = prices.rolling(200).mean()
    last_price = prices.iloc[-1]
    last_ma200 = ma200.iloc[-1]
    rise = (last_price - last_ma200) / last_ma200
    
    if last_price > last_ma200:
        if rise > threshold:
            return True, f"Bull: {rise:.0%}"

    return False, "MA200: bear"


def sell_zscore(prices: pd.Series,
                k: int = 200,
                threshold: float = 2.0) -> tuple[bool, str]:
    """
    Сигнал продажу на основі Z-score (ZDCA з статті).

    Якщо поточна ціна відхиляється від середньої за k днів більше ніж на
    threshold стандартних відхилень — ціна "перегріта" і варто продавати.

    Параметри:
    - prices: серія цін
    - k: кількість торгових днів для розрахунку середнього та std
    - threshold: поріг z-score для сигналу продажу
    """
    if len(prices) < k:
        return False, "ZScore: N/A"

    window = prices.iloc[-k:]
    mu = window.mean()
    sigma = window.std()

    z = (prices.iloc[-1] - mu) / sigma

    if z > threshold:
        return True, f"ZScore: {z:.2f}"

    return False, f"ZScore: {z:.2f}"
