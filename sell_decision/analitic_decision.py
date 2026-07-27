import pandas as pd
import numpy as np
import ta


def sell_portfolio(portfolio_current: int,
                    warmup_invest: int = 1000,
                    invest_years: float = 4.0) -> tuple[float, str]:
    """
    Продає частину портфелю пропорційно до перевищення ліміту інвестицій.

    Параметри:
    - prices: серія цін
    - portfolio_current: поточна вартість портфелю
    - warmup_invest: сума портфелю, яку ми вважаємо "оптимальною" після періоду розігріву
    - invest_years: кількість років для визначення максимальної суми портфелю
    """

    denominator = invest_years - 1
    ratio = portfolio_current / warmup_invest
    overvalue = (ratio - 1) / denominator

    if overvalue > 0:
        sell_fraction = min(overvalue, 0.5)
        return sell_fraction, f"Limit: {int(portfolio_current)}$ [-{sell_fraction:.0%}]"
         
    return 0.0, "Wait"


def sell_ma200(prices: pd.Series) -> tuple[float, str]:
    """
    Продає частину портфелю пропорційно до відхилення ціни від 200-денної MA.

    Параметри:
    - prices: серія цін
    """

    ma200 = prices.rolling(200).mean()
    last_price = prices.iloc[-1]
    last_ma200 = ma200.iloc[-1]
    overvalue = (last_price - last_ma200) / last_ma200

    if overvalue > 0:
        sell_fraction = min(overvalue, 0.5)
        return sell_fraction, f"Bull: {overvalue:.0%} [-{sell_fraction:.0%}]"

    return 0.0, "Wait"


def sell_zscore(prices: pd.Series,
                k: int = 200,
                threshold: float = 3.0) -> tuple[float, str]:

    """
    Продає частину портфелю, якщо Z-score перевищує поріг.
    """

    ma = prices.rolling(k).mean()
    std = prices.rolling(k).std()
    zscore = (prices - ma) / std
    last_zscore = zscore.iloc[-1]
    overvalue = (last_zscore - threshold) / threshold

    if overvalue > 0:
        sell_fraction = min(overvalue, 0.5)
        return sell_fraction, f"Z-score: {last_zscore:.2f} [-{sell_fraction:.0%}]"

    return 0.0, "Wait"


def sell_rsi(prices: pd.Series) -> tuple[float, str]:
    """
    Продає частину портфелю пропорційно до відхилення RSI від нейтральної зони 50

    Параметри:
    - prices: серія цін
    """

    rsi = ta.momentum.RSIIndicator(close=prices).rsi()
    last_rsi = rsi.iloc[-1]
    overvalue = (last_rsi - 50) / 100

    if overvalue > 0:
        sell_fraction = min(overvalue, 0.5)
        return float(sell_fraction), f"RSI: {last_rsi:.0f} [-{sell_fraction:.0%}]"
    
    return 0.0, "Wait"
