import pandas as pd
import numpy as np
import ta


def sell_portfolio(portfolio_current: int,
                    warmup_invest: int = 1000,
                    threshold: float = 3.0,
                    sell_fraction: float = 1.0) -> tuple[float, str]:
    """
    Продає частину портфелю пропорційно до перевищення ліміту інвестицій.

    Параметри:
    - prices: серія цін
    - portfolio_current: поточна вартість портфелю
    - warmup_invest: сума портфелю, яку ми вважаємо "оптимальною" після періоду розігріву
    - threshold: поріг для визначення перевищення ліміту інвестицій
    - sell_fraction: фракція портфелю для продажу
    """

    threshold_portfolio = warmup_invest * threshold
    overvalue = portfolio_current > threshold_portfolio

    if overvalue:
        return sell_fraction, f"Limit: {int(portfolio_current)}$ [-{sell_fraction:.0%}]"
         
    return 0.0, "Wait"


def sell_ma200(prices: pd.Series,
               threshold: float = 2.0,
               sell_fraction: float = 0.5) -> tuple[float, str]:
    """
    Продає частину портфелю пропорційно до відхилення ціни від 200-денної MA.

    Параметри:
    - prices: серія цін
    """

    ma200 = prices.rolling(200).mean()
    last_price = prices.iloc[-1]
    last_ma200 = ma200.iloc[-1]
    threshold_ma = last_ma200 * threshold
    overvalue = last_price > threshold_ma

    if overvalue:
        return sell_fraction, f"MA200: {last_ma200:.2f} [-{sell_fraction:.0%}]"

    return 0.0, "Wait"


def sell_ma20(prices: pd.Series,
               threshold: float = 3.0,
               sell_fraction: float = 0.5) -> tuple[float, str]:
    """
    Продає частину портфелю пропорційно до відхилення ціни від 20-денної MA.

    Параметри:
    - prices: серія цін
    """

    ma20 = prices.rolling(20).mean()
    last_price = prices.iloc[-1]
    last_ma20 = ma20.iloc[-1]
    threshold_ma = last_ma20 * threshold
    overvalue = last_price > threshold_ma

    if overvalue:
        return sell_fraction, f"MA20: {last_ma20:.2f} [-{sell_fraction:.0%}]"

    return 0.0, "Wait"


def sell_zscore(prices: pd.Series,
                k: int = 200,
                threshold: float = 1.5,
                sell_fraction: float = 0.25) -> tuple[float, str]:

    """
    Продає частину портфелю, якщо Z-score перевищує поріг.
    """

    ma = prices.rolling(k).mean()
    std = prices.rolling(k).std()
    zscore = (prices - ma) / std
    last_zscore = zscore.iloc[-1]
    overvalue = last_zscore > threshold

    if overvalue:
        return sell_fraction, f"Z-score: {last_zscore:.2f} [-{sell_fraction:.0%}]"

    return 0.0, "Wait"


def sell_rsi(prices: pd.Series,
             threshold: float = 75.0,
             sell_fraction: float = 0.25) -> tuple[float, str]:
    """
    Продає частину портфелю пропорційно до відхилення RSI від нейтральної зони 50

    Параметри:
    - prices: серія цін
    """

    rsi = ta.momentum.RSIIndicator(close=prices).rsi()
    last_rsi = rsi.iloc[-1]
    overvalue = last_rsi > threshold

    if overvalue:
        return float(sell_fraction), f"RSI: {last_rsi:.0f} [-{sell_fraction:.0%}]"
    
    return 0.0, "Wait"
