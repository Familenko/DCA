import pandas as pd
import numpy as np
import ta


def sell_ma200(prices: pd.Series,
               threshold: float = 2.0,
               sell_fraction: float = 0.5) -> tuple[float, str]:
    """
    Продає частину портфелю, якщо ціна виходить за верхню межу 200-денної MA.

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


def sell_bolinger(prices: pd.Series,
                    k: int = 200,
                    threshold: float = 2.0,
                    sell_fraction: float = 0.25) -> tuple[float, str]:

    """
    Продає частину портфелю, якщо ціна виходить за верхню межу Болінджера.
    """

    bb = ta.volatility.BollingerBands(close=prices, window_dev=threshold)

    last_price = prices.iloc[-1]
    last_upper = bb.bollinger_hband().iloc[-1]
    overvalue = last_price > last_upper

    if overvalue:
        return sell_fraction, f"BB: {last_upper:.2f} [-{sell_fraction:.0%}]"

    return 0.0, "Wait"


def sell_rsi(prices: pd.Series,
             threshold: float = 80.0,
             sell_fraction: float = 0.25) -> tuple[float, str]:
    """
    Продає частину портфелю, якщо RSI виходить за верхню межу.

    Параметри:
    - prices: серія цін
    """

    rsi = ta.momentum.RSIIndicator(close=prices).rsi()
    last_rsi = rsi.iloc[-1]
    overvalue = last_rsi > threshold

    if overvalue:
        return float(sell_fraction), f"RSI: {last_rsi:.0f} [-{sell_fraction:.0%}]"
    
    return 0.0, "Wait"


def sell_ppo(prices: pd.Series,
             threshold: float = 5.0,
             sell_fraction: float = 0.25) -> tuple[float, str]:
    """
    Продає частину портфелю, якщо PPO виходить за верхню межу.
    """
    ppo_ind = ta.momentum.PercentagePriceOscillator(close=prices)
    last_ppo = ppo_ind.ppo().iloc[-1]
    overvalue = last_ppo > threshold

    if overvalue:
        return sell_fraction, f"PPO: {last_ppo:.1f}% [-{sell_fraction:.0%}]"

    return 0.0, "Wait"
