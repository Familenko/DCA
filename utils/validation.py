import pandas as pd
import numpy as np


def check_prices(prices: pd.Series):
    if not isinstance(prices, pd.Series):
        raise ValueError("prices must be a pandas Series")

    if prices.empty:
        raise ValueError("prices series is empty")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices series must have a DatetimeIndex")

    if prices.min() <= 0:
        raise ValueError("prices series must have positive values")

    if prices.isnull().any():
        raise ValueError("prices series must not contain null values")
    
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices series must be sorted by date in ascending order")
    

def check_buy_amount(buy_amount: float):
    if buy_amount <= 0:
        raise ValueError("buy_amount must be positive")
    

def check_freq(freq: str, freq_multiplier: dict):
    if freq not in freq_multiplier:
        raise ValueError(f"Unsupported frequency: {freq}")
    

def check_fee(fee: float):
    if not (0.0 <= fee < 1.0):
        raise ValueError("fee must be between 0 and 1")


def check_minimum_profit(minimum_profit: float):
    if minimum_profit < 1.0:
        raise ValueError("minimum_profit must be greater than 1.0")

def check_minimum_loss(minimum_loss: float):
    if not (0.0 <= minimum_loss <= 1.0):
        raise ValueError("minimum_loss must be between 0 and 1")

def check_sell_fraction(sell_fraction: float):
    if not (0.0 <= sell_fraction <= 1.0):
        raise ValueError("sell_fraction must be between 0 and 1")

def check_buy_embargo(buy_embargo: int):
    if not (1 <= buy_embargo <= 365):
        raise ValueError("buy_embargo must be between 1 and 365")
    
    
def check_cooldown(cooldown_long: int, cooldown_short: int):
    if not (1 <= cooldown_long <= 365):
        raise ValueError("cooldown_long must be between 1 and 365")

    if not (1 <= cooldown_short <= 365):
        raise ValueError("cooldown_short must be between 1 and 365")
