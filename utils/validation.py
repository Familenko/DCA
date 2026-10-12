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
    
    
def check_cooldown(cooldown_long: int, cooldown_short: int):
    if cooldown_long not in range(1, 366):
        raise ValueError("cooldown_long must be between 1 and 365")

    if cooldown_short not in range(1, 366):
        raise ValueError("cooldown_short must be between 1 and 365")


def validation(self, variables):
    check_prices(self.prices)
    check_buy_amount(self.buy_amount)
    check_freq(self.freq, variables["freq_multiplier"])
    check_fee(self.fee)
    check_cooldown(self.cooldown_long, self.cooldown_short)
