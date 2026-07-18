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
        raise ValueError("minimum_profit must be greater than or equal to 1.0")
    
def check_cooldown(cooldown_days: int, cooldown_wait: int):
    if cooldown_days not in range(1, 366):
        raise ValueError("cooldown_days must be between 1 and 365")

    if cooldown_wait not in range(1, 366):
        raise ValueError("cooldown_wait must be between 1 and 365")
    
def check_threshold_invest_years(threshold_invest_years: int):
    if threshold_invest_years < 2:
        raise ValueError("threshold_invest_years must be at least 2")
    
def check_manual_sell_fraction(manual_sell_fraction: float):
    if manual_sell_fraction:
        if not (0.1 <= manual_sell_fraction <= 1.0):
            raise ValueError("sell_fraction must be between 0.1 and 1.0")


def validation(self, variables):
    check_prices(self.prices)
    check_buy_amount(self.buy_amount)
    check_freq(self.freq, variables["freq_multiplier"])
    check_fee(self.fee)
    check_minimum_profit(self.minimum_profit)
    check_cooldown(self.cooldown_days, self.cooldown_wait)
    check_threshold_invest_years(self.threshold_invest_years)
    check_manual_sell_fraction(self.manual_sell_fraction)
