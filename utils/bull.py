import pandas as pd
import numpy as np

def bull_history(prices: pd.Series, average_price: float) -> int:
    return int((prices >= average_price).mean() * 100)