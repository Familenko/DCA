import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.validation import check_prices


class FallenTokenFinder:
    def __init__(
        self,
        prices_df: pd.DataFrame,

        min_last_days: int = 20,
        max_last_days: int = 200,
        step_day: int = 1,
        min_procent: float = 0.1,
        max_procent: float = 1.0,
        ma_days: int = 20,
        stabilization_days: int = 100,
        history_period_days: int = 365 * 2,
    ):
        self.min_last_days = min_last_days
        self.max_last_days = max_last_days
        self.step_day = step_day

        self.min_procent = min_procent
        self.max_procent = max_procent

        self.ma_days = ma_days
        self.stabilization_days = stabilization_days
        self.history_period_days = history_period_days

        self.prices_df = self._validate_df(prices_df)

    def _validate_df(self, prices_df: pd.DataFrame):
        if not isinstance(prices_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if len(prices_df) < self.history_period_days:
            raise ValueError(f"DataFrame must have at least {self.history_period_days} rows.")

        for col in prices_df.columns:
            check_prices(prices_df[col])
        
        return prices_df

    def _optimize_symbol(self, close_series_symbol: pd.Series) -> dict:
        prices = close_series_symbol.iloc[-self.history_period_days:]

        best_params = {
            "procent_ATL": self.max_procent,
            "confidence": 0.0,
        }

        if len(prices) < self.history_period_days:
            return best_params

        last_price = prices.iloc[-1]
        min_support = prices.iloc[-(self.ma_days + 1):-1].min()
        price_threshold_min = np.percentile(prices.iloc[:-(self.ma_days + 1)], 5)

        recent_min = prices.iloc[-self.stabilization_days:].min()
        older_min = prices.iloc[:-self.stabilization_days].min()
        no_new_lower_lows = recent_min >= older_min

        if not no_new_lower_lows or last_price < min_support:
            return best_params

        for last_days in range(self.min_last_days, self.max_last_days + 1, self.step_day):
            slice_mean = prices.iloc[-last_days:].mean()

            required_procent = (slice_mean / price_threshold_min) - 1.0
            required_procent = round(required_procent, 2)

            if not (self.min_procent <= required_procent < self.max_procent):
                continue

            current_confidence = last_days / self.max_last_days

            procent_improvement = required_procent < best_params["procent_ATL"]
            procent_same = required_procent == best_params["procent_ATL"]
            confidence_improvement = current_confidence > best_params["confidence"]

            if procent_improvement or (procent_same and confidence_improvement):
                best_params = {
                    "procent_ATL": required_procent,
                    "confidence": current_confidence,
                }

        return best_params

    def result_df(self) -> pd.DataFrame:
        df_stocks = self.prices_df.copy()

        symbols_iter = df_stocks.columns
        symbols_iter = tqdm(symbols_iter, desc="Optimizing")

        optimal_symbols = {}

        for symbol in symbols_iter:
            price_series = df_stocks[symbol].dropna()
            best_params = self._optimize_symbol(price_series)
            optimal_symbols[symbol] = best_params

        candidates = pd.DataFrame.from_dict(optimal_symbols, orient="index")
        candidates = candidates[candidates["procent_ATL"] < self.max_procent]
        candidates.sort_values(["procent_ATL", "confidence"], ascending=[True, False], inplace=True)

        return candidates

    def result_plot(self):
        candidates = self.result_df()
        
        for symbol in candidates.index:
            max_procent = candidates.loc[symbol, "procent_ATL"]
            confidence = candidates.loc[symbol, "confidence"]
            min_value = self.prices_df[symbol].iloc[-self.stabilization_days:].min()
            ma200 = self.prices_df[symbol].rolling(window=200).mean()

            plt.figure(figsize=(14, 6))
            plt.plot(self.prices_df[symbol].iloc[-self.history_period_days:], label="Price", color="darkblue", linewidth=1.0)
            plt.plot(ma200.iloc[-self.history_period_days:], label="MA200", color="green", linestyle='--', linewidth=1.0)
            plt.axhline(min_value, color="red", linestyle="--", linewidth=1.0, label=f"Min {self.stabilization_days} days")

            plt.title(
                f"[{symbol}] {max_procent * 100:.0f}% > ATL, "
                f"Confidence: {confidence:.2f}"
            )
            plt.legend()
            plt.tight_layout()
            plt.show()
