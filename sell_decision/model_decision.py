import numpy as np
import pandas as pd
import ta

from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier


_HORIZON_DAYS = 90
_DROP_THRESHOLD = 0.1


def _build_features(prices: pd.Series) -> pd.DataFrame:
	ret = prices.pct_change()
	roll_max_30 = prices.rolling(30).max()
	roll_max_100 = prices.rolling(100).max()

	roll_min_30 = prices.rolling(30).min()
	roll_min_100 = prices.rolling(100).min()

	X = pd.DataFrame(index=prices.index)

	# Returns & volatility
	X["ret_30"] = prices.pct_change(30)
	X["ret_100"] = prices.pct_change(100)
     
	X["vol_30"] = ret.rolling(30).std()
	X["vol_100"] = ret.rolling(100).std()

	# Moving-average ratios
	X["ma_ratio_30"] = prices / prices.rolling(30).mean() - 1
	X["ma_ratio_100"] = prices / prices.rolling(100).mean() - 1

	# Drawdown and drawup
	X["drawdown_30"] = prices / roll_max_30 - 1
	X["drawdown_100"] = prices / roll_max_100 - 1

	X["drawup_30"] = prices / roll_min_30 - 1
	X["drawup_100"] = prices / roll_min_100 - 1

	# RSI (14)
	rsi = ta.momentum.RSIIndicator(close=prices, window=14)
	X["rsi_14"] = rsi.rsi() / 100

	# Stochastic RSI
	stoch_rsi = ta.momentum.StochRSIIndicator(close=prices, window=14)
	X["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
	X["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

	# MACD histogram (normalised by price)
	macd = ta.trend.MACD(close=prices)
	X["macd_diff"] = macd.macd_diff() / prices

	# Bollinger Bands %B and bandwidth
	bb = ta.volatility.BollingerBands(close=prices, window=20)
	X["bb_pct"] = bb.bollinger_pband()
	X["bb_width"] = bb.bollinger_wband()

	# Rate of Change (12-period)
	roc = ta.momentum.ROCIndicator(close=prices, window=12)
	X["roc_12"] = roc.roc() / 100
      
	return X



class SellModel:
    def __init__(
        self,
        threshold=0.5,
        sell_fraction=0.5,
        retrain_days=180
    ):
        self.threshold = threshold
        self.sell_fraction = sell_fraction
        self.retrain_days = retrain_days

        self.model = None
        self.last_train_date = None

    def _train(self, prices: pd.Series):

        train = prices.iloc[:-1]

        X = _build_features(train)

        future_return = train.shift(-_HORIZON_DAYS) / train - 1
        y = (future_return < -_DROP_THRESHOLD).astype(int)

        dataset = pd.concat(
            [X, y.rename("target")],
            axis=1
        ).dropna()

        X_train = dataset.drop(columns=["target"])
        y_train = dataset["target"]

        clf = RandomForestClassifier(
            bootstrap=True,
            n_jobs=-1,
            class_weight="balanced_subsample",
            n_estimators=256,
            random_state=8
        )

        clf.fit(X_train, y_train)

        self.model = clf
        self.last_train_date = prices.index[-1]

    def predict(self, prices: pd.Series):

        if len(prices) < 200:
            return 0.0, "Model: N/A"

        current_date = prices.index[-1]

        # Потрібно навчити модель вперше
        if self.model is None:
            self._train(prices)

        # Потрібно перенавчити модель
        elif current_date >= self.last_train_date + timedelta(days=self.retrain_days):
            self._train(prices)

        # prediction
        current = _build_features(prices).iloc[[-1]]

        proba = self.model.predict_proba(current)

        if proba.shape[1] == 2:

            prob_downtrend = float(
                np.clip(proba[0, 1], 0.0, 1.0)
            )

            if prob_downtrend > self.threshold:
                return (
                    self.sell_fraction,
                    f"Model: {prob_downtrend:.0%} "
                    f"[-{self.sell_fraction:.0%}]"
                )

        return 0.0, "Model: N/A"
