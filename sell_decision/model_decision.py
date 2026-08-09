import numpy as np
import pandas as pd
import ta

from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier


_HORIZON_DAYS = 90
_DROP_THRESHOLD = 0.1


def model_features(prices: pd.Series) -> pd.DataFrame:
    roll_max_30 = prices.rolling(30).max()
    roll_max_100 = prices.rolling(100).max()

    roll_min_30 = prices.rolling(30).min()
    roll_min_100 = prices.rolling(100).min()

    roll_mean_30 = prices.rolling(30).mean()
    roll_mean_100 = prices.rolling(100).mean()

    X = pd.DataFrame(index=prices.index)

    # Returns
    daily_return = prices.pct_change() * 100
    for days in range(1, 31):
        X[f"ret_{days}"] = daily_return.shift(days - 1)

    # MA ratios
    X["ma_ratio_30"] = (prices / roll_mean_30 - 1) * 100
    X["ma_ratio_100"] = (prices / roll_mean_100 - 1) * 100

    # Drawdown 
    X["drawdown_30"] = (prices / roll_max_30 - 1) * 100
    X["drawdown_100"] = (prices / roll_max_100 - 1) * 100

    # Drawup
    X["drawup_30"] = (prices / roll_min_30 - 1) * 100
    X["drawup_100"] = (prices / roll_min_100 - 1) * 100
        
    return X


class SellModel:
    def __init__(
        self,
        features,
        threshold=0.5,
        sell_fraction=0.5,
        retrain_days=180,
    ):
        self.features = features
        self.threshold = threshold
        self.sell_fraction = sell_fraction
        self.retrain_days = retrain_days

        self.model = None
        self.last_train_date = None

    def _train(self, X, y):
        data = X.join(y.rename("target"), how="inner").dropna()
        X_train = data.drop(columns="target")
        y_train = data["target"]

        self.model = RandomForestClassifier(
            bootstrap=True,
            n_jobs=-1,
            class_weight="balanced_subsample",
            n_estimators=256,
            random_state=8
        )
        self.model.fit(X_train, y_train)

    def predict(self, prices: pd.Series):
        if len(prices) < 200:
            return 0.0, "Model: N/A"

        current_date = prices.index[-1]
        current_data = self.features.loc[[current_date]]

        train_data = prices.iloc[:-1]
        future_return = train_data.shift(-_HORIZON_DAYS) / train_data - 1
        y = (future_return < -_DROP_THRESHOLD).astype(int)
        X = self.features.loc[train_data.index]

        # First train
        if self.model is None:
            self._train(X, y)
            self.last_train_date = current_date

        # Retrain
        elif current_date >= self.last_train_date + timedelta(days=self.retrain_days):
            self._train(X, y)
            self.last_train_date = current_date

        proba = self.model.predict_proba(current_data)

        if proba.shape[1] == 2:
            prob_downtrend = float(np.clip(proba[0, 1], 0.0, 1.0))

            if prob_downtrend > self.threshold:
                return self.sell_fraction, f"Model: {prob_downtrend:.0%} [-{self.sell_fraction:.0%}]"

        return 0.0, "Model: N/A"
