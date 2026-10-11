import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from functools import cached_property
import yaml

from utils.mdd import max_drawdown
from utils.survival_ma200 import survival_ma200
from sell_decision.analitic_decision import sell_ma200, sell_bolinger, sell_rsi, sell_ppo
from utils.validation import validation


with open("configs/variables.yaml", "r", encoding="utf-8") as f:
    VARIABLES = yaml.safe_load(f)


@dataclass
class State:
    date: pd.Timestamp = None
    price: float = 0.0
    portfolio_qty: float = 0.0
    baseline_qty: float = 0.0
    cost_basis: float = 0.0
    cash_spent: float = 0.0
    profit: float = 0.0
    returns: float = 0.0
    extra_cash: float = 0.0
    trigger_msg: str = ""
    history: dict = field(default_factory=dict)

    @property
    def portfolio(self) -> float:
        return self.price * self.portfolio_qty

    @property
    def baseline(self) -> float:
        return self.price * self.baseline_qty
    
    @property
    def average_price(self) -> float:
        return self.cost_basis / self.portfolio_qty if self.portfolio_qty > 0 else np.nan

    @property
    def value(self) -> float:
        return self.portfolio + self.extra_cash
    
    def clear_day(self):
        self.profit = 0.0
        self.returns = 0.0
        self.trigger_msg = ""

    def update_history(self):
        self.history[self.date] = {
            "Price": self.price,
            "Portfolio_qty": self.portfolio_qty,
            "Baseline_qty": self.baseline_qty,
            "Cost_basis": self.cost_basis,
            "Cash_spent": self.cash_spent,
            "Profit": self.profit,
            "Returns": self.returns,
            "Extra_cash": self.extra_cash,
            "Trigger_msg": self.trigger_msg,
            "Portfolio": self.portfolio,
            "Baseline": self.baseline,
            "Average_price": self.average_price,
            "Value": self.value
        }


@dataclass
class Configuration:
    target: str
    prices: pd.Series
    buy_amount: float
    freq: str
    fee: float
    minimum_profit: float
    cooldown_days: int
    cooldown_wait: int
    sell_fraction: dict
    enable_sell: bool
    enable_buy: bool
    threshold_ma200_sell: float
    threshold_bolinger_sell: float
    threshold_rsi_sell: float
    threshold_ppo_sell: float

    def __post_init__(self):
        validation(self, VARIABLES)
            
    @property
    def warmup_invest(self) -> float:
        freq_multiplier = VARIABLES["freq_multiplier"]
        multiplier_freq = freq_multiplier[self.freq]
        return int(self.buy_amount * multiplier_freq)
    
    @cached_property
    def buy_dates(self):
        return set(self.prices.resample(self.freq).first().dropna().index)


class BacktestDCA:
    def __init__(
        self,
        target: str,
        prices: pd.Series,
        strategy: dict | None = None
    ):
        # --- initialize configuration ---
        strategy = strategy or {}
        defaults = VARIABLES["default_params"]
        merged_strategy = {**defaults, **strategy}

        self.config = Configuration(
            target=target,
            prices=prices,
            **merged_strategy
        )

        # --- initialize state ---
        self.state = State()

        # --- initialize metrics and history ---
        self.metrics = None
        self.history = None

        # --- initialize indicators ---
        self.cooldown = self.config.cooldown_wait

    def buy(self, date: float, buy_amount: float):
        last_price = self.config.prices.loc[:date].iloc[-1]

        buy_qty = buy_amount / last_price

        self.state.portfolio_qty += buy_qty
        self.state.baseline_qty += buy_qty
        self.state.cost_basis += buy_amount

        # metric only for buy
        self.state.cash_spent += buy_amount

    def decide_buy_time(self, date: pd.Timestamp):
        last_price = self.config.prices.loc[:date].iloc[-1]

        ma = self.config.prices.loc[:date].iloc[-200:].mean()
        below_ma = last_price < ma
        not_enough_data = len(self.config.prices.loc[:date]) < 200
        enable_buy = self.config.enable_buy
        date_for_buy = date in self.config.buy_dates
        if (below_ma or not_enough_data or enable_buy) and date_for_buy:
            return True

        return False   

    def decide_buy_amount(self, date: pd.Timestamp):
        buy_amount = self.config.buy_amount * (1 - self.config.fee)
        return buy_amount

    def sell(self, date: float, sell_fraction: float):
        last_price = self.config.prices.loc[:date].iloc[-1]
        
        sell_qty = self.state.portfolio_qty * sell_fraction
        sell_basis = self.state.cost_basis * sell_fraction
        sell_returns = sell_qty * last_price * (1 - self.config.fee)

        self.state.portfolio_qty -= sell_qty
        self.state.cost_basis -= sell_basis

        # metric only for sell
        self.state.profit += sell_returns - sell_basis
        self.state.returns += sell_returns
        self.state.extra_cash += sell_returns

    def decide_sell_time(self):
        enaugh_waited = self.cooldown == 0
        have_liquidity = self.state.portfolio_qty > 0
        profit_threshold = self.config.minimum_profit
        upper_limit = self.state.portfolio > self.config.warmup_invest
        profit_sell = self.state.portfolio > profit_threshold * self.state.cost_basis
        enable_sell = self.config.enable_sell

        if enaugh_waited and have_liquidity and upper_limit and profit_sell and enable_sell:
            return True
            
        return False

    def decide_sell_amount(self, date: pd.Timestamp):
        # --- manual sell ---
        if self.config.sell_fraction['manual']:
            sell_fraction = self.config.sell_fraction['manual']
            return sell_fraction, f"Fixed: {self.config.sell_fraction['manual'] * 100:.0f}%"

        # --- major sell ---
        ma200_sell = sell_ma200(
            prices=self.config.prices.loc[:date],
            threshold=self.config.threshold_ma200_sell,
            sell_fraction=self.config.sell_fraction['major']
            )

        ppo_sell = sell_ppo(
            prices=self.config.prices.loc[:date],
            threshold=self.config.threshold_ppo_sell,
            sell_fraction=self.config.sell_fraction['major']
        )

        # --- minor sell ---
        bolinger_sell = sell_bolinger(
            prices=self.config.prices.loc[:date],
            threshold=self.config.threshold_bolinger_sell,
            sell_fraction=self.config.sell_fraction['minor']
        )

        rsi_sell = sell_rsi(
            prices=self.config.prices.loc[:date],
            threshold=self.config.threshold_rsi_sell,
            sell_fraction=self.config.sell_fraction['minor']
        )

        # --- agresive choise ---
        signals = sorted([ma200_sell, bolinger_sell, rsi_sell, ppo_sell], key=lambda x: x[0])
        sell_fraction, sell_msg = signals[-1]

        return sell_fraction, sell_msg

    def run(self):
        for date, price in self.config.prices.items():
            self.state.date = date
            self.state.price = price

            if self.cooldown > 0:
                self.cooldown -= 1

            # --- DCA buy ---
            if self.decide_buy_time(date=date):
                buy_amount = self.decide_buy_amount(date=date)
                if buy_amount > 0:
                    self.buy(date=date, buy_amount=buy_amount)

            # --- decide and execute sell ---
            if self.decide_sell_time():
                self.cooldown = self.config.cooldown_wait

                sell_fraction, sell_msg = self.decide_sell_amount(date=date)
                if sell_fraction > 0:
                    self.sell(date=date, sell_fraction=sell_fraction)
                    self.state.trigger_msg = sell_msg
                    self.cooldown = self.config.cooldown_days

            # --- record history ---
            self.state.update_history()
            self.state.clear_day()

        # --- finalize history and metrics ---
        self.history = pd.DataFrame.from_dict(self.state.history, orient="index")
        self.history.index.name = "Date"

        self.metrics = self.calculate_metrics()

        return self.history, self.metrics

    def calculate_metrics(self):
        cash_spent = int(self.history["Cash_spent"].iloc[-1])
        cash_return = int(self.history["Returns"].sum())
        portfolio = int(self.history["Portfolio"].iloc[-1])
        value = int(self.history["Value"].iloc[-1])
        profit = int(self.history["Profit"].sum())
        ma200_survival_days = survival_ma200(prices=self.config.prices)
        bull_history = int((self.history['Price'] >= self.history['Average_price']).mean() * 100)
        avg_price = self.history['Average_price'].iloc[-1]
        num_take_profits = int((self.history["Trigger_msg"] != "").sum())
        mdd_pct, mdd_usd = max_drawdown(portfolio_value=self.history["Portfolio"],
                                         realized_returns=np.cumsum(self.history["Returns"]))
        
        return {
            "Target": self.config.target,
            "Cash_spent": cash_spent,
            "Cash_return": cash_return,
            "Extra_cash": int(self.state.extra_cash),
            "Portfolio": portfolio,
            "Value": value,
            "Profit": profit,
            "Bull_history": bull_history,
            "Avg_price": avg_price,
            "Num_take_profits": num_take_profits,
            "MDD": mdd_pct,
            "MDD_usd": mdd_usd,
            "MA200_survival_days": ma200_survival_days,
        }
