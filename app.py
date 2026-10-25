import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from functools import cached_property
import yaml

from utils.mdd import max_drawdown
from utils.survival_ma200 import survival_ma200
from utils.roac import return_on_avg_capital
from utils.validation import validation


with open("configs/variables.yaml", "r", encoding="utf-8") as f:
    VARIABLES = yaml.safe_load(f)


@dataclass
class State:
    date: pd.Timestamp = None
    price: float = 0.0
    portfolio_qty: float = 0.0
    cost_basis: float = 0.0
    cash_spent: float = 0.0
    profit: float = 0.0
    returns: float = 0.0
    realized_cash: float = 0.0
    trigger_msg: str = ""
    history: dict = field(default_factory=dict)

    @property
    def portfolio(self) -> float:
        return self.price * self.portfolio_qty

    @property
    def average_price(self) -> float:
        return self.cost_basis / self.portfolio_qty if self.portfolio_qty > 0 else np.nan

    @property
    def total_value(self) -> float:
        return self.portfolio + self.realized_cash
    
    def clear_day(self):
        self.profit = 0.0
        self.returns = 0.0
        self.trigger_msg = ""

    def update_history(self):
        self.history[self.date] = {
            "Price": self.price,
            "Portfolio_qty": self.portfolio_qty,
            "Cost_basis": self.cost_basis,
            "Cash_spent": self.cash_spent,
            "Profit": self.profit,
            "Returns": self.returns,
            "Realized_cash": self.realized_cash,
            "Trigger_msg": self.trigger_msg,
            "Portfolio": self.portfolio,
            "Average_price": self.average_price,
            "Total_value": self.total_value
        }


@dataclass
class Configuration:
    target: str
    prices: pd.Series
    buy_amount: float
    freq: str
    fee: float
    minimum_profit: float
    minimum_loss: float
    sell_fraction: float
    buy_embargo: int
    cooldown_short: int
    cooldown_long: int

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
        self.cooldown = 0
        self.embargo = 0

    def buy(self, date: float, buy_amount: float):
        last_price = self.config.prices.loc[:date].iloc[-1]

        buy_qty = buy_amount / last_price

        self.state.portfolio_qty += buy_qty
        self.state.cost_basis += buy_amount

        # metric only for buy
        self.state.cash_spent += buy_amount

    def decide_buy_time(self, date: pd.Timestamp):
        date_for_buy = date in self.config.buy_dates
        embargo_passed = self.embargo == 0

        if date_for_buy and embargo_passed:
            return True

        return False   

    def decide_buy_amount(self, date: pd.Timestamp):
        last_price = self.config.prices.loc[:date].iloc[-1]

        trigger_msg = ""
        buy_amount = self.config.buy_amount * (1 - self.config.fee)
        below_portfolio = last_price < self.state.average_price * self.config.minimum_loss
        if below_portfolio:
            buy_amount = self.config.warmup_invest * (1 - self.config.fee)
            trigger_msg = f"Buy: {self.config.warmup_invest:.0f}$"
            self.embargo = self.config.buy_embargo

        return buy_amount, trigger_msg

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
        self.state.realized_cash += sell_returns

    def decide_sell_time(self):
        if self.cooldown > 0:
            return False

        have_liquidity = self.state.portfolio_qty > 0
        upper_limit = self.state.portfolio > self.config.warmup_invest
        profit_sell = self.state.portfolio > self.config.minimum_profit * self.state.cost_basis

        if have_liquidity and upper_limit and profit_sell:
            self.cooldown = self.config.cooldown_long
            return True
            
        self.cooldown = self.config.cooldown_short
        return False

    def decide_sell_amount(self, date: pd.Timestamp):
        sell_fraction = self.config.sell_fraction
        return sell_fraction, f"Sell: {self.config.sell_fraction * 100:.0f}%"


    def update_decision_restrictions(self):
        if self.cooldown > 0:
            self.cooldown -= 1

        if self.embargo > 0:
            self.embargo -= 1

    def run(self):
        for date, price in self.config.prices.items():
            self.state.date = date
            self.state.price = price

            self.update_decision_restrictions()

            # --- DCA buy ---
            if self.decide_buy_time(date=date):
                buy_amount, buy_msg = self.decide_buy_amount(date=date)
                if buy_amount > 0:
                    self.buy(date=date, buy_amount=buy_amount)
                    self.state.trigger_msg = buy_msg

            # --- decide and execute sell ---
            if self.decide_sell_time():
                sell_fraction, sell_msg = self.decide_sell_amount(date=date)
                if sell_fraction > 0:
                    self.sell(date=date, sell_fraction=sell_fraction)
                    self.state.trigger_msg = sell_msg

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
        portfolio = int(self.history["Portfolio"].iloc[-1])
        realized_cash = int(self.history["Realized_cash"].iloc[-1])
        invested = int(self.history["Cost_basis"].iloc[-1])
        total_value = int(self.history["Total_value"].iloc[-1])
        
        ma200_survival_days = survival_ma200(prices=self.config.prices)
        bull_history = int((self.history['Price'] >= self.history['Average_price']).mean() * 100)
        avg_price = self.history['Average_price'].iloc[-1]
        num_take_profits = int((self.history["Trigger_msg"] != "").sum())
        mdd_usd = max_drawdown(total_value=self.history["Total_value"])
        roac = return_on_avg_capital(total_value=self.history["Total_value"],
                                     cash_spent=self.history["Cash_spent"])

        return {
            "Target": self.config.target,
            "Cash_spent": cash_spent,
            "Realized_cash": realized_cash,
            "Portfolio": portfolio,
            "Invested": invested,
            "Total_value": total_value,
            "Bull_history": bull_history,
            "Avg_price": avg_price,
            "Num_take_profits": num_take_profits,
            "MDD": mdd_usd,
            "MA200_survival_days": ma200_survival_days,
            "RoAC": roac,
        }
