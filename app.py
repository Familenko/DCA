import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from functools import cached_property
import yaml

from utils.mdd import max_drawdown
from utils.banking import complex_percent
from utils.survival_ma200 import survival_ma200
from sell_decision.analitic_decision import sell_ma200, sell_portfolio, sell_zscore
from sell_decision.model_decision import sell_model
from utils.validation import validation


with open("configs/variables.yaml", "r", encoding="utf-8") as f:
    VARIABLES = yaml.safe_load(f)


@dataclass
class State:
    date: pd.Timestamp = None
    price: float = 0.0
    qty: float = 0.0
    cost_basis: float = 0.0
    cash_spent: float = 0.0
    profit: float = 0.0
    returns: float = 0.0
    trigger_msg: str = ""
    history: dict = field(default_factory=dict)

    @property
    def portfolio(self) -> float:
        return self.price * self.qty
    
    @property
    def average_price(self) -> float:
        return self.cost_basis / self.qty if self.qty > 0 else np.nan
    
    def clear_day(self):
        self.profit = 0.0
        self.returns = 0.0
        self.trigger_msg = ""

    def update_history(self):
        self.history[self.date] = {
            "Price": self.price,
            "Qty": self.qty,
            "Cost_basis": self.cost_basis,
            "Cash_spent": self.cash_spent,
            "Profit": self.profit,
            "Returns": self.returns,
            "Trigger_msg": self.trigger_msg,
            "Portfolio": self.portfolio,
            "Average_price": self.average_price,
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
    manual_sell_fraction: float
    auto_sell_fraction: float
    enable_sell: bool
    enable_model: bool
    threshold_invest_years: int
    threshold_model_sell: float
    threshold_ma200_sell: float
    threshold_zscore_sell: float

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

        # --- initialize cooldown ---
        self.cooldown = self.config.cooldown_days

    def execute_buy(self, price: float):
        effective_amount = self.config.buy_amount * (1 - self.config.fee)
        buy_qty = effective_amount / price

        self.state.qty += buy_qty
        self.state.cost_basis += self.config.buy_amount

        # metric only for buy
        self.state.cash_spent += self.config.buy_amount

    def execute_sell(self, price: float, sell_fraction: float):
        sell_qty = self.state.qty * sell_fraction
        sell_basis = self.state.cost_basis * sell_fraction
        sell_returns = sell_qty * price * (1 - self.config.fee)

        self.state.qty -= sell_qty
        self.state.cost_basis -= sell_basis

        # metric only for sell
        self.state.profit += sell_returns - sell_basis
        self.state.returns += sell_returns

    def decide_time(self):
        enaugh_waited = self.cooldown == 0
        have_liquidity = self.state.qty > 0
        upper_limit = self.state.portfolio > self.config.warmup_invest
        profit_sell = self.state.portfolio > self.config.minimum_profit * self.state.cost_basis

        if enaugh_waited and have_liquidity and upper_limit and profit_sell:
            return True
            
        return False

    def decide_sell(self, date: pd.Timestamp):
        if self.config.manual_sell_fraction:
            sell_fraction = self.config.manual_sell_fraction
            return sell_fraction, f"Fixed: {self.config.manual_sell_fraction * 100:.0f}%"
        
        else:
            prices_to_date = self.config.prices.loc[:date]

            ma200_sell = sell_ma200(prices=prices_to_date, 
                                    threshold=self.config.threshold_ma200_sell)
            
            portfolio_sell = sell_portfolio(portfolio_current=self.state.portfolio,
                                            warmup_invest=self.config.warmup_invest,
                                            invest_years=self.config.threshold_invest_years)
            
            zscore_sell = sell_zscore(prices=prices_to_date, 
                                      threshold=self.config.threshold_zscore_sell)

            model_sell = sell_model(prices=prices_to_date,
                                    threshold=self.config.threshold_model_sell
                                    ) if self.config.enable_model else (False, "Model: N/A")

            signals = [model_sell, ma200_sell, zscore_sell, portfolio_sell]
            signals = [(confirm, msg) for confirm, msg in signals if confirm]
            confirm_lvl, sell_msg = next(iter(signals), (False, ""))
            sell_fraction = self.config.auto_sell_fraction if confirm_lvl else 0.0

            return sell_fraction, sell_msg

    def run(self):
        for date, price in self.config.prices.items():
            self.state.date = date
            self.state.price = price

            if self.cooldown > 0:
                self.cooldown -= 1

            # --- DCA buy ---
            if date in self.config.buy_dates:
                self.execute_buy(price)

            # --- decide and execute sell ---
            if self.decide_time() and self.config.enable_sell:
                self.cooldown = self.config.cooldown_wait

                sell_fraction, sell_msg = self.decide_sell(date=date)
                if sell_fraction > 0:
                    self.execute_sell(price=price, sell_fraction=sell_fraction)
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
        profit = int(self.history["Profit"].sum())
        ma200_survival_days = survival_ma200(prices=self.config.prices)
        bank_profit = complex_percent(returns=self.history["Returns"], rate=VARIABLES["banking_rate"])
        bull_history = int((self.history['Price'] >= self.history['Average_price']).mean() * 100)
        num_take_profits = int((self.history["Trigger_msg"] != "").sum())
        mdd_pct, mdd_usd = max_drawdown(portfolio_value=self.history["Portfolio"],
                                         realized_returns=np.cumsum(self.history["Returns"]))
        
        return {
            "Target": self.config.target,
            "Cash_spent": cash_spent,
            "Cash_return": cash_return,
            "Portfolio": portfolio,
            "Profit": profit,
            "Bull_history": bull_history,
            "Num_take_profits": num_take_profits,
            "MDD": mdd_pct,
            "MDD_usd": mdd_usd,
            "Bank_profit": bank_profit,
            "MA200_survival_days": ma200_survival_days,
        }
