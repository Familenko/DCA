import itertools
import random
import pandas as pd
from tqdm import tqdm

from app import BacktestDCA


def _bootstrap_score(prices, strategy, n_iterations, sample_size, sequence_length):
    stocks = list(prices.columns)
    results = []

    for iteration in range(1, n_iterations + 1):
        sampled = random.choices(stocks, k=sample_size)

        for stock in sampled:
            series = prices[stock].dropna()
            if len(series) < sequence_length:
                continue

            start = random.randint(0, len(series) - sequence_length)
            window = series.iloc[start:start + sequence_length]
            if window.empty:
                continue

            _, metrics = BacktestDCA(stock, window, strategy=strategy).run()
            metrics['iteration'] = iteration
            results.append(metrics)

    return results


def grid_search(
    prices: pd.DataFrame,
    param_grid: dict[str, list],
    n_boot: int = 100,
    sample_size: int = 10,
    sequence_length: int = 365 * 4,
):
    keys = list(param_grid)
    values = [param_grid[k] for k in keys]
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    results = []
    for params in tqdm(grid):
        strategy_name = "_".join([f"{k}={v}" for k, v in params.items()])
        metrics = _bootstrap_score(prices, params, n_boot, sample_size, sequence_length)
        results.extend({**{"strategy": strategy_name}, **metric} for metric in metrics)
    
    return pd.DataFrame(results)


def strategy_compare(
    prices: pd.DataFrame,
    strategies: dict[str, dict],
    n_boot: int = 100,
    sample_size: int = 10,
    sequence_length: int = 365 * 4,
):
    results = []
    for strategy_name, strategy in tqdm(strategies.items()):
        metrics = _bootstrap_score(prices, strategy, n_boot, sample_size, sequence_length)
        results.extend({**{"strategy": strategy_name}, **metric} for metric in metrics)

    return pd.DataFrame(results)
