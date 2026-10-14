import itertools
import random
import pandas as pd
from tqdm import tqdm

from app import BacktestDCA


DEFAULT_SEARCH_SPACE = {
    "minimum_profit": [1.1, 1.2, 1.5, 2.0],
    "cooldown_long": [30, 60, 90],
    "cooldown_short": [7, 14, 21],
}


def _bootstrap_score(prices, strategy, n_boot, sample_size, sequence_length):
    stocks = list(prices.columns)
    boots = []

    for _ in range(n_boot):
        sampled = random.choices(stocks, k=sample_size)
        totals = None
        valid = 0

        for stock in sampled:
            series = prices[stock].dropna()
            if len(series) < sequence_length:
                continue

            start = random.randint(0, len(series) - sequence_length)
            window = series.iloc[start:start + sequence_length].iloc[200:]
            if window.empty:
                continue

            _, m = BacktestDCA(stock, window, strategy=strategy).run()
            m = {k: v for k, v in m.items() if k != "Target"}

            if totals is None:
                totals = {k: 0 for k in m}
            for k in totals:
                totals[k] += m[k]
            valid += 1

        if valid:
            boots.append({k: totals[k] / valid for k in totals})

    if not boots:
        return {}

    result = {}
    for k in boots[0]:
        result[k] = sum(b[k] for b in boots) / len(boots)
    return result


def grid_search(
    prices,
    param_grid=None,
    metric="RoAC",
    n_boot=100,
    sample_size=10,
    sequence_length=365 * 4,
):
    param_grid = param_grid or DEFAULT_SEARCH_SPACE

    keys = list(param_grid)
    values = [param_grid[k] for k in keys]
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    rows = []
    for params in tqdm(grid):
        metrics = _bootstrap_score(prices, params, n_boot, sample_size, sequence_length)
        rows.append({**params, **metrics})

    results = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
    best = {k: results.iloc[0][k] for k in keys}
    
    return results, best
