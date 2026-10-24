import numpy as np


def _dollar_drawdown(total_value) -> tuple[np.ndarray, int | None, int | None]:
    """
    Shared dollar-drawdown calc.
    Returns (dollar_drawdowns, peak_idx, trough_idx).
    peak/trough are None if empty or no drawdown.
    """
    equity = np.asarray(total_value, dtype=float)
    if equity.size == 0:
        return equity, None, None

    peaks = np.maximum.accumulate(equity)
    dollar_drawdowns = equity - peaks
    if dollar_drawdowns.min() >= 0:
        return dollar_drawdowns, None, None

    trough_idx = int(np.argmin(dollar_drawdowns))
    peak_idx = int(np.argmax(equity[: trough_idx + 1]))
    return dollar_drawdowns, peak_idx, trough_idx


def mdd_period(total_value) -> tuple[int | None, int | None]:
    """Indices of the max dollar-drawdown window (peak → trough)."""
    _, peak_idx, trough_idx = _dollar_drawdown(total_value)
    return peak_idx, trough_idx


def max_drawdown(total_value) -> int:
    """
    Maximum dollar drawdown of total_value over time.
    Returns int dollar loss (e.g. -400), or 0 if none.
    """
    dollar_drawdowns, peak_idx, _ = _dollar_drawdown(total_value)
    if peak_idx is None:
        return 0
    return int(dollar_drawdowns.min())
