import numpy as np
import pandas as pd
from dataclasses import dataclass
from lifelines import KaplanMeierFitter


@dataclass
class RegimeRun:
    state_is_above_ma: bool
    length_days: int
    is_completed: bool


def _extract_runs(states: pd.Series) -> list[RegimeRun]:
    """Розбиває булеву серію на послідовні відрізки однакового стану."""
    if states.empty:
        return []

    runs: list[RegimeRun] = []
    current_state = bool(states.iloc[0])
    length = 1

    for value in states.iloc[1:]:
        state_continues = bool(value) == current_state
        if state_continues:
            length += 1
        else:
            runs.append(RegimeRun(current_state, length, is_completed=True))
            current_state = bool(value)
            length = 1

    last_run_is_still_active = True
    runs.append(RegimeRun(current_state, length, is_completed=not last_run_is_still_active))
    return runs


def survival_ma200(prices: pd.Series, ma_window: int = 200) -> float:
    """
    Оцінює, скільки ще днів ціна залишатиметься у поточному режимі відносно MA.

    Режим — безперервний відрізок, коли ціна тримається вище або нижче MA.
    Використовує Kaplan-Meier, щоб коректно врахувати поточний незавершений відрізок.
    """
    close = prices.astype(float)
    ma = close.rolling(ma_window).mean()

    rows_where_ma_is_valid = ma.notna()
    min_required_rows = 30
    if rows_where_ma_is_valid.sum() < min_required_rows:
        return np.nan

    close_where_ma_valid = close[rows_where_ma_is_valid]
    ma_where_valid = ma[rows_where_ma_is_valid]
    price_is_above_ma = (close_where_ma_valid >= ma_where_valid).astype(bool)

    runs = _extract_runs(price_is_above_ma)
    if not runs:
        return np.nan

    current_run = runs[-1]
    current_regime_is_above = current_run.state_is_above_ma
    days_already_spent_in_current_regime = current_run.length_days

    runs_in_same_regime = [r for r in runs if r.state_is_above_ma == current_regime_is_above]
    run_durations = [r.length_days for r in runs_in_same_regime]
    run_ended = [int(r.is_completed) for r in runs_in_same_regime]

    # --- Kaplan-Meier та умовне очікування залишку ---
    kmf = KaplanMeierFitter()
    kmf.fit(durations=np.array(run_durations), event_observed=np.array(run_ended))

    longest_observed_duration = int(kmf.timeline.max())
    elapsed_clamped_to_observed = max(0, min(days_already_spent_in_current_regime, longest_observed_duration))

    survival_prob_at_elapsed = float(kmf.survival_function_at_times(elapsed_clamped_to_observed).iloc[0])
    has_valid_survival_prob = np.isfinite(survival_prob_at_elapsed) and survival_prob_at_elapsed > 0
    if not has_valid_survival_prob:
        return np.nan

    future_day_range = np.arange(elapsed_clamped_to_observed + 1, longest_observed_duration + 1)
    survival_probs_for_future_days = kmf.survival_function_at_times(future_day_range).values
    expected_remaining_days = float(survival_probs_for_future_days.sum() / survival_prob_at_elapsed)

    return expected_remaining_days
