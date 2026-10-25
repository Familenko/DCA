import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from utils.mdd import mdd_period


def detect_top_changepoints(prices, changepoint_n=3):
    prices.index = pd.to_datetime(prices.index)
    df = pd.DataFrame({"ds": prices.index, "y": prices.values})

    m = Prophet()
    m.fit(df)

    cps = m.changepoints
    deltas = m.params['delta'].mean(0)

    if len(cps) == 0:
        return []

    cp_df = pd.DataFrame({"date": cps, "strength": deltas})

    cp_df = cp_df.reindex(cp_df["strength"].abs().sort_values(ascending=False).index)
    top3 = cp_df.sort_values("date").tail(changepoint_n)

    results = []
    for _, row in top3.iterrows():
        mask = prices.index >= row["date"]
        if not mask.any():
            continue
        closest_idx = prices.index[mask][0]
        results.append(prices.iloc[prices.index.get_loc(closest_idx)])

    return results


def plot_changepoints(ax_price, history):
    top_changepoints = detect_top_changepoints(history["Price"], changepoint_n=3)
    if top_changepoints:
        for cp in top_changepoints:
            ax_price.axhline(
                y=cp,
                color="orange",
                linestyle="--",
                linewidth=1.0,
                alpha=1.0,
                label="Top changepoint" if cp == top_changepoints[0] else None,
            )


def plot_price(ax_price, history):
    ax_price.plot(
        history.index,
        history["Price"],
        color="orange",
        label="Asset price",
        linewidth=1.0,
    )


def plot_value(ax_bottom, history):
    # Realized cash = Total_value - Portfolio (area between the two)
    ax_bottom.fill_between(
        history.index,
        history["Portfolio"],
        history["Total_value"],
        color="green",
        alpha=0.05,
        label="Realized cash ($)",
        step="mid",
    )
    ax_bottom.plot(
        history.index,
        history["Total_value"],
        color="green",
        linewidth=0.5,
        alpha=0.5,
    )


def plot_portfolio(ax_bottom, history):
    # Money still in open position (0 → Portfolio)
    ax_bottom.fill_between(
        history.index,
        0,
        history["Portfolio"],
        color="red",
        alpha=0.05,
        label="Actual portfolio ($)",
        step="mid",
    )
    ax_bottom.plot(
        history.index,
        history["Portfolio"],
        color="red",
        linewidth=0.5,
        alpha=0.5,
    )


def plot_cash_spent(ax_bottom, history):
    ax_bottom.plot(
        history.index,
        history["Cash_spent"],
        color="green",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label="Invested capital ($)",
    )


def plot_mdd(ax_bottom, history):
    peak_idx, trough_idx = mdd_period(history["Total_value"])
    if peak_idx is None:
        return

    segment = history["Total_value"].iloc[peak_idx : trough_idx + 1]
    ax_bottom.plot(
        segment.index,
        segment.values,
        color="red",
        linewidth=2.0,
        alpha=0.9,
        zorder=5,
        label="MDD period",
    )


def plot_invested(ax_bottom, history):
    ax_bottom.plot(
        history.index,
        history["Cost_basis"],
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label="Nominal portfolio ($)",
    )


def plot_average_price(ax_price, history):
    ax_price.plot(
        history.index,
        history["Average_price"],
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label="Average price",
    )


def plot_ma200(ax_price, history):
    price = history["Price"]
    ma200 = price.rolling(200).mean()
    valid = ma200.notna()
    y_top = price.max() * 1.20
    y_bot = price.min() * 0.95

    ax_price.fill_between(
        history.index[valid],
        ma200[valid],
        y_top,
        color="green",
        alpha=0.05,
        label="Above MA200",
    )
    ax_price.fill_between(
        history.index[valid],
        y_bot,
        ma200[valid],
        color="red",
        alpha=0.05,
        label="Below MA200",
    )


def plot_take_profit(ax_price, history):
    # --- take-profits ---
    msgs = history["Trigger_msg"].astype(str)
    trigger_mask = msgs.ne("")
    trigger_dates = history.index[trigger_mask]

    if len(trigger_dates) > 0:
        buy_trigger = msgs.loc[trigger_dates].str.startswith("Buy")
        colors = ["red" if is_buy else "green" for is_buy in buy_trigger]

        # Scale marker sizes based on returns (larger for higher returns)
        sell_values = history.loc[trigger_dates, "Returns"].clip(lower=0).astype(float)
        min_size, max_size = 10, 100
        if sell_values.max() > sell_values.min():
            normalized = (sell_values - sell_values.min()) / (sell_values.max() - sell_values.min())
            sizes = (min_size + normalized * (max_size - min_size)).tolist()
        else:
            sizes = [60] * len(trigger_dates)

        # Scatter plot for take-profit triggers
        ax_price.scatter(
            trigger_dates,
            history.loc[trigger_dates, "Price"],
            color=colors,
            marker="o",
            s=sizes,
            zorder=5,
            label="Take-profit",
        )

        # Annotate each trigger with profit and return info
        for dt in trigger_dates:
            profit = history.loc[dt, "Profit"]
            returns = history.loc[dt, "Returns"]
            msg = history.loc[dt, "Trigger_msg"]
            price = history.loc[dt, "Price"]

            if returns > 0:
                text = f"{msg} p={int(profit)} r={int(returns)}"
            else:
                text = msg
            ax_price.annotate(
                text,
                xy=(dt, price),
                xytext=(0, 8),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color="black",
                rotation=90
            )


def last_sale_info(history):
    trigger_mask = history["Trigger_msg"].astype(str).ne("")

    if trigger_mask.any():
        last_trigger_date = history.index[trigger_mask][-1]
        last_trigger_row = history.loc[last_trigger_date]

        last_sale_text = (
            f"Last sell: {last_trigger_date.strftime('%Y-%m-%d')} | "
            f"Profit: {int(last_trigger_row['Profit'])}\\$ | "
            f"Return: {int(last_trigger_row['Returns'])}\\$ | "
            f"Price: {int(last_trigger_row['Price'])}\\$"
        )
    else:
        last_sale_text = "Last sell: None"

    return last_sale_text


def title_with_metrics(ax_price, metrics, last_sale_text):
    title = (
        f"<{metrics['Target']}> Cash_spent: {metrics['Cash_spent']}\\$ | Total_value: {metrics['Total_value']}\\$"
        f"\n Portfolio: {metrics['Portfolio']}\\$ | Invested: {metrics['Invested']}\\$ | Realized_cash: {metrics['Realized_cash']}\\$"
        f"\n MDD: {metrics['MDD']}$ | Bull %: {metrics['Bull_history']}% | Survival: {metrics['Survival']:.0f}d | RoAC: {metrics['RoAC']}%"
        f"\n {last_sale_text}"
    )
    ax_price.text(
        0.5, 0.98, title,
        transform=ax_price.transAxes,
        ha="center", va="top",
        fontsize=10,
    )


def ploter(metrics, history, return_fig: bool = False):
    fig, (ax_price, ax_bottom) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    plot_price(ax_price, history)
    plot_changepoints(ax_price, history)
    plot_average_price(ax_price, history)
    plot_ma200(ax_price, history)
    plot_take_profit(ax_price, history)

    plot_portfolio(ax_bottom, history)
    plot_invested(ax_bottom, history)
    plot_value(ax_bottom, history)
    plot_cash_spent(ax_bottom, history)
    plot_mdd(ax_bottom, history)

    # --- settings ---
    ax_price.set_ylabel("Asset price ($)")
    ax_price.grid(True, linestyle="--", alpha=0.5)
    ax_bottom.set_ylabel("Portfolio / Invested ($)")
    ax_bottom.set_xlabel("Date")
    ax_bottom.grid(True, linestyle="--", alpha=0.5)

    y_min = history["Price"].min()
    y_max = history["Price"].max()
    ax_price.set_ylim(y_min * 0.95, y_max * 1.20)

    last_sale_text = last_sale_info(history)
    title_with_metrics(ax_price, metrics, last_sale_text)

    # --- legend ---
    ax_price.legend(loc="upper left")
    ax_bottom.legend(loc="upper left")

    plt.tight_layout()

    if return_fig:
        return fig

    plt.show()
