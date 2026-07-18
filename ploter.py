import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet


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


def plot_portfolio(portfolio_invested, history):
    portfolio_invested.plot(
        history.index,
        history["Portfolio"],
        label="Portfolio ($)",
        alpha=0.5,
    )


def plot_invested(portfolio_invested, history):
    invested_plot = history["Cost_basis"].resample("ME").last()
    portfolio_invested.bar(
        invested_plot.index,
        invested_plot.values,
        width=20,
        alpha=0.1,
        label="Invested capital ($)",
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
    ma200 = history["Price"].rolling(200).mean()
    ax_price.plot(
        history.index,
        ma200,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label="200-day MA",
    )


def plot_take_profit(ax_price, history):
    # --- take-profits ---
    trigger_mask = history["Trigger_msg"].astype(str).ne("")
    trigger_dates = history.index[trigger_mask]

    if len(trigger_dates) > 0:
        # Highlight the last trigger date in red, others in green
        last_trigger_date = trigger_dates[-1]
        colors = ["red" if dt == last_trigger_date else "green" for dt in trigger_dates]

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
                text = f"p={int(profit)} r={int(returns)} {msg}"
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
            f"Return: {int(last_trigger_row['Returns'])}\\$"
        )
    else:
        last_sale_text = "Last sell: None"

    return last_sale_text


def title_with_metrics(metrics, last_sale_text):
    plt.title(f"<{metrics['Target']}> Cash_spent: {metrics['Cash_spent']}\\$ | Cash_return: {metrics['Cash_return']}\\$" +
                f"\n Profit: {metrics['Profit']}\\$ | Portfolio: {metrics['Portfolio']}\\$ | Banking: {metrics['Bank_profit']}\\$" +
                f"\n MDD: {metrics['MDD']}% ({metrics['MDD_usd']}$) | Bull %: {metrics['Bull_history']}% | Survival: {metrics['MA200_survival_days']:.0f}d" +
                f"\n {last_sale_text}")


def ploter(metrics, history, return_fig: bool = False):
    fig, ax_price = plt.subplots(figsize=(16, 8))

    plot_price(ax_price, history)
    plot_changepoints(ax_price, history)
    plot_average_price(ax_price, history)
    plot_ma200(ax_price, history)

    portfolio_invested = ax_price.twinx()
    plot_portfolio(portfolio_invested, history)
    plot_invested(portfolio_invested, history)

    plot_take_profit(ax_price, history)

    last_sale_text = last_sale_info(history)
    title_with_metrics(metrics, last_sale_text)

    # --- settings ---
    portfolio_invested.set_ylabel("Portfolio / Invested ($)")
    ax_price.set_ylabel("Asset price ($)")
    ax_price.set_xlabel("Date")

    y_min = history["Price"].min()
    y_max = history["Price"].max()
    ax_price.set_ylim(y_min * 0.95, y_max * 1.20)

    # --- legend ---
    l1, lab1 = ax_price.get_legend_handles_labels()
    l2, lab2 = portfolio_invested.get_legend_handles_labels()
    ax_price.legend(l1 + l2, lab1 + lab2, loc="upper left")

    plt.tight_layout()
    
    if return_fig:
        return fig
    
    plt.show()
