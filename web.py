from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Optional
import yaml

import yfinance as yf
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from app import BacktestDCA
from ploter import ploter
from configs import web_config as cfg


with open("configs/variables.yaml", "r", encoding="utf-8") as f:
    VARIABLES = yaml.safe_load(f)


@st.cache_data(ttl=3600, show_spinner=True)
def _fetch_asset_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)


def load_asset_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.Series:
    with st.spinner(f"Fetching {ticker_symbol}..."):
        data = _fetch_asset_data(ticker_symbol, start_date, end_date)
    
    if data.empty:
        raise ValueError(f"No data found for {ticker_symbol}")
    
    if isinstance(data, pd.DataFrame):
        return data["Close"].squeeze()
    
    return data


def main() -> None:
    st.set_page_config(page_title="DCA Backtest", 
                       page_icon="🇺🇦",
                       layout="wide", 
                       initial_sidebar_state="expanded",
                       menu_items=cfg.MENU_ITEMS
                       )

    st.title("DCA Backtest Web App")

    with st.sidebar:
        st.header("Settings")

        asset_type = st.radio("Asset Type", list(cfg.ASSETS.keys()), horizontal=True)
        assets = cfg.ASSETS[asset_type]

        target = st.selectbox("Asset", assets)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", value=pd.to_datetime(cfg.YEAR_AGO))
        with col2:
            end_date = st.date_input("End date", value=pd.to_datetime(cfg.TODAY))

        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
            return

        st.divider()
        
        buy_amount = st.number_input("Buy amount", min_value=1.0, value=10.0, step=1.0)
        freq = st.selectbox("Frequency", VARIABLES["freq_multiplier"], index=0)
        minimum_profit = st.number_input("Minimum profit", min_value=1.0, value=1.2, step=0.1)
        minimum_loss = st.number_input("Minimum loss", max_value=1.0, value=0.8, step=0.1)

        use_fixed_sell_fraction = st.checkbox("Manual sell fraction", value=False)
        manual_sell_fraction: Optional[float] = None
        if use_fixed_sell_fraction:
            manual_sell_fraction = st.number_input("Sell fraction", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        
        if not use_fixed_sell_fraction:
            enable_sell = st.checkbox("Enable selling", value=True)
        else:
            enable_sell = True

        advanced_settings = st.expander("Advanced settings", expanded=False)
        with advanced_settings:
            cooldown_days = st.number_input("Cooldown days", min_value=1, max_value=365, value=30, step=1)
            cooldown_wait = st.number_input("Cooldown wait", min_value=1, max_value=365, value=7, step=1)
            fee = st.number_input("Fee %", min_value=0.0, max_value=1.0, value=0.01, step=0.001)
            max_invest_years = st.number_input("Max invest years", min_value=2, max_value=20, value=5, step=1)

        run_clicked = st.button("Run backtest", type="primary")

    st.info(
        f"**Asset**: {target} | "
        f"**Period**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    if not run_clicked:
        st.info(cfg.INSTRUCTIONS)
        return

    try:
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        prices = load_asset_data(target, start_date_str, end_date_str)

        strategy = {
                "buy_amount": float(buy_amount),
                "freq": freq,
                "fee": float(fee),
                "minimum_profit": float(minimum_profit),
                "minimum_loss": float(minimum_loss),
                "cooldown_days": int(cooldown_days),
                "cooldown_wait": int(cooldown_wait),
                "manual_sell_fraction": manual_sell_fraction,
                "enable_sell": bool(enable_sell),
                "threshold_invest_years": int(max_invest_years),
            }

        bt = BacktestDCA(
        target=target,
        prices=prices,
        strategy=strategy,
        )
        history, metrics = bt.run()

    except ValueError as exc:
        st.error(f"ValueError: {str(exc)}")
        return
    except (TypeError, KeyError, ZeroDivisionError) as exc:
        st.exception(f"{type(exc).__name__}: {str(exc)}")
        return
    
    metric_cash_spent = metrics.get("Cash_spent", 0.0)
    metric_value = metrics.get("Value", 0.0)
    metric_portfolio = metrics.get("Portfolio", 0.0)
    metric_extra_cash = metrics.get("Extra_cash", 0.0)

    trigger_mask = history["Trigger_msg"].astype(str).ne("")

    if trigger_mask.any():
        last_trigger_date = history.index[trigger_mask][-1]
        last_trigger_row = history.loc[last_trigger_date]

        last_sale_text = (
            f"Date: {last_trigger_date.strftime('%Y-%m-%d')} | "
            f"Profit: \\${int(last_trigger_row['Profit'])} | "
            f"Return: \\${int(last_trigger_row['Returns'])}"
        )
    else:
        last_sale_text = "No sales triggered"

    st.subheader("Metrics")
    st.markdown(
        f"""
        - Spend: \\${metric_cash_spent:,.0f} | Value: \\${metric_value:,.0f}
        - Portfolio: \\${metric_portfolio:,.0f} | Extra cash: \\${metric_extra_cash:,.0f}
        """
    )

    st.subheader("Last sell details")
    st.markdown(last_sale_text)

    st.subheader("Backtest Chart")
    fig = ploter(metrics=metrics, history=history, return_fig=True)
    st.pyplot(fig, width="stretch")

    trigger_rows = history.loc[trigger_mask].copy()
    if not trigger_rows.empty:
        st.subheader("Take-profit events")
        st.dataframe(
            trigger_rows[["Price", "Profit", "Returns", "Trigger_msg"]].sort_index(ascending=False),
            width="stretch",
        )
    else:
        st.info("No take-profit events for selected period/parameters.")

    st.subheader("History table")
    st.dataframe(history.sort_index(ascending=False), width="stretch")


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        )
    main()
