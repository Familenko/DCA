import yfinance as yf
import pandas as pd
import time


def validate_input(tickers: dict):
    """
    Validate the input tickers dictionary to ensure it has the correct structure and types.

    Args:
        tickers (dict): A dictionary containing asset information.

    Example of expected structure:
        {
            "BTC": {"tickers": "BTC-USD", "start_date": "2024-02-13", "end_date": "2026-02-13"},
            "ETH": {"tickers": "ETH-USD", "start_date": "2024-02-13", "end_date": "2026-02-13"},
            ...
        }
    """
    for k, v in tickers.items():
        if not isinstance(v, dict):
            raise ValueError(f"Invalid value for {k}: {v}. Expected a dictionary.")
        if "tickers" not in v or "start_date" not in v or "end_date" not in v:
            raise ValueError(f"Missing required keys for {k}: {v}. Expected keys: 'tickers', 'start_date', 'end_date'.")
        if not isinstance(v["tickers"], str):
            raise ValueError(f"Invalid tickers value for {k}: {v['tickers']}. Expected a string.")
        if not isinstance(v["start_date"], str) or not isinstance(v["end_date"], str):
            raise ValueError(f"Invalid date values for {k}: start_date={v['start_date']}, end_date={v['end_date']}. Expected strings.")
        

def validate_output(dfs: dict):
    """
    Validate the output dataframes to ensure they contain the expected data.
    Args:
        dfs (dict): A dictionary containing dataframes for each field.
    Example of expected structure:
        {
            "Close": pd.DataFrame(index=pd.DatetimeIndex(...), columns=["BTC", "ETH", ...]),
            "Volume": pd.DataFrame(index=pd.DatetimeIndex(...), columns=["BTC", "ETH", ...])
            ...
        }
    """
    for field, df in dfs.items():
        if df.empty:
            raise ValueError(f"Output DataFrame for field '{field}' is empty.")


def loader(tickers, fields=("Close", "Volume")):
    validate_input(tickers)

    dfs = {field: pd.DataFrame() for field in fields}

    for k, v in tickers.items():
        print(f"Loading data for {k} from {v['start_date']} to {v['end_date']}...")
        raw = yf.download(
            v["tickers"],
            start=v["start_date"],
            end=v["end_date"],
            auto_adjust=False,
        )
        for field in fields:
            if field in raw.columns:
                dfs[field][k] = raw[field]
        time.sleep(1)

    validate_output(dfs)

    return dfs
