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


if __name__ == "__main__":
    tickers = {
        "BTC": {"tickers": "BTC-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "ETH": {"tickers": "ETH-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "XRP": {"tickers": "XRP-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "BNB": {"tickers": "BNB-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "ADA": {"tickers": "ADA-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "SOL": {"tickers": "SOL-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "DOGE": {"tickers": "DOGE-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "DOT": {"tickers": "DOT-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "LTC": {"tickers": "LTC-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "LINK": {"tickers": "LINK-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "XLM": {"tickers": "XLM-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "XMR": {"tickers": "XMR-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "TRX": {"tickers": "TRX-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "ETC": {"tickers": "ETC-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "BCH": {"tickers": "BCH-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "ATOM": {"tickers": "ATOM-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "ALGO": {"tickers": "ALGO-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "VET": {"tickers": "VET-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "EOS": {"tickers": "EOS-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "XTZ": {"tickers": "XTZ-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "DASH": {"tickers": "DASH-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "ZEC": {"tickers": "ZEC-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "MIOTA": {"tickers": "MIOTA-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "MKR": {"tickers": "MKR-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "AVAX": {"tickers": "AVAX-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "THETA": {"tickers": "THETA-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "LUNC": {"tickers": "LUNC-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "USTC": {"tickers": "USTC-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "FTT": {"tickers": "FTT-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "CEL": {"tickers": "CEL-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "SRM": {"tickers": "SRM-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "XEM": {"tickers": "XEM-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "NEO": {"tickers": "NEO-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "OMG": {"tickers": "OMG-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "BTG": {"tickers": "BTG-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "BSV": {"tickers": "BSV-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "WAVES": {"tickers": "WAVES-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "STEEM": {"tickers": "STEEM-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "MAID": {"tickers": "MAID-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "REP": {"tickers": "REP-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "XVG": {"tickers": "XVG-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "BCN": {"tickers": "BCN-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "NXT": {"tickers": "NXT-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "PPC": {"tickers": "PPC-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "NMC": {"tickers": "NMC-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "DGB": {"tickers": "DGB-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "SC": {"tickers": "SC-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "QTUM": {"tickers": "QTUM-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "ICX": {"tickers": "ICX-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
        "LSK": {"tickers": "LSK-USD", "start_date": "2012-01-01", "end_date": "2026-10-25"},
    }
    dfs = loader(tickers)
    close = dfs["Close"]
    close.to_csv("data/test_crypto.csv")
    print("Data saved to data/test_crypto.csv")
