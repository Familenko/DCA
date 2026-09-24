from datetime import datetime


START_DATE = "2026-02-13"
TODAY = datetime.now().strftime("%Y-%m-%d")

# crypto
CRYPTO = {
    "BTC": {"tickers": "BTC-USD", "start_date": START_DATE, "end_date": TODAY},
    "ETH": {"tickers": "ETH-USD", "start_date": START_DATE, "end_date": TODAY},
    "BNB": {"tickers": "BNB-USD", "start_date": START_DATE, "end_date": TODAY},
    "XRP": {"tickers": "XRP-USD", "start_date": START_DATE, "end_date": TODAY},
    "SOL": {"tickers": "SOL-USD", "start_date": START_DATE, "end_date": TODAY},
    "DOT": {"tickers": "DOT-USD", "start_date": START_DATE, "end_date": TODAY},
    "ADA": {"tickers": "ADA-USD", "start_date": START_DATE, "end_date": TODAY},
    "LINK": {"tickers": "LINK-USD", "start_date": START_DATE, "end_date": TODAY},
    "AVAX": {"tickers": "AVAX-USD", "start_date": START_DATE, "end_date": TODAY}
}
crypto_buy_amount = 10
crypto_freq = "W-MON"

# stocks
STOCKS = {
    'SPY': {"tickers": "SPY", "start_date": START_DATE, "end_date": TODAY}
}
stock_buy_amount = 300
stock_freq = "WOM-1MON"