import yfinance as yf
import pandas as pd

def download_close_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)

    if len(tickers) == 1:
        prices = df['Close'].to_frame(name=tickers[0])
    else:
        prices = pd.concat([df[ticker]['Close'] for ticker in tickers], axis=1)
        prices.columns = tickers

    return prices.dropna()
