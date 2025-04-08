import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_factor_analysis(tickers, start, end):
    # Download price data
    price_data = yf.download(tickers, start=start, end=end)['Adj Close']
    returns = price_data.pct_change().dropna()
    portfolio_returns = returns.mean(axis=1)

    # Load Fama-French 3-factor data
    ff_factors = pd.read_csv(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
        skiprows=3, index_col=0
    )
    ff_factors = ff_factors[~ff_factors.index.str.contains('^\\s*$', regex=True)]
    ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m%d')
    ff_factors = ff_factors.loc[start:end]
    ff_factors = ff_factors.astype(float) / 100

    # Align factor data with portfolio returns
    aligned = pd.concat([portfolio_returns, ff_factors], axis=1).dropna()
    aligned.columns = ['Portfolio', 'Mkt-RF', 'SMB', 'HML', 'RF']
    excess_returns = aligned['Portfolio'] - aligned['RF']

    # Run factor regression
    X = aligned[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    y = excess_returns
    model = sm.OLS(y, X).fit()

    result = {
        "coefficients": model.params.to_dict(),
        "r_squared": model.rsquared,
        "summary": model.summary().as_text()
    }
    return result