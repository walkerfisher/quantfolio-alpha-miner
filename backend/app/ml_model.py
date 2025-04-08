import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from .data_utils import download_close_prices


def predict_returns(tickers, start, end):
    prices = download_close_prices(tickers, start, end)
    returns = prices.pct_change().dropna()

    # Features
    momentum = returns.rolling(window=5).mean()
    volatility = returns.rolling(window=5).std()

    features = pd.DataFrame({
        'momentum': momentum.mean(axis=1),
        'volatility': volatility.mean(axis=1)
    }).dropna()

    target = returns.mean(axis=1).shift(-1).loc[features.index]

    # Clean
    features = features.loc[target.index]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    clean_preds = np.nan_to_num(predictions, nan=0.0).tolist()
    clean_actuals = np.nan_to_num(y_test, nan=0.0).tolist()

    return {
        "predictions": clean_preds,
        "actuals": clean_actuals,
        "feature_importances": model.feature_importances_.tolist()
    }
