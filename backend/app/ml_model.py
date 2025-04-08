import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import StreamingResponse
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

def plot_return_predictions(tickers, start, end):
    result = predict_returns(tickers, start, end)

    preds = np.array(result["predictions"])
    actuals = np.array(result["actuals"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(preds, label="Predicted", linestyle="--")
    ax.plot(actuals, label="Actual", alpha=0.7)
    ax.set_title("Predicted vs Actual Returns")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
