# Quantfolio Alpha Miner

Quantfolio Alpha Miner is a full-stack quantitative investing toolkit designed to analyze portfolios using machine learning and factor-based finance models. The backend is powered by FastAPI and provides endpoints for return prediction, factor attribution, and chart generation — all ready for integration with a modern frontend.

## 🚀 Features

Machine Learning Return Prediction (/predict-returns)

Uses Random Forest Regressor with momentum and volatility features

Fama-French 3-Factor Regression (/factor-analysis)

Breaks down portfolio returns into market, size, and value exposures

Chart Generation (/plot-returns)

Returns a PNG plot of predicted vs actual returns

Clean Project Structure with modular utilities and exception handling

## 🛠️ Setup Instructions

###	1. Clone and Navigate

git clone https://github.com/your-username/quantfolio-alpha-miner.git
cd quantfolio-alpha-miner/backend

###	2. Create and Activate Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

###	3. Install Dependencies

pip install -r requirements.txt

###	4. Run FastAPI Server

uvicorn app.main:app --reload

Visit http://127.0.0.1:8000/docs to access the interactive Swagger UI.

## 📦 Example Request

### /predict-returns

{
  "tickers": ["AAPL", "MSFT", "GOOG"],
  "start_date": "2022-01-01",
  "end_date": "2023-01-01"
}

### /plot-returns

Returns an image of predicted vs. actual returns for the same input.

## 📁 Project Structure

quantfolio-alpha-miner/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── ml_model.py
│   │   ├── factor_model.py
│   │   ├── data_utils.py
│   ├── requirements.txt

## 🧠 Built With

### Python, FastAPI

### Pandas, NumPy, Scikit-Learn, Statsmodels

### yFinance, Matplotlib

## 💡 Author

Walker FisherFeel free to connect or fork — contributions welcome!