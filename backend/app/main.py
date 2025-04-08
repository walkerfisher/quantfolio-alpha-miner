from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .factor_model import run_factor_analysis
from .ml_model import predict_returns
from .ml_model import plot_return_predictions
from fastapi.responses import JSONResponse

app = FastAPI()

class StockRequest(BaseModel):
    tickers: List[str]
    start_date: str  # format: 'YYYY-MM-DD'
    end_date: str    # format: 'YYYY-MM-DD'

@app.post("/predict-returns")
def predict_returns_endpoint(request: StockRequest):
    try:
        return predict_returns(request.tickers, request.start_date, request.end_date)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/factor-analysis")
def factor_analysis(request: StockRequest):
    try:
        return run_factor_analysis(request.tickers, request.start_date, request.end_date)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/plot-returns")
def plot_returns(request: StockRequest):
    return plot_return_predictions(request.tickers, request.start_date, request.end_date)