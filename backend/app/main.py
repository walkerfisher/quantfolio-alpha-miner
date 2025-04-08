from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .factor_model import run_factor_analysis

app = FastAPI()

class StockRequest(BaseModel):
    tickers: List[str]
    start_date: str  # format: 'YYYY-MM-DD'
    end_date: str    # format: 'YYYY-MM-DD'

@app.post("/factor-analysis")
def factor_analysis(request: StockRequest):
    return run_factor_analysis(request.tickers, request.start_date, request.end_date)