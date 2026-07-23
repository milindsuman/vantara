from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd

from src import segmentation
from src import recommendation

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

app = FastAPI(title="Vantara API")

state = {}


@app.on_event("startup")
def load_models():
    state['segments'] = pd.read_csv(DATA_DIR / 'customer_segments.csv')

    state['als_model'] = joblib.load(DATA_DIR / 'als_model.pkl')
    state['matrix'] = joblib.load(DATA_DIR / 'interaction_matrix.pkl')
    state['customer_lookup'] = joblib.load(DATA_DIR / 'customer_lookup.pkl')
    state['product_lookup'] = joblib.load(DATA_DIR / 'product_lookup.pkl')
    state['customer_id_to_idx'] = {v: k for k, v in state['customer_lookup'].items()}
    state['similarity'] = joblib.load(DATA_DIR / 'similarity_matrix.pkl')
    state['product_index_lookup'] = joblib.load(DATA_DIR / 'product_index_lookup.pkl')
    state['products'] = pd.read_pickle(DATA_DIR / 'products.pkl')
    state['description_lookup'] = recommendation.build_description_lookup(state['products'])

    state['forecast_model'] = joblib.load(DATA_DIR / 'forecast_model.pkl')
    state['monthly_demand'] = pd.read_pickle(DATA_DIR / 'monthly_demand.pkl')

    print("All models loaded and ready.")


@app.get("/")
def root():
    return {"status": "Vantara API is running"}


@app.get("/segment/{customer_id}")
def get_segment(customer_id: float):
    segments = state['segments']
    row = segments[segments['CustomerID'] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    return row.iloc[0].to_dict()


@app.get("/recommend/{customer_id}")
def get_recommendations(customer_id: float, n: int = 5):
    customer_id_to_idx = state['customer_id_to_idx']
    if customer_id not in customer_id_to_idx:
        raise HTTPException(status_code=404, detail="Customer not found")

    idx = customer_id_to_idx[customer_id]
    recs = recommendation.get_cf_recommendations(
        state['als_model'], state['matrix'], idx, state['product_lookup'], state['description_lookup'], n=n
    )
    return {"customer_id": customer_id, "recommendations": recs}


@app.get("/similar-products/{stock_code}")
def get_similar_products(stock_code: str, n: int = 5):
    recs = recommendation.get_content_recommendations(
        stock_code, state['similarity'], state['product_index_lookup'], state['products'], n=n
    )
    if not recs:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"stock_code": stock_code, "similar_products": recs}


@app.get("/forecast/{stock_code}")
def get_forecast(stock_code: str):
    from src import forecasting
    result = forecasting.get_forecast_for_product(
        state['forecast_model'], state['monthly_demand'], stock_code
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return result