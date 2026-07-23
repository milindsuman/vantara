from pathlib import Path
from fastapi import FastAPI, HTTPException
import pandas as pd

from src import segmentation
from src import forecasting
from src import recommendation
import requests

DATA_URLS = {
    "recommendation_ready.csv": "https://github.com/milindsuman/vantara/releases/download/v1.0/recommendation_ready.csv",
    "segmentation_ready.csv": "https://github.com/milindsuman/vantara/releases/download/v1.0/segmentation_ready.csv",
    "forecasting_ready.csv": "https://github.com/milindsuman/vantara/releases/download/v1.0/forecasting_ready.csv",
}

def ensure_data_file(path: Path, url: str):
    if not path.exists() or path.stat().st_size < 1000:  # catches leftover LFS pointers too
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

app = FastAPI(title="Vantara API")

state = {}


@app.on_event("startup")
def load_models():
    for filename, url in DATA_URLS.items():
        ensure_data_file(DATA_DIR / filename, url)
    seg_result = segmentation.run_segmentation(
        input_path=DATA_DIR / 'segmentation_ready.csv',
        output_path=DATA_DIR / 'customer_segments.csv',
        k=5
    )
    state['segments'] = pd.read_csv(DATA_DIR / 'customer_segments.csv')

    summary, als_model, matrix, customer_lookup, product_lookup, similarity, product_index_lookup, products = \
        recommendation.run_recommendation_pipeline(DATA_DIR / 'recommendation_ready.csv')

    state['als_model'] = als_model
    state['matrix'] = matrix
    state['customer_lookup'] = customer_lookup
    state['product_lookup'] = product_lookup
    state['customer_id_to_idx'] = {v: k for k, v in customer_lookup.items()}
    state['similarity'] = similarity
    state['product_index_lookup'] = product_index_lookup
    state['products'] = products
    state['description_lookup'] = recommendation.build_description_lookup(products)

    forecast_model, monthly_demand = forecasting.run_forecasting_pipeline(
        DATA_DIR / 'forecasting_ready.csv'
    )
    state['forecast_model'] = forecast_model
    state['monthly_demand'] = monthly_demand

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
    result = forecasting.get_forecast_for_product(
        state['forecast_model'], state['monthly_demand'], stock_code
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return result