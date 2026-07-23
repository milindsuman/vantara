from pathlib import Path
from fastapi import FastAPI, HTTPException
import pandas as pd


from src import forecasting
from src import recommendation
import requests

DATA_URLS = {
    "als_model.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/als_model.pkl",
    "customer_lookup.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/customer_lookup.pkl",
    "customer_segments.csv": "https://github.com/milindsuman/vantara/releases/download/v1.0/customer_segments.csv",
    "forecast_model.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/forecast_model.pkl",
    "interaction_matrix.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/interaction_matrix.pkl",
    "monthly_demand.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/monthly_demand.pkl",
    "product_index_lookup.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/product_index_lookup.pkl",
    "product_lookup.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/product_lookup.pkl",
    "products.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/products.pkl",
    "scaler.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/scaler.pkl",
    "similarity_matrix.pkl": "https://github.com/milindsuman/vantara/releases/download/v1.0/similarity_matrix.pkl",
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

    import joblib

    state["segments"] = pd.read_csv(DATA_DIR / "customer_segments.csv")

    state["als_model"] = joblib.load(DATA_DIR / "als_model.pkl")
    state["matrix"] = joblib.load(DATA_DIR / "interaction_matrix.pkl")
    state["customer_lookup"] = joblib.load(DATA_DIR / "customer_lookup.pkl")
    state["product_lookup"] = joblib.load(DATA_DIR / "product_lookup.pkl")

    state["customer_id_to_idx"] = {
        v: k for k, v in state["customer_lookup"].items()
    }

    state["similarity"] = joblib.load(DATA_DIR / "similarity_matrix.pkl")
    state["product_index_lookup"] = joblib.load(DATA_DIR / "product_index_lookup.pkl")
    state["products"] = pd.read_pickle(DATA_DIR / "products.pkl")

    state["description_lookup"] = recommendation.build_description_lookup(
        state["products"]
    )

    state["forecast_model"] = joblib.load(DATA_DIR / "forecast_model.pkl")
    state["monthly_demand"] = pd.read_pickle(DATA_DIR / "monthly_demand.pkl")

    print("All pretrained models loaded.")


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