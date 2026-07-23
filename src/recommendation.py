from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import implicit

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'


def load_recommendation_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_interaction_matrix(df: pd.DataFrame) -> tuple:
    interactions = df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().reset_index()

    customer_ids = interactions['CustomerID'].astype('category')
    product_ids = interactions['StockCode'].astype('category')

    interactions['customer_idx'] = customer_ids.cat.codes
    interactions['product_idx'] = product_ids.cat.codes

    matrix = csr_matrix(
        (interactions['Quantity'], (interactions['customer_idx'], interactions['product_idx']))
    )

    customer_lookup = dict(enumerate(customer_ids.cat.categories))
    product_lookup = dict(enumerate(product_ids.cat.categories))

    return matrix, customer_lookup, product_lookup


def train_als(matrix: csr_matrix, factors: int = 50) -> implicit.als.AlternatingLeastSquares:
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=0.1, iterations=20)
    model.fit(matrix)
    return model


def get_cf_recommendations(model, matrix, customer_idx: int, product_lookup: dict, description_lookup: dict, n: int = 10) -> list:
    recommended = model.recommend(customer_idx, matrix[customer_idx], N=n)
    results = []
    for idx, score in zip(recommended[0], recommended[1]):
        stock_code = str(product_lookup[idx])
        results.append({
            "stock_code": stock_code,
            "description": description_lookup.get(stock_code, "Unknown product"),
            "score": float(score)
        })
    return results


def build_content_similarity(df: pd.DataFrame):
    products = df[['StockCode', 'Description']].drop_duplicates(subset='StockCode')
    products = products.dropna(subset=['Description'])

    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(products['Description'])

    similarity = cosine_similarity(tfidf_matrix)

    product_index_lookup = {code: i for i, code in enumerate(products['StockCode'])}
    return similarity, product_index_lookup, products


def build_description_lookup(products: pd.DataFrame) -> dict:
    return dict(zip(products['StockCode'], products['Description']))


def get_content_recommendations(stock_code: str, similarity, product_index_lookup: dict, products: pd.DataFrame, n: int = 10) -> list:
    if stock_code not in product_index_lookup:
        return []
    idx = product_index_lookup[stock_code]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return [(products.iloc[i]['StockCode'], score) for i, score in scores]


def run_recommendation_pipeline(input_path: str):
    df = load_recommendation_data(input_path)

    matrix, customer_lookup, product_lookup = build_interaction_matrix(df)
    als_model = train_als(matrix)

    similarity, product_index_lookup, products = build_content_similarity(df)

    joblib.dump(als_model, DATA_DIR / 'als_model.pkl')
    joblib.dump(matrix, DATA_DIR / 'interaction_matrix.pkl')
    joblib.dump(customer_lookup, DATA_DIR / 'customer_lookup.pkl')
    joblib.dump(product_lookup, DATA_DIR / 'product_lookup.pkl')
    joblib.dump(similarity, DATA_DIR / 'similarity_matrix.pkl')
    joblib.dump(product_index_lookup, DATA_DIR / 'product_index_lookup.pkl')
    products.to_pickle(DATA_DIR / 'products.pkl')

    summary = {
        'n_customers': matrix.shape[0],
        'n_products': matrix.shape[1],
        'interaction_density': matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    }
    return summary, als_model, matrix, customer_lookup, product_lookup, similarity, product_index_lookup, products


if __name__ == '__main__':
    summary, *_ = run_recommendation_pipeline(DATA_DIR / 'recommendation_ready.csv')
    print(summary)