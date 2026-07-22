import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_rfm(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    df = df.copy()
    df['Revenue'] = df['Quantity'] * df['Price']

    if reference_date is None:
        reference_date = df['InvoiceDate'].max()

    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('Revenue', 'sum')
    ).reset_index()

    return rfm


def scale_rfm(rfm: pd.DataFrame) -> tuple:
    features = rfm[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def find_optimal_k(scaled_data, k_range=range(2, 9)) -> dict:
    results = {'k': [], 'inertia': [], 'silhouette': []}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_data)
        results['k'].append(k)
        results['inertia'].append(km.inertia_)
        results['silhouette'].append(silhouette_score(scaled_data, labels))
    return results


def fit_kmeans(scaled_data, k: int) -> KMeans:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_data)
    return km


def label_segments(rfm: pd.DataFrame, cluster_labels) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm['Segment'] = cluster_labels
    return rfm


def profile_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    return rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(1)


def run_segmentation(input_path: str, output_path: str, k: int = 5) -> dict:
    df = pd.read_csv(input_path, parse_dates=['InvoiceDate'])

    rfm = compute_rfm(df)
    scaled, scaler = scale_rfm(rfm)

    km = fit_kmeans(scaled, k)
    rfm = label_segments(rfm, km.labels_)

    rfm.to_csv(output_path, index=False)

    summary = {
        'n_customers': len(rfm),
        'k': k,
        'silhouette_score': silhouette_score(scaled, km.labels_),
        'segment_profile': profile_segments(rfm).to_dict()
    }
    return summary


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# ... (all existing functions stay unchanged) ...

if __name__ == '__main__':
    result = run_segmentation(
        input_path=DATA_DIR / 'segmentation_ready.csv',
        output_path=DATA_DIR / 'customer_segments.csv',
        k=5
    )
    print(result)