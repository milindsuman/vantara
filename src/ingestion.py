from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

NON_PRODUCT_CODES = [
    'POST', 'DOT', 'M', 'C2', 'D', 'S',
    'BANK CHARGES', 'ADJUST', 'AMAZONFEE', 'PADS'
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['InvoiceDate'])
    return df


def flag_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_cancellation'] = df['Invoice'].astype(str).str.startswith('C')
    df['is_bad_price'] = df['Price'] <= 0
    df['is_non_product'] = df['StockCode'].isin(NON_PRODUCT_CODES)
    df['is_missing_customer'] = df['CustomerID'].isnull()
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def build_segmentation_view(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (~df.is_cancellation) &
        (~df.is_bad_price) &
        (~df.is_non_product) &
        (~df.is_missing_customer)
    )
    return df[mask].copy()


def build_forecasting_view(df: pd.DataFrame) -> pd.DataFrame:
    mask = (~df.is_bad_price) & (~df.is_non_product)
    return df[mask].copy()


def build_recommendation_view(df: pd.DataFrame) -> pd.DataFrame:
    return build_segmentation_view(df)


def run_pipeline(raw_path, output_dir) -> dict:
    df = load_raw(raw_path)
    n_raw = len(df)

    df = flag_quality_issues(df)
    df = drop_duplicates(df)
    n_after_dedup = len(df)

    seg_df = build_segmentation_view(df)
    fc_df = build_forecasting_view(df)
    rec_df = build_recommendation_view(df)

    seg_df.to_csv(Path(output_dir) / 'segmentation_ready.csv', index=False)
    fc_df.to_csv(Path(output_dir) / 'forecasting_ready.csv', index=False)
    rec_df.to_csv(Path(output_dir) / 'recommendation_ready.csv', index=False)

    return {
        'raw_rows': n_raw,
        'after_dedup': n_after_dedup,
        'segmentation_rows': len(seg_df),
        'forecasting_rows': len(fc_df),
        'recommendation_rows': len(rec_df),
    }


if __name__ == '__main__':
    result = run_pipeline(
        raw_path=DATA_DIR / 'merged_raw.csv',
        output_dir=DATA_DIR
    )
    print(result)