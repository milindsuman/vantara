import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb


def load_forecasting_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['InvoiceDate'])
    return df


def build_monthly_demand(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

    monthly = df.groupby(['StockCode', 'YearMonth']).agg(
        Quantity=('Quantity', 'sum'),
        Revenue=('Price', lambda x: (x * df.loc[x.index, 'Quantity']).sum())
    ).reset_index()

    monthly['YearMonth'] = monthly['YearMonth'].dt.to_timestamp()
    return monthly


def engineer_features(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.copy()
    monthly = monthly.sort_values(['StockCode', 'YearMonth'])

    monthly['Month'] = monthly['YearMonth'].dt.month
    monthly['Year'] = monthly['YearMonth'].dt.year

    monthly['Quantity_lag1'] = monthly.groupby('StockCode')['Quantity'].shift(1)
    monthly['Quantity_lag2'] = monthly.groupby('StockCode')['Quantity'].shift(2)
    monthly['Quantity_rolling3'] = monthly.groupby('StockCode')['Quantity'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    monthly = monthly.dropna()
    return monthly


def time_based_split(monthly: pd.DataFrame, cutoff_date: str) -> tuple:
    cutoff = pd.Timestamp(cutoff_date)
    train = monthly[monthly['YearMonth'] < cutoff]
    test = monthly[monthly['YearMonth'] >= cutoff]
    return train, test


def get_feature_target(df: pd.DataFrame) -> tuple:
    features = ['Month', 'Year', 'Quantity_lag1', 'Quantity_lag2', 'Quantity_rolling3']
    X = df[features]
    y = df['Quantity']
    return X, y


def train_baseline(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(random_state=42, n_estimators=200, verbose=-1)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    naive_mae = mean_absolute_error(y_test, X_test['Quantity_lag1'])
    return {'mae': mae, 'rmse': rmse, 'naive_baseline_mae': naive_mae}


def run_forecasting(input_path: str, output_path: str, cutoff_date: str = '2011-10-01') -> dict:
    df = load_forecasting_data(input_path)
    monthly = build_monthly_demand(df)
    monthly = engineer_features(monthly)

    train, test = time_based_split(monthly, cutoff_date)

    X_train, y_train = get_feature_target(train)
    X_test, y_test = get_feature_target(test)

    baseline_model = train_baseline(X_train, y_train)
    baseline_results = evaluate(baseline_model, X_test, y_test)

    lgb_model = train_lightgbm(X_train, y_train)
    lgb_results = evaluate(lgb_model, X_test, y_test)

    monthly.to_csv(output_path, index=False)

    summary = {
        'train_rows': len(train),
        'test_rows': len(test),
        'linear_regression': baseline_results,
        'lightgbm': lgb_results
    }
    return summary


def train_final_model(monthly: pd.DataFrame) -> lgb.LGBMRegressor:
    X, y = get_feature_target(monthly)
    model = train_lightgbm(X, y)
    return model


def get_forecast_for_product(model, monthly: pd.DataFrame, stock_code: str) -> dict:
    product_data = monthly[monthly['StockCode'] == stock_code].sort_values('YearMonth')
    if product_data.empty:
        return None

    latest = product_data.iloc[-1]
    next_month = latest['YearMonth'] + pd.DateOffset(months=1)

    features = pd.DataFrame([{
        'Month': next_month.month,
        'Year': next_month.year,
        'Quantity_lag1': latest['Quantity'],
        'Quantity_lag2': latest['Quantity_lag1'],
        'Quantity_rolling3': np.mean([
            latest['Quantity'], latest['Quantity_lag1'], latest['Quantity_lag2']
        ])
    }])

    prediction = model.predict(features)[0]
    return {
        'stock_code': stock_code,
        'forecast_month': str(next_month.date()),
        'predicted_quantity': float(prediction)
    }


def run_forecasting_pipeline(input_path: str) -> tuple:
    df = load_forecasting_data(input_path)
    monthly = build_monthly_demand(df)
    monthly = engineer_features(monthly)
    model = train_final_model(monthly)
    return model, monthly


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

if __name__ == '__main__':
    result = run_forecasting(
        input_path=DATA_DIR / 'forecasting_ready.csv',
        output_path=DATA_DIR / 'monthly_demand_features.csv'
    )
    print(result)