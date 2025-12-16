import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

from ta import add_all_ta_features
from ta.utils import dropna

import numpy as np

from tinygrad import Tensor, dtypes

DATA_URL = "data_0.csv"

def download_data(url):
    df = pd.read_csv(
            url,
            skiprows=[1, 2],
            parse_dates=["Price"],
    )
    df = df.rename(columns={"Price": "Datetime"})
    return df

def plot_price(df, column_name="Close"):
    line_fig = px.line(df, x="Datetime", y=column_name)

    fig = sp.make_subplots(
            rows = 2, cols = 1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=["Price Over Time", "Correlation width " + column_name],
            row_heights=[0.7, 0.3]
    )
    for trace in line_fig.data: fig.add_trace(trace, row=1, col=1)

    correlation = df.corr()[column_name].drop(column_name)
    correlation = correlation.sort_values(ascending=False)
    bar_trace = go.Bar(
            x = correlation.index,
            y = correlation.values,
            marker=dict(color=correlation.values, colorscale='RdYlGn', showscale=True)
    )
    fig.add_trace(bar_trace, row=2, col=1)

    fig.update_xaxes(title_text="Datetime", row=1, col=1)
    fig.update_xaxes(title_text="Features", row=2, col=1)
    fig.update_yaxes(title_text=column_name, row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    
    fig.show()


def plot_nans(df):
    nan_counts = df.isna().sum()
    fig = px.bar(
            x = nan_counts.index,
            y = nan_counts.values,
            labels={'x': 'Columns', 'y': 'Number of NaN values'},
            title="Number of NaN Values per Column"
    )
    fig.show()


def drop_n_nans(df, n_nans=10):
    nan_counts = df.isna().sum()
    df = df.loc[:, nan_counts <= n_nans]
    return df

def drop_low_correlation_columns(df, target_column="Close", threshold=0.5):
    correlation_matrix = df.corr()
    target_corr = correlation_matrix[target_column]
    cols_to_keep = target_corr[(target_corr >= threshold) | (target_corr <= -threshold)].index
    return df[cols_to_keep]

def add_indicators(df):
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df = df.dropna(axis=1, how='all')
    return df

def datetime_to_categoric(df):
    df['day_of_week'] = df['Datetime'].dt.weekday  # Monday=0, Sunday=6
    df['hour'] = df['Datetime'].dt.hour  # Extract hour
    df['minute'] = df['Datetime'].dt.minute  # Extract minute
    return df

def normalize_dataset(df):
    numerical_features      = df.select_dtypes(include=['number']).columns.tolist()
    scaler                  = RobustScaler()
    df[numerical_features]  = scaler.fit_transform(df[numerical_features])
    return df, scaler
    
def create_sequence_data(df, target_col='Close', window_size=60, horizon=5):
    X, y        = [], []

    df          = df.drop(columns=["Datetime"])
    values      = df.values
    target_idx  = df.columns.get_loc(target_col)

    for i in range(len(df) - window_size - horizon):
        X.append(values[i:i+window_size, :])
        y.append(values[i+window_size:i+window_size+horizon, target_idx])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Data preprocessing
    data = download_data(DATA_URL)
    data = add_indicators(data)
    data = datetime_to_categoric(data)
    data = drop_n_nans(data, 100)
    data = data.iloc[91:]
    print(data.head())
    print(data.shape)
    data = data.drop(['volume_vwap', 'volume_mfi', 'volume_cmf', 'volatility_kcp'], axis=1)

    # First Plots
    plot_price(data)

    # Normalize Data
    data_normalized, scaler = normalize_dataset(data)
    plot_price(data_normalized)
    print(data.head())
    print(data.shape)

    # Create Sequence Data
    WINDOW_SIZE = 60
    HORIZON = 5
    X, y = create_sequence_data(
            data_normalized,
            window_size = WINDOW_SIZE,
            horizon     = HORIZON
    )
    print("THE DTYPE: ", X.dtype)
    print(X.shape)
    # print(X[0][0][1])
    # print(X[0][0][1].dtype)
    print(y.shape)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train = Tensor(X_train, dtype=dtypes.float32)
    y_train = Tensor(y_train, dtype=dtypes.float32)
    X_val   = Tensor(X_val, dtype=dtypes.float32)
    y_val = Tensor(y_val, dtype=dtypes.float32)
