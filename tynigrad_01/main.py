import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

from ta import add_all_ta_features
from ta.utils import dropna

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

def create_sequence_data(df, window_size=1):
    X, y = [], []
    print(df[['Close', 'High']].values)

if __name__ == "__main__":
    data = download_data(DATA_URL)
    data = add_indicators(data)
    # data = data.iloc[1:]
    # print(data['others_dlr'].isna().sum())
    # data = drop_low_correlation_columns(data)
    data = datetime_to_categoric(data)
    data = drop_n_nans(data, 100)
    data = data.iloc[91:]
    print(data.head())
    print(data.shape)
    data = data.drop(['volume_vwap', 'volume_mfi', 'volume_cmf', 'volatility_kcp'], axis=1)


    plot_nans(data)
    plot_price(data)
    # create_sequence_data(data)
