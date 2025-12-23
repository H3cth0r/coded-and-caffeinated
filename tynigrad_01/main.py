import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

from ta import add_all_ta_features
from ta.utils import dropna

import numpy as np

from tinygrad import Tensor, dtypes, TinyJit

from tqdm import tqdm
import time

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

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size  # <--- required

        # Reset gate
        self.Wr = Tensor.randn(input_size, hidden_size, requires_grad=True)
        self.Ur = Tensor.randn(hidden_size, hidden_size, requires_grad=True)
        self.br = Tensor.zeros(hidden_size, requires_grad=True)

        # Update gate
        self.Wz = Tensor.randn(input_size, hidden_size, requires_grad=True)
        self.Uz = Tensor.randn(hidden_size, hidden_size, requires_grad=True)
        self.bz = Tensor.zeros(hidden_size, requires_grad=True)

        # Candidate hidden
        self.Wh = Tensor.randn(input_size, hidden_size, requires_grad=True)
        self.Uh = Tensor.randn(hidden_size, hidden_size, requires_grad=True)
        self.bh = Tensor.zeros(hidden_size, requires_grad=True)

    def parameters(self):
        return [
            self.Wr, self.Ur, self.br,
            self.Wz, self.Uz, self.bz,
            self.Wh, self.Uh, self.bh,
        ]

    def __call__(self, x, h_prev):
        r = (x @ self.Wr + h_prev @ self.Ur + self.br).sigmoid()
        z = (x @ self.Wz + h_prev @ self.Uz + self.bz).sigmoid()
        h_hat = (x @ self.Wh + (r * h_prev) @ self.Uh + self.bh).tanh()
        return (1 - z) * h_prev + z * h_hat

class GRUModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.gru    = GRUCell(input_size, hidden_size)
        self.Wo     = Tensor.randn(hidden_size, output_size)
        self.bo     = Tensor.zeros(output_size)

    def parameters(self):
        return self.gru.parameters() + [self.Wo, self.bo]

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        h = Tensor.zeros(batch_size, self.gru.hidden_size)

        for t in range(seq_len):
            h = self.gru(x[:, t, :], h)

        # Output multi-step forecast
        y_hat = h @ self.Wo + self.bo
        return y_hat

@TinyJit
def train_step(model, xb, yb, lr):
    preds = model(xb)
    loss = ((preds - yb) ** 2).mean()
    loss.backward()

    # Safe update
    for p in model.parameters():
        if p.grad is not None:
            p -= lr * p.grad
            p.grad = None

    return loss

@TinyJit
def train_epoch(model, X, y, lr, batch_size, steps_per_epoch):
    total_loss = Tensor.zeros(())

    for step in range(steps_per_epoch):
        i = step * batch_size
        xb = Tensor(X_train[i:i+batch_size].numpy(), dtype=dtypes.float32)
        yb = Tensor(y_train[i:i+batch_size].numpy(), dtype=dtypes.float32)

        preds = model(xb)
        loss = ((preds - yb) ** 2).mean()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad = None

        total_loss += loss

    return total_loss / steps_per_epoch

@TinyJit
def train_epoch(model, X, y, lr, batch_size, steps_per_epoch):
    total_loss = Tensor.zeros(())

    for step in range(steps_per_epoch):
        i = step * batch_size
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]

        preds = model(xb)
        loss = ((preds - yb) ** 2).mean()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad = None

        total_loss += loss

    return total_loss / steps_per_epoch


def jit_warmup(model, X_train, y_train, lr, batch_size):
    print("🔧 JIT warm-up...")

    # VERY small fixed workload
    warmup_steps = 2
    _ = train_epoch(
        model,
        X_train,
        y_train,
        lr,
        batch_size,
        warmup_steps
    ).item()

    print("JIT: 1 \n")

    # Second call ensures compilation
    _ = train_epoch(
        model,
        X_train,
        y_train,
        lr,
        batch_size,
        warmup_steps
    ).item()

    print("JIT: 2 \n")

    _ = train_epoch(
        model,
        X_train,
        y_train,
        lr,
        batch_size,
        warmup_steps
    ).item()

    print("✅ JIT compiled\n")

def evaluate(model, X_val, y_val, batch_size=64):
    losses = []

    for i in range(0, len(X_val), batch_size):
        xb = X_val[i:i+batch_size]
        yb = y_val[i:i+batch_size]

        preds = model(xb)
        loss = ((preds - yb) ** 2).mean()
        losses.append(loss.item())

    return float(np.mean(losses))

def train(model, X_train, y_train, X_val, y_val,
          epochs=30, lr=1e-3, batch_size=64):

    n_train = len(X_train)
    steps_per_epoch = n_train // batch_size

    print("======================================")
    print("Training GRU Model with TinyJit")
    print(f"Train samples : {n_train}")
    print(f"Val samples   : {len(X_val)}")
    print(f"Batch size   : {batch_size}")
    print(f"Steps/epoch  : {steps_per_epoch}")
    print("======================================\n")

    jit_warmup(model, X_train, y_train, lr, batch_size)

    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        epoch_start = time.time()

        train_loss = train_epoch(
            model,
            X_train,
            y_train,
            lr,
            batch_size,
            steps_per_epoch
        ).item()

        val_loss = evaluate(model, X_val, y_val, batch_size)

        pbar.set_postfix({
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "time": f"{time.time() - epoch_start:.2f}s"
        })

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
    data = drop_low_correlation_columns(data, target_column="Close", threshold=0.3)

    # First Plots
    plot_price(data)

    # Normalize Data
    data_normalized, scaler = normalize_dataset(data)
    plot_price(data_normalized)
    print(data.head())
    print(data.shape)

    # Create Sequence Data
    WINDOW_SIZE = 30
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


    # Model Creation
    INPUT_SIZE  = X_train.shape[2]
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = HORIZON

    model = GRUModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=30,
        lr=1e-3,
        batch_size=64
    )

    last_window = X_val[-1:].realize()
    prediction = model(last_window)

    print("Predicted next 5 steps:", prediction.numpy())
